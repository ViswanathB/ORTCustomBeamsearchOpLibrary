#include "custom_beamsearch_op_library.h"

#include <vector>
#include <cmath>
#include <mutex>
#include <iostream>
#include <unordered_map>

#include "beam_search.h"
#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

#include "beam_search_parameters.h"

/* Returns the amount of milliseconds elapsed since the UNIX epoch. Works on both
 * windows and linux. */
uint64_t GetTimeMs64() {
#ifdef _WIN32
 /* Windows */
 FILETIME ft;
 LARGE_INTEGER li;

 /* Get the amount of 100 nano seconds intervals elapsed since January 1, 1601 (UTC) and copy it
  * to a LARGE_INTEGER structure. */
 GetSystemTimeAsFileTime(&ft);
 li.LowPart = ft.dwLowDateTime;
 li.HighPart = ft.dwHighDateTime;

 uint64_t ret = li.QuadPart;
 ret -= 116444736000000000LL; /* Convert from file time to UNIX epoch time. */
 ret /= 10000; /* From 100 nano seconds (10^-7) to 1 millisecond (10^-3) intervals */

 return ret;
#else
 /* Linux */
 struct timeval tv;

 gettimeofday(&tv, NULL);

 uint64_t ret = tv.tv_usec;
 /* Convert from micro seconds (10^-6) to milliseconds (10^-3) */
 ret /= 1000;

 /* Adds the seconds (10^0) after converting them to milliseconds (10^-3) */
 ret += (tv.tv_sec * 1000);

 return ret;
#endif
}

static const char* c_OpDomain = "test.beamsearchop";

struct OrtCustomOpDomainDeleter {
  explicit OrtCustomOpDomainDeleter(const OrtApi* ort_api) {
    ort_api_ = ort_api;
  }
  void operator()(OrtCustomOpDomain* domain) const {
    ort_api_->ReleaseCustomOpDomain(domain);
  }

  const OrtApi* ort_api_;
};

using OrtCustomOpDomainUniquePtr = std::unique_ptr<OrtCustomOpDomain, OrtCustomOpDomainDeleter>;
static std::vector<OrtCustomOpDomainUniquePtr> ort_custom_op_domain_container;
static std::mutex ort_custom_op_domain_mutex;

static void AddOrtCustomOpDomainToContainer(OrtCustomOpDomain* domain, const OrtApi* ort_api) {
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  auto ptr = std::unique_ptr<OrtCustomOpDomain, OrtCustomOpDomainDeleter>(domain, OrtCustomOpDomainDeleter(ort_api));
  ort_custom_op_domain_container.push_back(std::move(ptr));
}

struct OrtTensorDimensions : std::vector<int64_t> {
  OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue* value) {
    OrtTensorTypeAndShapeInfo* info = ort.GetTensorTypeAndShape(value);
    std::vector<int64_t>::operator=(ort.GetTensorShape(info));
    ort.ReleaseTensorTypeAndShapeInfo(info);
  }
};

struct CustomBeamsearchOpKernel {
  CustomBeamsearchOpKernel(OrtApi api, const OrtKernelInfo* info)
      : api_(api), ort_(api_) {
    // TODO move all the stuff to C++ API
    // InitAPI() can't be called - just set the global api_ that initapi() is setting to api passed in here
    // All other should be easy after
    //InitApiWith(api);
    model_path_ = nullptr;
#ifdef PRINT_TO_CONSOLE
    std::cout<<"Constructor called"<<std::endl;
#endif
    OrtStatus* status = api_.KernelInfoGetAttribute_string(info, "model_path", nullptr, &model_path_len_);
    if (status == nullptr && model_path_len_ > 0) {
#ifdef PRINT_TO_CONSOLE
      std::cout<<"Model exists with path length:"<<model_path_len_<<std::endl;
#endif
      char *c_model_path = new char[model_path_len_];
      status = api_.KernelInfoGetAttribute_string(info, "model_path", c_model_path, &model_path_len_);
      if (status != nullptr)
      {
        ORT_CXX_API_THROW("Couldn't find model_path attribute in CustomBeamsearchOpKernel Constructor", ORT_FAIL);
      }

      model_path_ = new wchar_t[model_path_len_];
      size_t afterconversionlen;
      mbstowcs_s(&afterconversionlen, model_path_, model_path_len_, (const char*) c_model_path, model_path_len_-1);

      delete c_model_path;
    }
    session_ = nullptr;

    parameters_.ParseFromAttributes(ort_, info);
    OrtOp* op_topk = InitTopK(info);

    if (op_topk == nullptr) {
      ORT_CXX_API_THROW("Couldn't create topk OP in CustomBeamsearchOpKernel Constructor", ORT_RUNTIME_EXCEPTION);
    }
    contrib_kernels_[std::string("topk")] = op_topk;

    OrtOp* op_softmax = InitLogSoftMax(info);
    if (op_softmax == nullptr) {
      ORT_CXX_API_THROW("Couldn't create softmax OP in CustomBeamsearchOpKernel Constructor", ORT_RUNTIME_EXCEPTION);
    }
    contrib_kernels_[std::string("softmax")] = op_softmax;    
  }

  OrtOp* InitLogSoftMax(const OrtKernelInfo* info) {
    const char* type_constraint_names[1] = {"T"};
    int type_constraint_values[1] = {1};

    int64_t axis_value = 1;
    OrtOpAttr* axis = ort_.CreateOpAttr("axis", &axis_value, 1, OrtOpAttrType::ORT_OP_ATTR_INT);

    if (!axis) {
      ORT_CXX_API_THROW("Failed to create attributes for InitLogSoftMax", ORT_RUNTIME_EXCEPTION);
    }

    OrtOpAttr* top_attrs[1] = {axis};
    OrtOp* op_lsm_ = ort_.CreateOp(info, "LogSoftmax", "", 13,
                  (const char**)type_constraint_names,
                  (const ONNXTensorElementDataType*)type_constraint_values,
                  1, top_attrs, 1, 1, 1);

    ort_.ReleaseOpAttr(axis);

    return op_lsm_;

  }

  OrtOp* InitTopK(const OrtKernelInfo* info) {
    const char* type_constraint_names[2] = {"T", "I"};
    int type_constraint_values[2] = {1, 7};

    // logits with num_beams will have the dimension: (batch_size, num_beams * vocab_size)
    // TopK will be extracted for each batch and the index would be [0, num_beams * vocab_size] for each batch
    // this is later converted in the range [0, vocab_size] for a particular beam, ProcessLogits() has this.
    int64_t axis_value = 1;
    OrtOpAttr* axis = ort_.CreateOpAttr("axis", &axis_value, 1, OrtOpAttrType::ORT_OP_ATTR_INT);

    int64_t largest_value = 1;  // return in ascending order
    OrtOpAttr* largest = ort_.CreateOpAttr("largest", &largest_value, 1, OrtOpAttrType::ORT_OP_ATTR_INT);

    int64_t sorted_value = 1;
    OrtOpAttr* sorted = ort_.CreateOpAttr("sorted", &sorted_value, 1, OrtOpAttrType::ORT_OP_ATTR_INT);

    if (!axis || !largest || !sorted) {
      ORT_CXX_API_THROW("Failed to create attributes for topk", ORT_RUNTIME_EXCEPTION);
    }

    OrtOpAttr* top_attrs[3] = {axis, largest, sorted};
    OrtOp* op_topk_ = ort_.CreateOp(info, "TopK", "", 14,
                  (const char**)type_constraint_names,
                  (const ONNXTensorElementDataType*)type_constraint_values,
                  2, top_attrs, 3, 2, 2);

    ort_.ReleaseOpAttr(axis);
    ort_.ReleaseOpAttr(largest);
    ort_.ReleaseOpAttr(sorted);

    return op_topk_;
  }

  void Compute(OrtKernelContext* context) {
    if (session_ == nullptr && model_path_ != nullptr) {
      uint64_t s_time = GetTimeMs64();
      // The first two arguments don't matter since we are not creating the env for the first
      // time. We would only access the existing env created for parent session with is triggering this
      // custom OP.
      api_.CreateEnv(OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO, "", &env_);

      OrtSessionOptions* sessionoptions;
      api_.CreateSessionOptions(&sessionoptions);

      OrtStatusPtr status = api_.CreateSession(env_, model_path_, sessionoptions, &session_);

      if (status != nullptr) {
        ORT_CXX_API_THROW("Unable to create internal session for beam", ORT_FAIL);
      }
      uint64_t e_time = GetTimeMs64();
      std::cout<<"Time taken for session creation:"<<e_time-s_time<<std::endl;
    }

    if (model_path_ != nullptr) {
      //TODO Pass all the ones that are coming from subgraph using a file or static json object
      parameters_.vocab_size = 50263;
      parameters_.num_heads = 16;
      parameters_.head_size = 64;
      parameters_.num_layers = 6;
      OrtStatusPtr status = custombsop::RunBeamSearchOnInternalSession(context, api_, ort_, session_, parameters_, contrib_kernels_);
      if (status != nullptr) {
        ORT_CXX_API_THROW("run internal session failed:", api_.GetErrorCode(status));
      }
    }

    for(auto& it: contrib_kernels_) {
      ort_.ReleaseOp(reinterpret_cast<OrtOp*>(it.second));
    }
  }

 private:
  OrtApi api_;  // keep a copy of the struct, whose ref is used in the ort_
  Ort::CustomOpApi ort_;
  OrtSession* session_;
  OrtEnv* env_;

  //Attributes
  wchar_t *model_path_;
  size_t model_path_len_;

  //Parameters
  custombsop::BeamSearchParameters parameters_;

  //Contrib Kernels
  std::unordered_map<std::string, OrtOp*> contrib_kernels_;
};

struct CustomBeamsearchOp : Ort::CustomOpBase<CustomBeamsearchOp, CustomBeamsearchOpKernel> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const {
    return new CustomBeamsearchOpKernel(api, info);
  };

  const char* GetName() const { return "CustomBeamsearchOp"; };
  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; };

  size_t GetInputTypeCount() const { 
  return 10;
  };

  ONNXTensorElementDataType GetInputType(size_t /*index*/index) const {
    switch (index) {
      case 0:
      case 1:
      case 2:
      case 3:
      case 4:
      case 8:
      case 9:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
      case 5:
      case 6:
      case 7:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
      default:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
  };

  size_t GetOutputTypeCount() const { 
    return 2;
  };

  ONNXTensorElementDataType GetOutputType(size_t /*index*/index) const {
    switch(index) {
      case 0:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
      case 1:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
      default:
         return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
  };

} c_CustomBeamsearchOp;

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
  OrtCustomOpDomain* domain = nullptr;
  const OrtApi* ortApi = api->GetApi(ORT_API_VERSION);

  if (auto status = ortApi->CreateCustomOpDomain(c_OpDomain, &domain)) {
    return status;
  }

  AddOrtCustomOpDomainToContainer(domain, ortApi);

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_CustomBeamsearchOp)) {
    return status;
  }

  return ortApi->AddCustomOpDomain(options, domain);
}
