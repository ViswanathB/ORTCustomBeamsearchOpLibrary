#include "custom_beamsearch_op_library.h"

#include <vector>
#include <cmath>
#include <mutex>
#include <iostream>
#include <fstream>
#include <unordered_map>

#include "json.hpp"
using json = nlohmann::json;

#include "greedy_search.h"
#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

#include "greedy_search_parameters.h"

static const char *c_OpDomain = "test.greedysearchop";

struct OrtCustomOpDomainDeleter
{
  explicit OrtCustomOpDomainDeleter(const OrtApi *ort_api)
  {
    ort_api_ = ort_api;
  }
  void operator()(OrtCustomOpDomain *domain) const
  {
    ort_api_->ReleaseCustomOpDomain(domain);
  }

  const OrtApi *ort_api_;
};

using OrtCustomOpDomainUniquePtr = std::unique_ptr<OrtCustomOpDomain, OrtCustomOpDomainDeleter>;
static std::vector<OrtCustomOpDomainUniquePtr> ort_custom_op_domain_container;
static std::mutex ort_custom_op_domain_mutex;

static void AddOrtCustomOpDomainToContainer(OrtCustomOpDomain *domain, const OrtApi *ort_api)
{
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  auto ptr = std::unique_ptr<OrtCustomOpDomain, OrtCustomOpDomainDeleter>(domain, OrtCustomOpDomainDeleter(ort_api));
  ort_custom_op_domain_container.push_back(std::move(ptr));
}

struct OrtTensorDimensions : std::vector<int64_t>
{
  OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue *value)
  {
    OrtTensorTypeAndShapeInfo *info = ort.GetTensorTypeAndShape(value);
    std::vector<int64_t>::operator=(ort.GetTensorShape(info));
    ort.ReleaseTensorTypeAndShapeInfo(info);
  }
};

struct CustomGreedySearchOpKernel
{
  CustomGreedySearchOpKernel(OrtApi api, const OrtKernelInfo *info)
      : api_(api), ort_(api_)
  {

    OrtStatus *status = api_.KernelInfoGetAttribute_string(info, "model_path", nullptr, &model_path_len_);
    if (status == nullptr && model_path_len_ > 0)
    {
      model_path_ = new char[model_path_len_];
      status = api_.KernelInfoGetAttribute_string(info, "model_path", model_path_, &model_path_len_);
      if (status != nullptr)
      {
        ORT_CXX_API_THROW("Couldn't find model_path attribute in CustomGreedySearchOpKernel Constructor", ORT_FAIL);
      }
    }
    session_ = nullptr;

    parameters_.ParseFromAttributes(ort_, info);
    OrtOp *op_topk = InitTopK(info);

    if (op_topk == nullptr)
    {
      ORT_CXX_API_THROW("Couldn't create topk OP in CustomGreedySearchOpKernel Constructor", ORT_RUNTIME_EXCEPTION);
    }
    contrib_kernels_[std::string("topk")] = op_topk;

    OrtOp *op_softmax = InitLogSoftMax(info);
    if (op_softmax == nullptr)
    {
      ORT_CXX_API_THROW("Couldn't create softmax OP in CustomGreedySearchOpKernel Constructor", ORT_RUNTIME_EXCEPTION);
    }
    contrib_kernels_[std::string("softmax")] = op_softmax;
  }

  ~CustomGreedySearchOpKernel()
  {
    /*Release the kernels created for internal operations
     */
    for (auto &it : contrib_kernels_)
    {
      ort_.ReleaseOp(reinterpret_cast<OrtOp *>(it.second));
    }

    delete model_path_;
  }

  OrtOp *InitLogSoftMax(const OrtKernelInfo *info)
  {
    const char *type_constraint_names[1] = {"T"};
    int type_constraint_values[1] = {1};

    int64_t axis_value = 1;
    OrtOpAttr *axis = ort_.CreateOpAttr("axis", &axis_value, 1, OrtOpAttrType::ORT_OP_ATTR_INT);

    if (!axis)
    {
      ORT_CXX_API_THROW("Failed to create attributes for InitLogSoftMax", ORT_RUNTIME_EXCEPTION);
    }

    OrtOpAttr *top_attrs[1] = {axis};
    OrtOp *op_lsm_ = ort_.CreateOp(info, "LogSoftmax", "", 13,
                                   (const char **)type_constraint_names,
                                   (const ONNXTensorElementDataType *)type_constraint_values,
                                   1, top_attrs, 1, 1, 1);

    ort_.ReleaseOpAttr(axis);

    return op_lsm_;
  }

  OrtOp *InitTopK(const OrtKernelInfo *info)
  {
    const char *type_constraint_names[2] = {"T", "I"};
    int type_constraint_values[2] = {1, 7};

    // logits with num_beams will have the dimension: (batch_size, num_beams * vocab_size)
    // TopK will be extracted for each batch and the index would be [0, num_beams * vocab_size] for each batch
    // this is later converted in the range [0, vocab_size] for a particular beam, ProcessLogits() has this.
    int64_t axis_value = 1;
    OrtOpAttr *axis = ort_.CreateOpAttr("axis", &axis_value, 1, OrtOpAttrType::ORT_OP_ATTR_INT);

    int64_t largest_value = 1; // return in ascending order
    OrtOpAttr *largest = ort_.CreateOpAttr("largest", &largest_value, 1, OrtOpAttrType::ORT_OP_ATTR_INT);

    int64_t sorted_value = 1;
    OrtOpAttr *sorted = ort_.CreateOpAttr("sorted", &sorted_value, 1, OrtOpAttrType::ORT_OP_ATTR_INT);

    if (!axis || !largest || !sorted)
    {
      ORT_CXX_API_THROW("Failed to create attributes for topk", ORT_RUNTIME_EXCEPTION);
    }

    OrtOpAttr *top_attrs[3] = {axis, largest, sorted};
    OrtOp *op_topk_ = ort_.CreateOp(info, "TopK", "", 14,
                                    (const char **)type_constraint_names,
                                    (const ONNXTensorElementDataType *)type_constraint_values,
                                    2, top_attrs, 3, 2, 2);

    ort_.ReleaseOpAttr(axis);
    ort_.ReleaseOpAttr(largest);
    ort_.ReleaseOpAttr(sorted);

    return op_topk_;
  }

  void ReadConfig()
  {
    std::string model_path_string(model_path_);

    size_t folder_ends = model_path_string.find_last_of("/\\");
    std::string config_file_path = model_path_string.substr(0, folder_ends + 1);
    config_file_path += "config.json";

    std::ifstream file(config_file_path);
    json json_data;
    file >> json_data;

    parameters_.vocab_size = json_data["vocab_size"];
    parameters_.num_heads = json_data["num_heads"];
    parameters_.head_size = json_data["head_size"];
    parameters_.num_layers = json_data["num_layers"];

    parameters_.length_penalty = json_data["length_penalty"];
    parameters_.repetition_penalty = json_data["repetetion_penalty"];
    parameters_.min_length = json_data["min_length"];
  }

  void Compute(OrtKernelContext *context)
  {
    if (session_ == nullptr && model_path_ != nullptr)
    {
#ifdef DEBUG_BEAM_SEARCH
      uint64_t s_time = GetTimeMs64();
#endif
      // The first two arguments don't matter since we are not creating the env for the first
      // time. We would only access the existing env created for parent session with is triggering this
      // custom OP.
      api_.CreateEnv(OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO, "", &env_);

      OrtSessionOptions *sessionoptions;
      api_.CreateSessionOptions(&sessionoptions);

      wchar_t *w_model_path = new wchar_t[model_path_len_];
      size_t afterconversionlen;
      mbstowcs_s(&afterconversionlen, w_model_path, model_path_len_, (const char *)model_path_, model_path_len_ - 1);

      OrtStatusPtr status = api_.CreateSession(env_, w_model_path, sessionoptions, &session_);

      if (status != nullptr)
      {
        ORT_CXX_API_THROW("Unable to create internal session for beam", ORT_FAIL);
      }

      ReadConfig();

#ifdef DEBUG_BEAM_SEARCH
      uint64_t e_time = GetTimeMs64();
      std::cout << "Time taken for session creation:" << e_time - s_time << std::endl;
#endif
    }

    if (session_ != nullptr)
    {
      OrtStatusPtr status = custombsop::RunGreedySearchOnInternalSession(context, api_, ort_, session_, parameters_, contrib_kernels_);
      if (status != nullptr)
      {
        ORT_CXX_API_THROW("run internal session failed:", api_.GetErrorCode(status));
      }
    }
  }

private:
  OrtApi api_;
  // keep a copy of the struct, whose ref is used in the ort_
  Ort::CustomOpApi ort_;

  OrtSession *session_;
  OrtEnv *env_;

  // Attributes
  char *model_path_;
  size_t model_path_len_;

  // Parameters
  custombsop::GreedySearchParameters parameters_;

  // Contrib Kernels
  std::unordered_map<std::string, OrtOp *> contrib_kernels_;
};

struct CustomGreedySearchOp : Ort::CustomOpBase<CustomGreedySearchOp, CustomGreedySearchOpKernel>
{
  void *CreateKernel(OrtApi api, const OrtKernelInfo *info) const
  {
    return new CustomGreedySearchOpKernel(api, info);
  };

  const char *GetName() const { return "CustomGreedySearchOp"; };
  const char *GetExecutionProviderType() const { return "CPUExecutionProvider"; };

  size_t GetInputTypeCount() const
  {
    return 10;
  };

  ONNXTensorElementDataType GetInputType(size_t /*index*/ index) const
  {
    switch (index)
    {
      case 0:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
      case 1:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
      case 2:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
      case 3:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
      default:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
  };

  size_t GetOutputTypeCount() const
  {
    return 1;
  };

  ONNXTensorElementDataType GetOutputType(size_t /*index*/ index) const
  {
    switch (index)
    {
    case 0:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    default:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
  };
} c_CustomGreedySearchOp;

OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options, const OrtApiBase *api)
{
  OrtCustomOpDomain *domain = nullptr;
  const OrtApi *ortApi = api->GetApi(ORT_API_VERSION);

  if (auto status = ortApi->CreateCustomOpDomain(c_OpDomain, &domain))
  {
    return status;
  }

  AddOrtCustomOpDomainToContainer(domain, ortApi);

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_CustomGreedySearchOp))
  {
    return status;
  }

  return ortApi->AddCustomOpDomain(options, domain);
}
