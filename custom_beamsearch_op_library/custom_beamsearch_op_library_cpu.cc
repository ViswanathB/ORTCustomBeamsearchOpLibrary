#include "custom_beamsearch_op_library.h"

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <vector>
#include <cmath>
#include <mutex>
#include <iostream>
#include "beam_search.h"

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

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
    model_path_ = nullptr;
    /* Inputs:
     * 
     * Needed to make sure, previous node output and internal graph inputs have
     * the dimensions.
     *    1. Seq_len
     *    2. batch_size,
     *    3. Any other dimension for input
     *
     */
    std::cout<<"Constructor called"<<std::endl;
    OrtStatus* status = api_.KernelInfoGetAttribute_string(info, "model_path", nullptr, &model_path_len_);
    if (status == nullptr && model_path_len_ > 0) {
      std::cout<<"Model exists with path length:"<<model_path_len_<<std::endl;
      //ORT_CXX_API_THROW("Didn't find model_path attribute in CustomBeamsearchOpKernel", ORT_FAIL);

      //Using generic heap memory since this is very small
      char *c_model_path = new char[model_path_len_];
      status = api_.KernelInfoGetAttribute_string(info, "model_path", c_model_path, &model_path_len_);
      if (status != nullptr)
      {
        ORT_CXX_API_THROW("Couldn't find model_path attribute in CustomBeamsearchOpKernel", ORT_FAIL);
      }

      //This also should work to create the wstring
      //char* model_path = "D:\\ai\\onnxruntime\\onnxruntime\\bart_mlp_megatron_basic_test.onnx";
      //std::wstring model_path_wstring(&model_path[0], &model_path[66]);

      model_path_ = new wchar_t[model_path_len_];
      size_t afterconversionlen;
      mbstowcs_s(&afterconversionlen, model_path_, model_path_len_, (const char*) c_model_path, model_path_len_-1);
      /*
      std::cout<<"CWpath:"<<std::endl;
      for (int i=0;i<model_path_len_;i++) {
        std::wcout<<model_path_[i];
      }
      std::cout<<std::endl;
      std::cout << "Length of WPath:" << afterconversionlen << std::endl;
      */
      delete c_model_path;
    }
    session_ = nullptr;
  }

  void RunInternalSession(OrtKernelContext* /*context*/){
    std::array<int64_t, 3> inputShape = {1, 2, 4};
    std::array<int64_t, 3> outputShape = {1, 2, 4};
    std::array<float, 1 * 2 * 4> input1 = {1.0f, -1.2f, 1.0f, 0.0f, -1.2f, 1.0f, 1.0f, 1.0f};
    std::array<float, 1 * 2 * 4> output1;
    std::array<const char*, 1> inputNames = {"input"};
    std::array<const char*, 1> outputNames = {"output"};

    OrtMemoryInfo *ortmemoryinfo;
    // Must be freed explicitly
    api_.CreateMemoryInfo("Cpu", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeCPU, &ortmemoryinfo);

    OrtValue* inputvalue;
    OrtValue* outputvalue;
    api_.CreateTensorWithDataAsOrtValue(ortmemoryinfo, input1.data(), 4*input1.size(), inputShape.data(),
        inputShape.size(), ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,  &inputvalue);
    api_.CreateTensorWithDataAsOrtValue(ortmemoryinfo, output1.data(), 4*output1.size(), outputShape.data(),
        outputShape.size(), ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &outputvalue);

    api_.Run(session_, nullptr, inputNames.data(), &inputvalue, 1, outputNames.data(), 1, &outputvalue);
    std::cout<<"Internal session run completed"<<std::endl;
    for (int i = 0; i < 1 * 2 * 4; i++) {
      std::cout << i << ":" << output1[i] << std::endl;
    }
  }

  void SetOutputToZero(OrtKernelContext* context) {
    const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
    OrtTensorDimensions dimensions(ort_, input_X);

    OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
    int* out = ort_.GetTensorMutableData<int>(output);

    OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
    int64_t size = ort_.GetTensorShapeElementCount(output_info);
    ort_.ReleaseTensorTypeAndShapeInfo(output_info);

    for (int64_t i = 0; i < size-1; i++) {
        out[i] = 1;
    }
  }

  void Compute(OrtKernelContext* context) {
    if (session_ == nullptr && model_path_ != nullptr) {
      // The first two arguments don't matter since we are not creating the env for the first
      // time. We would only access the existing env created for parent session with is triggering this
      // custom OP.
      api_.CreateEnv(OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO, "", &env_);

      OrtSessionOptions* sessionoptions;
      api_.CreateSessionOptions(&sessionoptions);

      std::time_t start_time = std::time(0);
      OrtStatusPtr status = api_.CreateSession(env_, model_path_, sessionoptions, &session_);

      if (status == nullptr) {

      }
      std::time_t end_time = std::time(0);
      std::cout << "Time elapsed for creating a session:" << end_time - start_time << std::endl;
    }
    
    if (model_path_ != nullptr) {
      RunInternalSession(context);
    }
/*
    const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
    const OrtValue* input_Y = ort_.KernelContext_GetInput(context, 1);
    const float* X = ort_.GetTensorData<float>(input_X);
    const float* Y = ort_.GetTensorData<float>(input_Y);

    // Setup output
    OrtTensorDimensions dimensions(ort_, input_X);

    OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
    int* out = ort_.GetTensorMutableData<int>(output);

    OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
    int64_t size = ort_.GetTensorShapeElementCount(output_info);
    ort_.ReleaseTensorTypeAndShapeInfo(output_info);

    BeamSearchCPU(size,
                  X,
                  Y,
                  out);
*/
   
   // TODO remove this.
   // Only used to set the output to some random value to not throw an error
   SetOutputToZero(context);
  }

 private:
  OrtApi api_;  // keep a copy of the struct, whose ref is used in the ort_
  Ort::CustomOpApi ort_;
  OrtSession* session_;
  OrtEnv* env_;

  //Attributes
  wchar_t *model_path_;
  size_t model_path_len_; 
};

struct CustomBeamsearchOp : Ort::CustomOpBase<CustomBeamsearchOp, CustomBeamsearchOpKernel> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const {
    return new CustomBeamsearchOpKernel(api, info);
  };

  const char* GetName() const { return "CustomBeamsearchOp"; };
  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; };

  size_t GetInputTypeCount() const { return 2; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/index) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32; 
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
