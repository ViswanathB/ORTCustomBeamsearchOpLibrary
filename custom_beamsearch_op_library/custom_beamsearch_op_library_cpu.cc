#include "custom_beamsearch_op_library.h"

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <vector>
#include <cmath>
#include <mutex>
#include <iostream>
#include "beam_search.h"

//#define CUSTOMOP 1

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

//#define PRINT_TO_CONSOLE 1

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
        ORT_CXX_API_THROW("Couldn't find model_path attribute in CustomBeamsearchOpKernel", ORT_FAIL);
      }

      model_path_ = new wchar_t[model_path_len_];
      size_t afterconversionlen;
      mbstowcs_s(&afterconversionlen, model_path_, model_path_len_, (const char*) c_model_path, model_path_len_-1);

      delete c_model_path;
    }
    session_ = nullptr;
  }

  void RunBeamSearchInternalSession(OrtKernelContext* context) {
    std::vector<OrtValue*> inputs;
    std::vector<const char*> input_names{"input_ids", "position_ids", "attention_mask", "past_0", "past_1", "past_2", "past_3", "past_4", "past_5"};

    OrtMemoryInfo *ortmemoryinfo;
    // Must be freed explicitly
    api_.CreateMemoryInfo("Cpu", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeCPU, &ortmemoryinfo);

    const OrtValue* input_ids_tensor = ort_.KernelContext_GetInput(context, 0);
    const int* input_ids = ort_.GetTensorData<int>(input_ids_tensor);
    
    OrtTensorTypeAndShapeInfo* input_ids_info = ort_.GetTensorTypeAndShape(input_ids_tensor);
    std::vector<int64_t> tensor_shape = ort_.GetTensorShape(input_ids_info);
    ort_.ReleaseTensorTypeAndShapeInfo(input_ids_info);

    int64_t batch_size = tensor_shape[0];
    int64_t seq_len = tensor_shape[1];

    std::cout<<"Batch_size:"<<batch_size<<std::endl;
    std::cout<<"Seq_len:"<<seq_len<<std::endl;
    
    std::vector<int> input_ids_data;
    std::vector<int> position_ids_data;
    std::vector<int> attention_mask_data;
    for (int i = 0; i < batch_size; i++) {
      for (int j = 0; j < seq_len; j++) {
        input_ids_data.push_back(input_ids[i*seq_len+j]);
        position_ids_data.push_back(j);
        attention_mask_data.push_back(1);
      }
    }

    OrtValue *ort_input_ids;
    api_.CreateTensorWithDataAsOrtValue(ortmemoryinfo, input_ids_data.data(), 4 * batch_size * seq_len, tensor_shape.data(), tensor_shape.size(),
          ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &ort_input_ids);
    inputs.emplace_back(std::move(ort_input_ids));

    OrtValue *ort_pos_ids;
    api_.CreateTensorWithDataAsOrtValue(ortmemoryinfo, position_ids_data.data(), 4 * batch_size * seq_len, tensor_shape.data(), tensor_shape.size(),
          ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &ort_pos_ids);
    inputs.emplace_back(std::move(ort_pos_ids));

    OrtValue *ort_attn_mask;
    api_.CreateTensorWithDataAsOrtValue(ortmemoryinfo, attention_mask_data.data(), 4 * batch_size * seq_len, tensor_shape.data(), tensor_shape.size(),
          ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &ort_attn_mask);
    inputs.emplace_back(std::move(ort_attn_mask));

    int64_t past_seq_len = 0;
    int64_t num_heads = 16;
    int64_t head_size = 64;

    std::vector<int64_t> past_dims{2, batch_size, num_heads, past_seq_len, head_size};
    std::vector<float> past_data;
    
    for (int i = 0; i < 6; i++) {
      OrtValue *ort_past_data;
      api_.CreateTensorWithDataAsOrtValue(ortmemoryinfo, past_data.data(), 0, past_dims.data(), past_dims.size(),
          ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &ort_past_data);

      inputs.emplace_back(std::move(ort_past_data));
    }

    std::vector<const char*> output_names{"logits", "present_0", "present_1", "present_2", "present_3", "present_4", "present_5"};
    std::vector<OrtValue*> outputs;
    std::vector<float> logits_data(batch_size * seq_len * 50263, 0);
    std::vector<int64_t> logits_dims{batch_size, seq_len, 50263};
    
    OrtValue* ortlogits;
    api_.CreateTensorWithDataAsOrtValue(ortmemoryinfo, logits_data.data(), 4*batch_size*seq_len*50263, logits_dims.data(),
        logits_dims.size(), ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &ortlogits);
    outputs.emplace_back(std::move(ortlogits));

    std::vector<float> present_0_data(2 * batch_size * num_heads * (seq_len+past_seq_len) * head_size);
    std::vector<int64_t> present_dims{2, batch_size, num_heads, seq_len+past_seq_len, head_size};
    for (int i = 0; i < 6; i++) {
      OrtValue* ort_present;
      // ort_present_1, ort_present_2, ort_present_3, ort_present_4, ort_present_5;
      api_.CreateTensorWithDataAsOrtValue(ortmemoryinfo, present_0_data.data(), 4*2*batch_size*num_heads*(seq_len+past_seq_len)*head_size, present_dims.data(),
          present_dims.size(), ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &ort_present);
      outputs.emplace_back(ort_present);
    }
    api_.Run(session_, nullptr, input_names.data(), inputs.data(), 9, output_names.data(), 7, outputs.data());

    std::cout<<"Printing logits"<<std::endl;
    int* logits = ort_.GetTensorMutableData<int>(outputs[0]);
    for (int i = 0; i < batch_size; i++) {
      std::cout<<"batch:"<<i<<std::endl;
      for (int j = 0; j < seq_len; j++) {
        std::cout<<logits[i*seq_len + j]<<",";
      }
      std::cout<<std::endl;
    }
  }

  void SetBeamSearchOutputToZero(OrtKernelContext* context) {
    std::cout<<"Inside SetBeamSearchOutputToZero"<<std::endl;
    const OrtValue* input_ids = ort_.KernelContext_GetInput(context, 0);
    OrtTensorDimensions dimensions(ort_, input_ids);

    const int64_t batch_size = dimensions[0];
    const int64_t seq_len = dimensions[1];

    const OrtValue* num_ret_seqs = ort_.KernelContext_GetInput(context, 4);
    const int* num_ret_seqs_data = ort_.GetTensorData<int>(num_ret_seqs);
    const int num_seqs = num_ret_seqs_data[0];
    std::cout<<"num_seqs:"<<num_seqs<<std::endl;

    std::vector<int64_t> output1_dims;
    output1_dims.push_back(batch_size);
    output1_dims.push_back(num_seqs);
    output1_dims.push_back(seq_len);

    OrtValue* output1 = ort_.KernelContext_GetOutput(context, 0, output1_dims.data(), output1_dims.size());
    int* out1 = ort_.GetTensorMutableData<int>(output1);

    OrtTensorTypeAndShapeInfo* output_info1 = ort_.GetTensorTypeAndShape(output1);
    std::vector<int64_t> tensor_shape = ort_.GetTensorShape(output_info1);

    std::cout<<"Tensor shape of first output of custom bs OP"<<std::endl;
    for (int i=0;i<tensor_shape.size();i++){
      std::cout<<tensor_shape[i]<<",";
    }
    std::cout<<std::endl;

    int64_t size1 = ort_.GetTensorShapeElementCount(output_info1);
    ort_.ReleaseTensorTypeAndShapeInfo(output_info1);
    for (int64_t i = 0; i < size1-1; i++) {
      out1[i] = 10;
    }

    std::vector<int64_t> output2_dims;
    output2_dims.push_back(batch_size);
    output2_dims.push_back(num_seqs);

    OrtValue* output2 = ort_.KernelContext_GetOutput(context, 1, output2_dims.data(), output2_dims.size());
    float* out2 = ort_.GetTensorMutableData<float>(output2);

    OrtTensorTypeAndShapeInfo* output_info2 = ort_.GetTensorTypeAndShape(output2);
    int64_t size2 = ort_.GetTensorShapeElementCount(output_info2);
    ort_.ReleaseTensorTypeAndShapeInfo(output_info2);

    for (int64_t i = 0; i < size2-1; i++) {
      out2[i] = 2.0f;
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

      OrtStatusPtr status = api_.CreateSession(env_, model_path_, sessionoptions, &session_);

      if (status != nullptr) {
        ORT_CXX_API_THROW("Unable to create internal session for beam", ORT_FAIL);
      }
    }

#ifdef CUSTOMOP
    if (model_path_ != nullptr) {
      RunInternalSession(context);
    }
    SetOutputToZero(context);
#else
    //Beamsearch entry point
    if (model_path_ != nullptr) {
      RunBeamSearchInternalSession(context);
    }
    SetBeamSearchOutputToZero(context);
#endif
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

  size_t GetInputTypeCount() const { 
#ifdef CUSTOMOP
    return 2; 
#else
    return 10;
#endif
};
  ONNXTensorElementDataType GetInputType(size_t /*index*/index) const {
#ifdef CUSTOMOP
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
#else
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
#endif
  };

  size_t GetOutputTypeCount() const { 
#ifdef CUSTOMOP
    return 1;
#else
    return 2;
#endif
};
  ONNXTensorElementDataType GetOutputType(size_t /*index*/index) const {
#ifdef CUSTOMOP
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
#else
    switch(index) {
      case 0:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
      case 1:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
      default:
         return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
#endif
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
