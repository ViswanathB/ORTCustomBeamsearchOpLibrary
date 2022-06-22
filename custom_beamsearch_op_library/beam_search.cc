// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <cstdint>
#include <iostream>
#include <safeint.h>

#include "beam_search.h"
#include "beam_search_parameters.h"
#include "beam_search_device_helper.h"
#include "utils.h"

using namespace std;

template <typename T>
class BeamSearchImpl {
 public:
  BeamSearchImpl(OrtApi &api,
                 Ort::CustomOpApi &ort,
                 OrtKernelContext* context,
                 //const SessionState& session_state,
                 //GptSubgraph& gpt_subgraph,
                 void* thread_pool,
                 //void* cuda_stream,
                 //IConsoleDumper* cuda_dumper,
                 custombsop::BeamSearchParameters &params,
                 //OrtAllocator *ort_allocator,
                 const BeamSearchDeviceHelper::CreateInputsFunc& create_inputs_func)
                 //const BeamSearchDeviceHelper::AddToFeedsFunc& add_to_feeds_func,
                 //const BeamSearchDeviceHelper::TopkFunc& topk_func,
                 //const BeamSearchDeviceHelper::ProcessLogitsFunc<T>& process_logits_func,
                 //const BeamSearchDeviceHelper::InitBeamStateFunc<T>& init_beam_state_func,
                 //const BeamSearchDeviceHelper::DeviceCopyFunc<float>& device_copy_func,
                 //const BeamSearchDeviceHelper::UpdateFeedsFunc<T>& update_feeds_func)
      : api_(api),
        ort_(ort),
        context_(context),
        //session_state_(session_state),
        //gpt_subgraph_(gpt_subgraph),
        thread_pool_(thread_pool),
        //implicit_inputs_(context_.GetImplicitInputs()),
        //cuda_stream_(cuda_stream),
        //cuda_dumper_(cuda_dumper),
        parameters_(params),
        //cpu_allocator_(ort_allocator),
        //temp_space_allocator_(nullptr),
        create_inputs_func_(create_inputs_func) {
        //add_to_feeds_func_(add_to_feeds_func),
        //topk_func_(topk_func),
        //process_logits_func_(process_logits_func),
        //init_beam_state_func_(init_beam_state_func),
        //device_copy_func_(device_copy_func),
        //update_feeds_func_(update_feeds_func) {

    /*parameters_->ParseFromInputs(&context);
    cpu_allocator_ = session_state.GetExecutionProviders()
                         .Get(onnxruntime::kCpuExecutionProvider)
                         ->GetAllocator(0, OrtMemTypeDefault);
                         */
    parameters_.ParseFromInputs(context, ort_);
    parameters_.Validate(api_);
  }

  // Initialize by validating all the inputs, and allocating the output tensors.
  OrtStatusPtr Initialize();

  // Execute beam search in iterations util stopping criteria is reached.
  // In each iteration, GPT subgraph is called, and next token for each sequence is generated.
  //Status Execute(const FeedsFetchesManager& feeds_fetches_manager);
  OrtStatusPtr Execute(const OrtValue *original_input_ids, OrtAllocator *ort_allocator);

 private:
  //bool IsCuda() const { return cuda_stream_ != nullptr; }

  // Validate inputs.
  OrtStatusPtr CheckInputs(const OrtKernelContext& context);

  // Prepare the inputs for first inference of subgraph
  //OrtStatusPtr CreateInitialFeeds(gsl::span<int32_t>& sequence_lengths, OrtValue& expanded_input_ids, std::vector<OrtValue>& feeds, IAllocatorUniquePtr<char>& buffer);

  // Update the input for next iteration.
  /*Status UpdateFeeds(
      const std::vector<OrtValue>& last_outputs,
      std::vector<OrtValue>& next_inputs,
      int current_length,
      OrtValue& position_ids,
      gsl::span<const int32_t> beam_next_tokens,
      gsl::span<const int32_t> beam_indices);
  */
  // Process logits and append next tokens to sequences.
  /*
  Status GenerateNextToken(const OrtValue& logits,
                           gsl::span<int32_t>& beam_next_tokens,
                           gsl::span<int32_t>& beam_indices,
                           BeamSearchState<T>& beam_state,
                           BeamSearchCpuState& cpu_state,
                           int counter);

  // Calculate scores from logits, then apply filtering and select next token for each beam.
  Status ProcessLogits(const OrtValue& logits,  // logits output of subgraph
                       BeamSearchState<T>& beam_state,
                       BeamSearchCpuState& cpu_state,
                       AllocatorPtr& allocator,
                       int counter);
  */
  //const IConsoleDumper* GetConsoleDumper() const { return IsCuda() ? cuda_dumper_ : &(cpu_dumper_); }

  OrtApi api_;

  Ort::CustomOpApi ort_;

  OrtKernelContext* context_;

  //const SessionState& session_state_;

  //GptSubgraph& gpt_subgraph_;

  void* thread_pool_;

  //const std::vector<const OrtValue*>& implicit_inputs_;

  //void* cuda_stream_;

  //IConsoleDumper* cuda_dumper_;
  //CpuTensorConsoleDumper cpu_dumper_;

  custombsop::BeamSearchParameters parameters_;

  //LogitsProcessorList logits_processors_;

  //std::unique_ptr<BeamSearchScorer> beam_scorer_;

  OrtAllocator* cpu_allocator_;
  //AllocatorPtr temp_space_allocator_;

  // Device specific functions
  BeamSearchDeviceHelper::CreateInputsFunc create_inputs_func_;
  //BeamSearchDeviceHelper::AddToFeedsFunc add_to_feeds_func_;
  //BeamSearchDeviceHelper::TopkFunc topk_func_;
  //BeamSearchDeviceHelper::ProcessLogitsFunc<T> process_logits_func_;
  //BeamSearchDeviceHelper::InitBeamStateFunc<T> init_beam_state_func_;
  //BeamSearchDeviceHelper::DeviceCopyFunc<float> device_copy_func_;
  //BeamSearchDeviceHelper::UpdateFeedsFunc<T> update_feeds_func_;
};


template <typename T>
OrtStatusPtr BeamSearchImpl<T>::Initialize() {
  //TODO why a temporary allocator is needed??
  //ORT_RETURN_IF_ERROR(context_.GetTempSpaceAllocator(&temp_space_allocator_));

#define CHECK_SCALAR_INPUT(name, index, required)                                                                             \
  auto* name##_tensor = ort_.KernelContext_GetInput(context_, index);                                                         \
  if (name##_tensor) {                                                                                                        \
    const OrtTensorTypeAndShapeInfo* tensor_info = ort_.GetTensorTypeAndShape(name##_tensor);                                 \
    std::vector<int64_t> shape = ort_.GetTensorShape(tensor_info);                                                            \
    if (!(shape.size() == 0 || (shape.size() == 1 && shape[0] == 1))) {                                                       \
      return api_.CreateStatus(OrtErrorCode::ORT_INVALID_ARGUMENT , "'BeamSearch' input " #name " doesn't have valid shape"); \
    }                                                                                                                         \
  } else if (required) {                                                                                                      \
    return api_.CreateStatus(OrtErrorCode::ORT_INVALID_ARGUMENT, "'BeamSearch' input " #name " is required");                 \
  }

  CHECK_SCALAR_INPUT(max_length, 1, false);

  CHECK_SCALAR_INPUT(min_length, 2, true);

  CHECK_SCALAR_INPUT(num_beams, 3, true);

  CHECK_SCALAR_INPUT(num_return_sequences, 4, true);

  CHECK_SCALAR_INPUT(temperature, 5, true);

  CHECK_SCALAR_INPUT(length_penalty, 6, true);

  //TODO Create a function in utils
  // to create status messages based on a condition
  if (parameters_.num_return_sequences > parameters_.num_beams) {
    return api_.CreateStatus(OrtErrorCode::ORT_INVALID_ARGUMENT, "'num_return_sequences' has to be smaller or equal to 'num_beams'.");
  }

  //TODO next_thing_to_be_implemented.
  //ORT_RETURN_IF_ERROR(CheckInputs(context_));

  // This flag will be updated later when the scores output exists.
  parameters_.output_scores = false;

  /*
  if (!IsCuda()) {
    // Logits processor is used in CPU only. In CUDA, cuda kernels are used instead.
    // Initialize processsors after CheckInputs so that parameters_->vocab_mask is ready.
    logits_processors_.Init(*parameters_);
  }
  */

  return nullptr;
  //return api_.CreateStatus(OrtErrorCode::ORT_OK, "Inputs good");
}

template <typename T>
gsl::span<T> AllocateBuffer(OrtAllocator *allocator,
                            void **buffer,
                            size_t elements,
                            bool fill = false,
                            T fill_value = T{}) {
  //size_t bytes = SafeInt<size_t>(sizeof(T)) * elements;
  size_t bytes = sizeof(T) * elements;
  *buffer = allocator->Alloc(allocator, bytes);

  //void* data = allocator->Alloc(bytes);
  //BufferUniquePtr temp_buffer(data, BufferDeleter(allocator));
  //buffer = std::move(temp_buffer);
  
  T* first = reinterpret_cast<T*>(*buffer);
  auto span = gsl::make_span(first, elements);

  if (fill) {
    std::fill_n(first, elements, fill_value);
  }

  return span;
}

template <typename T>
OrtStatusPtr BeamSearchImpl<T>::Execute(const OrtValue *original_input_ids, OrtAllocator *ort_allocator) {
  std::cout<<"Calling create inputs:"<<std::endl;

  void* sequence_lengths_buffer;
  gsl::span<int32_t> sequence_lengths_ = AllocateBuffer<int32_t>(ort_allocator, &sequence_lengths_buffer, parameters_.BatchBeamSize());

  OrtValue *expanded_inputs;
  OrtValue *expanded_position_ids;
  OrtValue *expanded_attention_mask;
  OrtStatusPtr status = create_inputs_func_(api_, ort_, original_input_ids, 
            parameters_.num_beams, parameters_.pad_token_id, sequence_lengths_, ort_allocator,
            &expanded_inputs, &expanded_position_ids, &expanded_attention_mask);
  if (status != nullptr) {
    std::cout<<"Error while expanding inputs:"<<api_.GetErrorMessage(status)<< std::endl;
    abort();
  }

  /*
  //TODO Remove Debugs to print expanded inputs
  int32_t* expanded_inputs_data = ort_.GetTensorMutableData<int32_t>(expanded_inputs);
  int32_t* expanded_position_ids_data = ort_.GetTensorMutableData<int32_t>(expanded_position_ids);
  int32_t* expanded_attention_mask_data = ort_.GetTensorMutableData<int32_t>(expanded_attention_mask);

  std::cout<<"ALL Expanded inputs:"<<std::endl;
  for (int i = 0; i < parameters_.batch_size; i++) {
    for (int j = 0; j < parameters_.num_beams;j++) {
      for (int k = 0; k < parameters_.sequence_length;k++){
        std::cout<<expanded_inputs_data[i*parameters_.num_beams + j*parameters_.sequence_length + k]<<",";
      }
      std::cout<<std::endl;
      for (int k = 0; k < parameters_.sequence_length;k++){
        std::cout<<expanded_position_ids_data[i*parameters_.num_beams + j*parameters_.sequence_length + k]<<",";
      }
      std::cout<<std::endl;
      for (int k = 0; k < parameters_.sequence_length;k++){
        std::cout<<expanded_attention_mask_data[i*parameters_.num_beams + j*parameters_.sequence_length + k]<<",";
      }
      std::cout<<std::endl;
    }
    std::cout<<std::endl;
  }
  */

  return nullptr;
}

void SetBeamSearchOutputToZero(OrtKernelContext* context, Ort::CustomOpApi &ort, int batch_size, int seq_len) {
  //TODO Temp function to set the outputs of custom beam search OP to some value
  // so there is no error, this would be ideally set to 
  //std::cout<<"Setting BeamSearchOutputToZero:"<<std::endl;
  const OrtValue* input_ids = ort.KernelContext_GetInput(context, 0);

  const OrtValue* num_ret_seqs = ort.KernelContext_GetInput(context, 4);
  const int* num_ret_seqs_data = ort.GetTensorData<int>(num_ret_seqs);
  const int num_seqs = num_ret_seqs_data[0];

  std::vector<int64_t> output1_dims;
  output1_dims.push_back(batch_size);
  output1_dims.push_back(num_seqs);
  output1_dims.push_back(seq_len);

  OrtValue* output1 = ort.KernelContext_GetOutput(context, 0, output1_dims.data(), output1_dims.size());
  int* out1 = ort.GetTensorMutableData<int>(output1);

  OrtTensorTypeAndShapeInfo* output_info1 = ort.GetTensorTypeAndShape(output1);
  std::vector<int64_t> tensor_shape = ort.GetTensorShape(output_info1);

#ifdef PRINT_TO_CONSOLE
  std::cout<<"Tensor shape of first output of custom bs OP"<<std::endl;
  for (int i=0;i<tensor_shape.size();i++){
    std::cout<<tensor_shape[i]<<",";
  }
  std::cout<<std::endl;
#endif

  int64_t size1 = ort.GetTensorShapeElementCount(output_info1);
  ort.ReleaseTensorTypeAndShapeInfo(output_info1);
  for (int64_t i = 0; i < size1-1; i++) {
    out1[i] = 10;
  }

  std::vector<int64_t> output2_dims;
  output2_dims.push_back(batch_size);
  output2_dims.push_back(num_seqs);

  OrtValue* output2 = ort.KernelContext_GetOutput(context, 1, output2_dims.data(), output2_dims.size());
  float* out2 = ort.GetTensorMutableData<float>(output2);

  OrtTensorTypeAndShapeInfo* output_info2 = ort.GetTensorTypeAndShape(output2);
  int64_t size2 = ort.GetTensorShapeElementCount(output_info2);
  ort.ReleaseTensorTypeAndShapeInfo(output_info2);

  // TODO these values should actually be set from logits
  for (int64_t i = 0; i < size2-1; i++) {
    out2[i] = 2.0f;
  }
}

void RunBeamSearchOnInternalSession(OrtKernelContext* context, OrtApi &api, Ort::CustomOpApi &ort, OrtSession *session, custombsop::BeamSearchParameters parameters) {
  std::vector<OrtValue*> inputs;
  std::vector<const char*> input_names{"input_ids", "position_ids", "attention_mask", "past_0", "past_1", "past_2", "past_3", "past_4", "past_5"};


  // Both of the following should provide the same thing, one for memory info and other for ort allocator.
  OrtMemoryInfo *ortmemoryinfo;
  // Must be freed explicitly
  api.CreateMemoryInfo("Cpu", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeCPU, &ortmemoryinfo);

  OrtAllocator *ortallocator;
  api.GetAllocatorWithDefaultOptions(&ortallocator);


  void* thread_pool = ort.KernelContext_GetThreadPool(context);

  BeamSearchImpl<float>impl{api,
                            ort,
                            context,
                            //*session_state,
                            //*gpt_subgraph_,
                            thread_pool,
                            //cuda_stream_,
                            //dumper_,
                            parameters,
                            //std::make_unique<OrtAllocator>(*ortmemoryinfo),
                            BeamSearchCpuDeviceHelper::CreateInputs};
                            //add_to_feeds_func_ ? add_to_feeds_func_ : BeamSearchCpuDeviceHelper::AddToFeeds,
                            //topk_func_ ? topk_func_ : BeamSearchCpuDeviceHelper::TopK,
                            //process_logits_func_ ? process_logits_func_ : BeamSearchCpuDeviceHelper::ProcessLogits<float>,
                            //init_beam_state_func_ ? init_beam_state_func_ : BeamSearchCpuDeviceHelper::InitBeamState<float>,
                            //device_copy_func_ ? device_copy_func_ : BeamSearchCpuDeviceHelper::DeviceCopy<float>,
                            //update_feeds_func_ ? update_feeds_func_ : BeamSearchCpuDeviceHelper::UpdateFeeds<float>};
  OrtStatusPtr status_ptr = impl.Initialize();
  if (status_ptr != nullptr) {
    //TODO handle better than abort
    std::cout<<" Beam search is not initialized properly : "<<api.GetErrorMessage(status_ptr)<<std::endl;
    abort();
  }

  const OrtValue* input_ids_tensor = ort.KernelContext_GetInput(context, 0);
  const int* input_ids = ort.GetTensorData<int>(input_ids_tensor);
  
  impl.Execute(input_ids_tensor, ortallocator);

  OrtTensorTypeAndShapeInfo* input_ids_info = ort.GetTensorTypeAndShape(input_ids_tensor);
  std::vector<int64_t> tensor_shape = ort.GetTensorShape(input_ids_info);
  ort.ReleaseTensorTypeAndShapeInfo(input_ids_info);

  int64_t batch_size = tensor_shape[0];
  int64_t seq_len = tensor_shape[1];

  // TODO, short cut to get the basic comparison with python version
  const OrtValue* max_length_tensor = ort.KernelContext_GetInput(context, 1);
  const int* max_length = ort.GetTensorData<int>(max_length_tensor);

  int iterations = max_length[0];
#ifdef PRINT_TO_CONSOLE
  std::cout<<"Batch_size:"<<batch_size<<std::endl;
  std::cout<<"Seq_len:"<<seq_len<<std::endl;
#endif

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
  api.CreateTensorWithDataAsOrtValue(ortmemoryinfo, input_ids_data.data(), 4 * batch_size * seq_len, tensor_shape.data(), tensor_shape.size(),
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &ort_input_ids);
  inputs.emplace_back(std::move(ort_input_ids));

  OrtValue *pos_ids;
  api.CreateTensorWithDataAsOrtValue(ortmemoryinfo, position_ids_data.data(), 4 * batch_size * seq_len, tensor_shape.data(), tensor_shape.size(),
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &pos_ids);
  inputs.emplace_back(std::move(pos_ids));

  OrtValue *attn_mask;
  api.CreateTensorWithDataAsOrtValue(ortmemoryinfo, attention_mask_data.data(), 4 * batch_size * seq_len, tensor_shape.data(), tensor_shape.size(),
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &attn_mask);
  inputs.emplace_back(std::move(attn_mask));

#ifdef PRINT_TO_CONSOLE
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < seq_len; j++) {
      std::cout<<(ort.GetTensorData<int>(inputs[0]))[i*seq_len+j]<<",";
    }
    std::cout<<std::endl;
  }

  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < seq_len; j++) {
      std::cout<<(ort.GetTensorData<int>(inputs[1]))[i*seq_len+j]<<",";
    }
    std::cout<<std::endl;
  }

  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < seq_len; j++) {
      std::cout<<(ort.GetTensorData<int>(inputs[2]))[i*seq_len+j]<<",";
    }
    std::cout<<std::endl;
  }
#endif

  int64_t past_seq_len = 0;
  int64_t num_heads = 16;
  int64_t head_size = 64;

  std::vector<int64_t> past_dims{2, batch_size, num_heads, past_seq_len, head_size};
  std::vector<float> past_data;
  
  for (int i = 0; i < 6; i++) {
    OrtValue *ort_past_data;
    api.CreateTensorWithDataAsOrtValue(ortmemoryinfo, past_data.data(), 0, past_dims.data(), past_dims.size(),
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &ort_past_data);

    inputs.emplace_back(std::move(ort_past_data));
  }

  std::vector<const char*> output_names{"logits", "present_0", "present_1", "present_2", "present_3", "present_4", "present_5"};
  std::vector<OrtValue*> outputs;
  std::vector<float> logits_data(batch_size * seq_len * 50263, 0);
  std::vector<int64_t> logits_dims{batch_size, seq_len, 50263};
  
  OrtValue* ortlogits;
  api.CreateTensorWithDataAsOrtValue(ortmemoryinfo, logits_data.data(), 4*batch_size*seq_len*50263, logits_dims.data(),
      logits_dims.size(), ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &ortlogits);
  outputs.emplace_back(std::move(ortlogits));

  std::vector<float> present_0_data(2 * batch_size * num_heads * (seq_len+past_seq_len) * head_size, 0);
  std::vector<int64_t> present_dims{2, batch_size, num_heads, seq_len+past_seq_len, head_size};
  for (int i = 0; i < 6; i++) {
    OrtValue* ort_present;
    // ort.present_1, ort.present_2, ort.present_3, ort.present_4, ort.present_5;
    api.CreateTensorWithDataAsOrtValue(ortmemoryinfo, present_0_data.data(), 4*2*batch_size*num_heads*(seq_len+past_seq_len)*head_size, present_dims.data(),
        present_dims.size(), ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &ort_present);
    outputs.emplace_back(ort_present);
  }
  
  //for (int i = 0; i < iterations; i++) {
  for (int i = 0; i < 1; i++) {
    //std::cout<<"Running it for "<<iterations<<std::endl;
    api.Run(session, nullptr, input_names.data(), inputs.data(), 9, output_names.data(), 7, outputs.data());
  }
#ifdef PRINT_TO_CONSOLE
  std::cout<<"Printing logits"<<std::endl;
  float* logits = ort.GetTensorMutableData<float>(outputs[0]);
  for (int i = 0; i < batch_size; i++) {
    std::cout<<"batch:"<<i<<std::endl;
    for (int j = 0; j < seq_len; j++) {
      for (int k = 0; k < 50; k++) {
      //for (int k = 0; k < 50263; k++) {
        std::cout<<logits[i*seq_len + j*50263 + k]<<"\t";
      }
      std::cout<<std::endl;
    }
    std::cout<<std::endl;
  }
#endif
  //TODO set this with the actual return sequences
  SetBeamSearchOutputToZero(context, ort, batch_size, seq_len);
}