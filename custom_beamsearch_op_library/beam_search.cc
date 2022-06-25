// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <cstdint>
#include <iostream>
#include <safeint.h>

#include "beam_search.h"
#include "beam_search_parameters.h"
#include "beam_search_device_helper.h"
#include "beam_search_scorer.h"
#include "sequences.h"
#include "dump_tensor.h"
#include "utils.h"

using namespace std;
using namespace msl::utilities;
namespace custombsop {

struct BeamSearchCpuState : public custombsop::IBeamSearchCpuState {
  custombsop::Sequences sequences;

  void Init(OrtAllocator* ort_allocator, size_t batch_beam_size, int max_length, bool is_cuda) {
    void *temp_ptr;
    this->sequence_lengths = AllocateBuffer<int32_t>(ort_allocator, &temp_ptr, batch_beam_size);
    sequence_lengths_buffer_ = std::move(std::unique_ptr<int32_t>(reinterpret_cast<int32_t*>(temp_ptr)));
    this->sequences_space = AllocateBuffer<int32_t>(ort_allocator, &temp_ptr, size_t(2) * batch_beam_size * max_length);
    sequences_space_buffer_ = std::move(std::unique_ptr<int32_t>(reinterpret_cast<int32_t*>(temp_ptr)));

    // TODO This is only needed for cuda, so commenting this out
    // rather this code path is never hit.
    if (is_cuda) {
     /* // buffers used by CUDA operator but not by CPU operator.
      this->topk_scores = AllocateBuffer<float>(ort_allocator, topk_scores_buffer_, 2 * batch_beam_size);
      this->topk_tokens = AllocateBuffer<int32_t>(ort_allocator, topk_tokens_buffer_, 2 * batch_beam_size);
      this->topk_indices = AllocateBuffer<int32_t>(ort_allocator, topk_indices_buffer_, 2 * batch_beam_size);
      this->final_beam_scores = AllocateBuffer<float>(ort_allocator, final_beam_scores_buffer_, batch_beam_size);
      */
    }
  }

 private:
  std::unique_ptr<float> final_beam_scores_buffer_;
  std::unique_ptr<int32_t> sequence_lengths_buffer_;
  std::unique_ptr<float> topk_scores_buffer_;
  std::unique_ptr<int32_t> topk_tokens_buffer_;
  std::unique_ptr<int32_t> topk_indices_buffer_;
  std::unique_ptr<int32_t> sequences_space_buffer_;
};

template<typename T>
struct BeamSearchState : public custombsop::IBeamSearchState<T> {
  void Init(OrtAllocator* allocator,
            int batch_size,
            int num_beams,
            int vocab_size,
            int sequence_length,
            int max_length,
            bool output_scores) {
    size_t batch_beam_size = SafeInt<size_t>(batch_size) * num_beams;

    size_t next_token_size = SafeInt<size_t>(batch_beam_size) * vocab_size;

    void *temp_ptr;
    this->next_token_logits = AllocateBuffer<T>(allocator, &temp_ptr, next_token_size);
    next_token_logits_buffer_ = std::move(std::unique_ptr<T>(reinterpret_cast<T*>(temp_ptr)));

    this->next_token_scores = AllocateBuffer<float>(allocator, &temp_ptr, next_token_size);
    next_token_scores_buffer_ = std::move(std::unique_ptr<float>(reinterpret_cast<float*>(temp_ptr)));

    this->next_tokens = AllocateBuffer<int32_t>(allocator, &temp_ptr, SafeInt<size_t>(2) * batch_beam_size);
    next_tokens_buffer_ = std::move(std::unique_ptr<int32_t>(reinterpret_cast<int32_t*>(temp_ptr)));

    this->next_indices = AllocateBuffer<int32_t>(allocator, &temp_ptr, SafeInt<size_t>(2) * batch_beam_size);
    next_indices_buffer_ = std::move(std::unique_ptr<int32_t>(reinterpret_cast<int32_t*>(temp_ptr)));

    this->next_positions = AllocateBuffer<int32_t>(allocator, &temp_ptr, batch_beam_size);
    next_positions_buffer_ = std::move(std::unique_ptr<int32_t>(reinterpret_cast<int32_t*>(temp_ptr)));

    this->beam_scores = AllocateBuffer<float>(allocator, &temp_ptr, batch_beam_size);
    beam_scores_buffer_ = std::move(std::unique_ptr<float>(reinterpret_cast<float*>(temp_ptr)));

    if (output_scores) {
      size_t elements = SafeInt<size_t>(max_length - sequence_length) * batch_size * num_beams * vocab_size;
      this->scores = AllocateBuffer<float>(allocator, &temp_ptr, elements);
      scores_buffer_ = std::move(std::unique_ptr<float>(reinterpret_cast<float*>(temp_ptr)));
      this->remaining_scores = this->scores;
    }
  }

 private:
  std::unique_ptr<T> next_token_logits_buffer_;
  std::unique_ptr<float> next_token_scores_buffer_;
  std::unique_ptr<int32_t> next_tokens_buffer_;
  std::unique_ptr<int32_t> next_indices_buffer_;
  std::unique_ptr<int32_t> next_positions_buffer_;
  std::unique_ptr<float> beam_scores_buffer_;
  std::unique_ptr<float> scores_buffer_;
};

template <typename T>
class BeamSearchImpl {
 public:
  BeamSearchImpl(OrtApi &api,
                 Ort::CustomOpApi &ort,
                 OrtKernelContext* context,
                 //const SessionState& session_state,
                 //GptSubgraph& gpt_subgraph,
                 void* thread_pool,
                 void* cuda_stream,
                 //IConsoleDumper* cuda_dumper,
                 custombsop::BeamSearchParameters &params,
                 //OrtAllocator *ort_allocator,
                 const BeamSearchDeviceHelper::CreateInputsFunc& create_inputs_func,
                 const BeamSearchDeviceHelper::AddToFeedsFunc& add_to_feeds_func,
                 //TODO next_thing_to_be_implemented.
                 //const BeamSearchDeviceHelper::TopkFunc& topk_func,
                 //const BeamSearchDeviceHelper::ProcessLogitsFunc<T>& process_logits_func,
                 const BeamSearchDeviceHelper::InitBeamStateFunc<T>& init_beam_state_func)
                 //const BeamSearchDeviceHelper::DeviceCopyFunc<float>& device_copy_func,
                 //const BeamSearchDeviceHelper::UpdateFeedsFunc<T>& update_feeds_func)
      : api_(api),
        ort_(ort),
        context_(context),
        //session_state_(session_state),
        //gpt_subgraph_(gpt_subgraph),
        thread_pool_(thread_pool),
        //implicit_inputs_(context_.GetImplicitInputs()),
        cuda_stream_(cuda_stream),
        //cuda_dumper_(cuda_dumper),
        parameters_(params),
        //cpu_allocator_(ort_allocator),
        //temp_space_allocator_(nullptr),
        create_inputs_func_(create_inputs_func),
        add_to_feeds_func_(add_to_feeds_func),
        //topk_func_(topk_func),
        //process_logits_func_(process_logits_func),
        init_beam_state_func_(init_beam_state_func){
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
  OrtStatusPtr Execute(OrtKernelContext* context, const OrtValue *original_input_ids, OrtAllocator *ort_allocator, OrtMemoryInfo *ortmemoryinfo);

 private:
  bool IsCuda() const { return cuda_stream_ != nullptr; }

  // Validate inputs.
  OrtStatusPtr CheckInputs(const OrtKernelContext* context);

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
  const custombsop::IConsoleDumper* GetConsoleDumper() const { return IsCuda() ? cuda_dumper_ : &(cpu_dumper_); }

  OrtApi api_;

  Ort::CustomOpApi ort_;

  OrtKernelContext* context_;

  //const SessionState& session_state_;

  //GptSubgraph& gpt_subgraph_;

  void* thread_pool_;

  //const std::vector<const OrtValue*>& implicit_inputs_;

  void* cuda_stream_;

  custombsop::IConsoleDumper* cuda_dumper_;
  custombsop::CpuTensorConsoleDumper cpu_dumper_;

  custombsop::BeamSearchParameters parameters_;

  //LogitsProcessorList logits_processors_;

  std::unique_ptr<custombsop::BeamSearchScorer> beam_scorer_;

  OrtAllocator* cpu_allocator_;
  //AllocatorPtr temp_space_allocator_;

  // Device specific functions
  BeamSearchDeviceHelper::CreateInputsFunc create_inputs_func_;
  BeamSearchDeviceHelper::AddToFeedsFunc add_to_feeds_func_;
  //BeamSearchDeviceHelper::TopkFunc topk_func_;
  //BeamSearchDeviceHelper::ProcessLogitsFunc<T> process_logits_func_;
  BeamSearchDeviceHelper::InitBeamStateFunc<T> init_beam_state_func_;
  //BeamSearchDeviceHelper::DeviceCopyFunc<float> device_copy_func_;
  //BeamSearchDeviceHelper::UpdateFeedsFunc<T> update_feeds_func_;
};


template <typename T>
OrtStatusPtr BeamSearchImpl<T>::CheckInputs(const OrtKernelContext* context) {
  // Input shapes:
  //   input_ids  : (batch_size, sequence_length)
  //   vocab_mask : (vocab_size) or nullptr

  const OrtValue* input_ids = ort_.KernelContext_GetInput(context, 0);
  OrtTensorTypeAndShapeInfo* input_ids_info = ort_.GetTensorTypeAndShape(input_ids);
  std::vector<int64_t> input_ids_shape = ort_.GetTensorShape(input_ids_info);
  if (input_ids_shape.size() != 2) {
    return api_.CreateStatus(OrtErrorCode::ORT_INVALID_ARGUMENT,
                             MakeString("Input 'input_ids' is expected to have 2 dimensions, got ", input_ids_shape.size()));
  }

  const OrtValue* vocab_mask = ort_.KernelContext_GetInput(context, 8);
  if (vocab_mask != nullptr) {  // vocab_mask is optional
    OrtTensorTypeAndShapeInfo* vocab_mask_info = ort_.GetTensorTypeAndShape(vocab_mask);
    std::vector<int64_t> vocab_mask_shape = ort_.GetTensorShape(vocab_mask_info);
    if (vocab_mask_shape.size() != 1) {
      return api_.CreateStatus(OrtErrorCode::ORT_INVALID_ARGUMENT,
                              MakeString("Input 'vocab_mask' is expected to have 1 dimension, got ", vocab_mask_shape.size()));
    }

    // There is dependency on vocab_size parameter, which shall be set before calling this function.
    if (static_cast<int>(vocab_mask_shape[0]) != parameters_.vocab_size) {
      return api_.CreateStatus(OrtErrorCode::ORT_INVALID_ARGUMENT,
                              MakeString("Input 'vocab_mask' shape does not match with vocab_size, got ", vocab_mask_shape[0]));
    }

    // store vocab mask in parameters.
    const int* vm_tensor = ort_.GetTensorData<int>(vocab_mask);
    parameters_.vocab_mask = gsl::make_span(vm_tensor, parameters_.vocab_size);
  }

  const OrtValue* prefix_vocab_mask = ort_.KernelContext_GetInput(context, 9);
  if (prefix_vocab_mask != nullptr) {
    // prefix_vocab_mask is optional
    OrtTensorTypeAndShapeInfo* p_vocab_mask_info = ort_.GetTensorTypeAndShape(prefix_vocab_mask);
    std::vector<int64_t> p_vocab_mask_shape = ort_.GetTensorShape(p_vocab_mask_info);
    if (p_vocab_mask_shape.size() != 2) {
      return api_.CreateStatus(OrtErrorCode::ORT_INVALID_ARGUMENT,
                               MakeString("Input 'prefix_vocab_mask' is expected to have 2 dimensions, got ", p_vocab_mask_shape.size()));
    }

    // prefix_vocab_mask first dimension should be same as the first dimension of input_ids
    if (static_cast<int>(p_vocab_mask_shape[0]) != static_cast<int>(input_ids_shape[0])) {
      return api_.CreateStatus(OrtErrorCode::ORT_INVALID_ARGUMENT, "input_ids and prefix_vocab_mask must have the same batch_size");
    }

    // There is dependency on vocab_size parameter, which shall be set before calling this function.
    if (static_cast<int>(p_vocab_mask_shape[1]) != parameters_.vocab_size) {
      return api_.CreateStatus(OrtErrorCode::ORT_INVALID_ARGUMENT,
                              MakeString("Input 'prefix_vocab_mask' shape does not match with vocab_size, got ", p_vocab_mask_shape[1]));
    }

    // store prefix vocab mask in parameters.
    const int* pvm_tensor = ort_.GetTensorData<int>(prefix_vocab_mask);
    parameters_.prefix_vocab_mask = gsl::make_span(pvm_tensor, parameters_.batch_size * parameters_.vocab_size);
  }
  return nullptr;
}

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

  CUSTOMOP_RETURN_IF_ERROR(CheckInputs(context_));

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
}

template <typename T>
OrtStatusPtr BeamSearchImpl<T>::Execute(OrtKernelContext* context, const OrtValue *original_input_ids, OrtAllocator *ort_allocator, OrtMemoryInfo *ortmemoryinfo) {
  std::cout<<"Calling create inputs:"<<std::endl;

  std::vector<OrtValue*> outputs;
  OrtValue* sequences;
  size_t sequences_data_size = parameters_.batch_size * parameters_.num_return_sequences * parameters_.max_length;
  std::vector<int32_t> sequences_data(sequences_data_size, 0);
  vector<int64_t> sequences_dims = {parameters_.batch_size, parameters_.num_return_sequences, parameters_.max_length};
  api_.CreateTensorWithDataAsOrtValue(ortmemoryinfo, sequences_data.data(), size_t(4)*sequences_data_size, sequences_dims.data(),
      sequences_dims.size(), ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &sequences);
  outputs.emplace_back(std::move(sequences));

  OrtValue* sequences_scores;
  size_t sequences_scores_data_size = parameters_.batch_size * parameters_.num_return_sequences;
  std::vector<float> sequences_scores_data(sequences_scores_data_size, 0);
  vector<int64_t> sequences_scores_dims = {parameters_.batch_size, parameters_.num_return_sequences};
  api_.CreateTensorWithDataAsOrtValue(ortmemoryinfo, sequences_scores_data.data(), size_t(4)*sequences_scores_data_size, sequences_scores_dims.data(),
      sequences_scores_dims.size(), ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &sequences_scores);
  outputs.emplace_back(std::move(sequences_scores));

  std::vector<OrtValue*> feeds;
  std::vector<OrtValue*> fetches;

  beam_scorer_ = std::make_unique<custombsop::BeamSearchScorer>(static_cast<size_t>(parameters_.batch_size),
                                                    static_cast<size_t>(parameters_.num_beams),
                                                    static_cast<size_t>(parameters_.max_length),
                                                    parameters_.length_penalty,
                                                    parameters_.early_stopping,
                                                    static_cast<size_t>(parameters_.num_return_sequences),
                                                    parameters_.pad_token_id,
                                                    parameters_.eos_token_id);
  beam_scorer_->Initialize(ort_allocator, parameters_.sequence_length);

  BeamSearchCpuState cpu_state;
  cpu_state.Init(ort_allocator, static_cast<size_t>(parameters_.BatchBeamSize()), parameters_.max_length, IsCuda());

  //TODO how is stack address working when call is made to AllocateBuffer
  //void* sequence_lengths_buffer;
  //gsl::span<int32_t> sequence_lengths_ = AllocateBuffer<int32_t>(ort_allocator, &sequence_lengths_buffer, parameters_.BatchBeamSize());

  OrtValue *expanded_inputs;
  OrtValue *expanded_position_ids;
  OrtValue *expanded_attention_mask;
  OrtStatusPtr status = create_inputs_func_(api_, ort_, original_input_ids, 
            parameters_.num_beams, parameters_.pad_token_id, cpu_state.sequence_lengths, ort_allocator,
            &expanded_inputs, &expanded_position_ids, &expanded_attention_mask);
  if (status != nullptr) {
    std::cout<<"Error while expanding inputs:"<<api_.GetErrorMessage(status)<< std::endl;
    return status;
  }

  BeamSearchCpuDeviceHelper::AddToFeeds(expanded_inputs, expanded_position_ids, expanded_attention_mask, feeds);

#ifdef DEBUG_BEAM_SEARCH
  const custombsop::IConsoleDumper* dumper = GetConsoleDumper();
  dumper->Print("Here are input_ids", ort_.GetTensorData<int32_t>(expanded_inputs), parameters_.batch_size*parameters_.num_beams, parameters_.sequence_length);
  dumper->Print("position_ids", ort_.GetTensorData<int32_t>(expanded_position_ids), parameters_.batch_size*parameters_.num_beams, parameters_.sequence_length);
  dumper->Print("attention_mask", ort_.GetTensorData<int32_t>(expanded_attention_mask), parameters_.batch_size*parameters_.num_beams, parameters_.sequence_length);
#endif

  BeamSearchState<T> beam_state;
  beam_state.Init(ort_allocator,
                  parameters_.batch_size,
                  parameters_.num_beams,
                  parameters_.vocab_size,
                  parameters_.sequence_length,
                  parameters_.max_length,
                  parameters_.output_scores);

  cpu_state.sequences.Init(cpu_state.sequences_space,
                          parameters_.BatchBeamSize(),
                          parameters_.sequence_length,
                          parameters_.max_length);

  gsl::span<const int32_t> input_ids = gsl::make_span(ort_.GetTensorData<int32_t>(expanded_inputs), 
                                                      size_t(parameters_.batch_size) * parameters_.num_beams *  parameters_.sequence_length);

  init_beam_state_func_(&beam_state,
                        &cpu_state,
                        cpu_state.sequence_lengths,
                        parameters_.batch_size,
                        parameters_.num_beams,
                        input_ids,
                        parameters_.sequence_length,
                        parameters_.max_length,
                        cuda_stream_);

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

#ifdef DEBUG_BEAM_SEARCH
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

OrtStatusPtr RunBeamSearchOnInternalSession(OrtKernelContext* context, OrtApi &api, Ort::CustomOpApi &ort, OrtSession *session, custombsop::BeamSearchParameters parameters) {
  std::vector<OrtValue*> inputs;
  std::vector<const char*> input_names{"input_ids", "position_ids", "attention_mask", "past_0", "past_1", "past_2", "past_3", "past_4", "past_5"};

  // Both of the following should provide the same thing, one for memory info and other for ort allocator.
  OrtMemoryInfo *ortmemoryinfo;
  // Must be freed explicitly
  api.CreateMemoryInfo("Cpu", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeCPU, &ortmemoryinfo);

  OrtAllocator *ortallocator;
  api.GetAllocatorWithDefaultOptions(&ortallocator);

  void* thread_pool = ort.KernelContext_GetThreadPool(context);
  void* cuda_stream_ = nullptr;

  BeamSearchImpl<float>impl{api,
                            ort,
                            context,
                            //*session_state,
                            //*gpt_subgraph_,
                            thread_pool,
                            cuda_stream_,
                            //dumper_,
                            parameters,
                            //std::make_unique<OrtAllocator>(*ortmemoryinfo),
                            BeamSearchCpuDeviceHelper::CreateInputs,
                            BeamSearchCpuDeviceHelper::AddToFeeds,
                            //topk_func_ ? topk_func_ : BeamSearchCpuDeviceHelper::TopK,
                            //BeamsearchCpuDeviceHelper::ProcessLogits<float>,
                            BeamSearchCpuDeviceHelper::InitBeamState<float>};
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
  
  return impl.Execute(context, input_ids_tensor, ortallocator, ortmemoryinfo);

  OrtTensorTypeAndShapeInfo* input_ids_info = ort.GetTensorTypeAndShape(input_ids_tensor);
  std::vector<int64_t> tensor_shape = ort.GetTensorShape(input_ids_info);
  ort.ReleaseTensorTypeAndShapeInfo(input_ids_info);

  int64_t batch_size = tensor_shape[0];
  int64_t seq_len = tensor_shape[1];

  // TODO, short cut to get the basic comparison with python version
  const OrtValue* max_length_tensor = ort.KernelContext_GetInput(context, 1);
  const int* max_length = ort.GetTensorData<int>(max_length_tensor);

  int iterations = max_length[0];
#ifdef DEBUG_BEAM_SEARCH
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

} //namespace custombsop