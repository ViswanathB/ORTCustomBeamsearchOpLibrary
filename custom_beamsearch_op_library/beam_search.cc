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
#include "logits_processor.h"
#include "utils.h"

using namespace std;
using namespace msl::utilities;
namespace custombsop {

struct BeamSearchCpuState : public custombsop::IBeamSearchCpuState {
  custombsop::Sequences sequences;

  void Init(OrtAllocator* ort_allocator, size_t batch_beam_size, int max_length, bool is_cuda) {
    this->sequence_lengths = AllocateBufferUniquePtr<int32_t>(ort_allocator, sequence_lengths_buffer_, batch_beam_size);
    this->sequences_space = AllocateBufferUniquePtr<int32_t>(ort_allocator, sequences_space_buffer_, size_t(2) * batch_beam_size * max_length);

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
  BufferUniquePtr sequence_lengths_buffer_;
  BufferUniquePtr sequences_space_buffer_;

  std::unique_ptr<float> topk_scores_buffer_;
  std::unique_ptr<int32_t> topk_tokens_buffer_;
  std::unique_ptr<int32_t> topk_indices_buffer_;
  std::unique_ptr<float> final_beam_scores_buffer_;
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
    //TODO Purpose of SafeInt, is it really helping?
    size_t batch_beam_size = SafeInt<size_t>(batch_size) * num_beams;
    size_t next_token_size = SafeInt<size_t>(batch_beam_size) * vocab_size;

    this->next_token_logits = AllocateBufferUniquePtr<T>(allocator, next_token_logits_buffer_, next_token_size);
    this->next_token_scores = AllocateBufferUniquePtr<float>(allocator, next_token_scores_buffer_, next_token_size);
    this->next_tokens = AllocateBufferUniquePtr<int32_t>(allocator, next_tokens_buffer_, SafeInt<size_t>(2) * batch_beam_size);
    this->next_indices = AllocateBufferUniquePtr<int32_t>(allocator, next_indices_buffer_, SafeInt<size_t>(2) * batch_beam_size);
    this->next_positions = AllocateBufferUniquePtr<int32_t>(allocator, next_positions_buffer_, batch_beam_size);
    this->beam_scores = AllocateBufferUniquePtr<float>(allocator, beam_scores_buffer_, batch_beam_size, true);

    if (output_scores) {
      size_t elements = SafeInt<size_t>(max_length - sequence_length) * batch_size * num_beams * vocab_size;
      this->scores = AllocateBufferUniquePtr<float>(allocator, scores_buffer_, elements);
      this->remaining_scores = this->scores;
    }
  }

 private:
  BufferUniquePtr next_token_logits_buffer_;
  BufferUniquePtr next_token_scores_buffer_;
  BufferUniquePtr next_tokens_buffer_;
  BufferUniquePtr next_indices_buffer_;
  BufferUniquePtr next_positions_buffer_;
  BufferUniquePtr beam_scores_buffer_;
  BufferUniquePtr scores_buffer_;
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
                 OrtAllocator *ort_allocator,
                 const BeamSearchDeviceHelper::CreateInputsFunc& create_inputs_func,
                 const BeamSearchDeviceHelper::AddToFeedsFunc& add_to_feeds_func,
                 //TODO next_thing_to_be_implemented.
                 //const BeamSearchDeviceHelper::TopkFunc& topk_func,
                 const BeamSearchDeviceHelper::ProcessLogitsFunc<T>& process_logits_func,
                 const BeamSearchDeviceHelper::InitBeamStateFunc<T>& init_beam_state_func,
                 //const BeamSearchDeviceHelper::DeviceCopyFunc<float>& device_copy_func,
                 const BeamSearchDeviceHelper::UpdateFeedsFunc<T>& update_feeds_func)
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
        cpu_allocator_(ort_allocator),
        //temp_space_allocator_(nullptr),
        create_inputs_func_(create_inputs_func),
        add_to_feeds_func_(add_to_feeds_func),
        //topk_func_(topk_func),
        process_logits_func_(process_logits_func),
        init_beam_state_func_(init_beam_state_func),
        //device_copy_func_(device_copy_func),
        update_feeds_func_(update_feeds_func) {

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
  OrtStatusPtr Execute(
    const OrtValue *original_input_ids,
    OrtAllocator *ort_allocator,
    OrtMemoryInfo *ortmemoryinfo,
    OrtSession* session,
    std::unordered_map<std::string, OrtOp*> &ops_map);

 private:
  bool IsCuda() const { return cuda_stream_ != nullptr; }

  // Validate inputs.
  OrtStatusPtr CheckInputs(const OrtKernelContext* context);

  // Prepare the inputs for first inference of subgraph
  //OrtStatusPtr CreateInitialFeeds(gsl::span<int32_t>& sequence_lengths, OrtValue& expanded_input_ids, std::vector<OrtValue>& feeds, IAllocatorUniquePtr<char>& buffer);

  // Update the input for next iteration.
  OrtStatusPtr UpdateFeeds(
      OrtMemoryInfo *ortmemoryinfo,
      std::vector<OrtValue*>& last_outputs,
      std::vector<OrtValue*>& next_inputs,
      int current_length,
      OrtValue* position_ids,
      gsl::span<const int32_t> beam_next_tokens,
      gsl::span<const int32_t> beam_indices);

  // Process logits and append next tokens to sequences.
  OrtStatusPtr GenerateNextToken(
                          const OrtValue& logits,
                          gsl::span<int32_t>& beam_next_tokens,
                          gsl::span<int32_t>& beam_indices,
                          BeamSearchState<T>& beam_state,
                          BeamSearchCpuState& cpu_state,
                          int counter,
                          std::unordered_map<std::string, OrtOp*> &ops_map);

  // Calculate scores from logits, then apply filtering and select next token for each beam.
  OrtStatusPtr ProcessLogits(const OrtValue& logits,  // logits output of subgraph
                       BeamSearchState<T>& beam_state,
                       BeamSearchCpuState& cpu_state,
                       int counter,
                       std::unordered_map<std::string, OrtOp*> &ops_map);

  const custombsop::IConsoleDumper* GetConsoleDumper() const { return IsCuda() ? cuda_dumper_ : &(cpu_dumper_); }

  OrtApi api_;

  Ort::CustomOpApi ort_;

  OrtKernelContext* context_;

  //const SessionState& session_state_;

  //GptSubgraph& gpt_subgraph_;

  void* thread_pool_;

  int gpt_subgraph_first_past_input_idx_;

  int gpt_subgraph_first_present_output_idx_;

  //const std::vector<const OrtValue*>& implicit_inputs_;

  void* cuda_stream_;

  custombsop::IConsoleDumper* cuda_dumper_;
  custombsop::CpuTensorConsoleDumper cpu_dumper_;

  custombsop::BeamSearchParameters parameters_;

  LogitsProcessorList logits_processors_;

  std::unique_ptr<custombsop::BeamSearchScorer> beam_scorer_;

  OrtAllocator* cpu_allocator_;
  //AllocatorPtr temp_space_allocator_;

  // Device specific functions
  BeamSearchDeviceHelper::CreateInputsFunc create_inputs_func_;
  BeamSearchDeviceHelper::AddToFeedsFunc add_to_feeds_func_;
  //BeamSearchDeviceHelper::TopkFunc topk_func_;
  BeamSearchDeviceHelper::ProcessLogitsFunc<T> process_logits_func_;
  BeamSearchDeviceHelper::InitBeamStateFunc<T> init_beam_state_func_;
  //BeamSearchDeviceHelper::DeviceCopyFunc<float> device_copy_func_;
  BeamSearchDeviceHelper::UpdateFeedsFunc<T> update_feeds_func_;
};

template <typename T>
OrtStatusPtr BeamSearchImpl<T>::ProcessLogits(
    const OrtValue& logits,
    BeamSearchState<T>& beam_state,
    BeamSearchCpuState& cpu_state,
    int counter,
    std::unordered_map<std::string, OrtOp*> &ops_map) {
  return process_logits_func_(context_, api_, ort_, logits, &beam_state, &cpu_state,
                              &(cpu_state.sequences), cpu_allocator_, thread_pool_, &logits_processors_,
                              beam_scorer_.get(), &parameters_, counter, nullptr, GetConsoleDumper(),
                              ops_map);
}

template <typename T>
OrtStatusPtr BeamSearchImpl<T>::GenerateNextToken(
    const OrtValue& logits,
    gsl::span<int32_t>& beam_next_tokens,
    gsl::span<int32_t>& beam_indices,
    BeamSearchState<T>& beam_state,
    BeamSearchCpuState& cpu_state,
    int counter,
    std::unordered_map<std::string, OrtOp*> &ops_map) {
  // Process logits to get next token scores
  CUSTOMOP_RETURN_IF_ERROR(ProcessLogits(logits, beam_state, cpu_state, counter, ops_map));

  gsl::span<float>& beam_scores = beam_scorer_->GetNextScores();
  // It is optional to clone beam_scores. Change it to use same buffer also works for CPU:
  //    beam_state.beam_scores = beam_scores
  // Here we make a copy to reduce the coupling with little cost (the buffer size is small).
  
  //TODO This is good enough for CPU
  // FOR CUDA might need something else
  // ORT_RETURN_IF_ERROR(device_copy_func_(beam_state.beam_scores, beam_scores, cuda_stream_, DeviceCopyDirection::hostToDevice));
  gsl::copy(beam_scores, beam_state.beam_scores);
  
  beam_next_tokens = beam_scorer_->GetNextTokens();
  beam_indices = beam_scorer_->GetNextIndices();

#ifdef DEBUG_BEAM_SEARCH
  cpu_dumper_.Print("beam_scores after scorer", beam_scores.data(), parameters_.batch_size, parameters_.num_beams);
  cpu_dumper_.Print("beam_next_tokens after scorer", beam_next_tokens.data(), parameters_.batch_size, parameters_.num_beams);
  cpu_dumper_.Print("beam_indices after scorer", beam_indices.data(), parameters_.batch_size, parameters_.num_beams);
#endif

  cpu_state.sequences.AppendNextTokenToSequences(beam_indices, beam_next_tokens);

#ifdef DEBUG_BEAM_SEARCH
  cpu_state.sequences.PrintSequences(&cpu_dumper_);
#endif

 return nullptr;
}

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

  //TODO is this flag even needed
  if (!IsCuda()) {
    // Logits processor is used in CPU only. In CUDA, cuda kernels are used instead.
    // Initialize processsors after CheckInputs so that parameters_->vocab_mask is ready.
    logits_processors_.Init(parameters_);
  }

  //TODO this has to be extracted somewhere, mostly config file or when internal session is created.
  gpt_subgraph_first_past_input_idx_ = 3;
  gpt_subgraph_first_present_output_idx_ = 1;
  
  return nullptr;
}

template <typename T>
OrtStatusPtr BeamSearchImpl<T>::UpdateFeeds(
    OrtMemoryInfo *ortmemoryinfo,
    std::vector<OrtValue*>& last_outputs,
    std::vector<OrtValue*>& next_inputs,
    int current_length,
    OrtValue* position_ids,
    gsl::span<const int32_t> beam_next_tokens,
    gsl::span<const int32_t> beam_indices) {
  return update_feeds_func_(api_, ort_, ortmemoryinfo, cpu_allocator_, nullptr, last_outputs, next_inputs, current_length, position_ids,
                            beam_next_tokens, beam_indices, parameters_.num_beams, gpt_subgraph_first_past_input_idx_,
                            gpt_subgraph_first_present_output_idx_, GetConsoleDumper());
}

template <typename T>
OrtStatusPtr BeamSearchImpl<T>::Execute(
              const OrtValue *original_input_ids,
              OrtAllocator *ort_allocator,
              OrtMemoryInfo *ortmemoryinfo,
              OrtSession* session,
              std::unordered_map<std::string, OrtOp*> &ops_map) {
#ifdef DEBUG_BEAM_SEARCH
  std::cout<<"Calling create inputs:"<<std::endl;
#endif

  std::vector<int64_t> output1_dims;
  output1_dims.push_back(parameters_.batch_size);
  output1_dims.push_back(parameters_.num_return_sequences);
  output1_dims.push_back(parameters_.max_length);
  OrtValue* sequences = ort_.KernelContext_GetOutput(context_, 0, output1_dims.data(), output1_dims.size());

  std::vector<int64_t> output2_dims;
  output2_dims.push_back(parameters_.batch_size);
  output2_dims.push_back(parameters_.num_return_sequences);
  OrtValue* sequences_scores = ort_.KernelContext_GetOutput(context_, 1, output2_dims.data(), output2_dims.size());

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

  OrtValue *position_ids;
  std::vector<int64_t> position_ids_dims{parameters_.BatchBeamSize(), 1};
  api_.CreateTensorWithDataAsOrtValue(ortmemoryinfo, beam_state.next_positions.data(), size_t(4)*parameters_.BatchBeamSize(), position_ids_dims.data(),
      position_ids_dims.size(), ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &position_ids);

  int current_length = parameters_.sequence_length;
  int iteration_counter = 0;
  int past_seq_len = 0;
  int seq_len = current_length;
  int batch_size = parameters_.batch_size;
  int num_heads = parameters_.num_heads;
  int head_size = parameters_.head_size;

  std::vector<const char*> input_names{"input_ids", "position_ids", "attention_mask", "past_0", "past_1", "past_2", "past_3", "past_4", "past_5"};
  std::vector<int64_t> past_dims{2, batch_size*parameters_.num_beams, num_heads, 0, head_size};
  std::vector<float> past_data;
  
  for (int i = 0; i < 6; i++) {
    OrtValue *ort_past_data;
    api_.CreateTensorWithDataAsOrtValue(ortmemoryinfo, past_data.data(), 0, past_dims.data(), past_dims.size(),
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &ort_past_data);

    feeds.emplace_back(std::move(ort_past_data));
  }

  std::vector<const char*> output_names{"logits", "present_0", "present_1", "present_2", "present_3", "present_4", "present_5"};
  //std::vector<float> logits_data(batch_size * seq_len * 50263, 0);
  std::vector<int64_t> logits_dims{batch_size, seq_len, 50263};
  OrtValue* ortlogits;
  api_.CreateTensorAsOrtValue(ort_allocator, logits_dims.data(), logits_dims.size(),
                ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &ortlogits);
  fetches.emplace_back(ortlogits);

  std::vector<int64_t> present_dims{2, batch_size, num_heads, seq_len, head_size};
  for (int i = 0; i < 6; i++) {
    OrtValue* ort_present;
    // ort.present_1, ort.present_2, ort.present_3, ort.present_4, ort.present_5;
    api_.CreateTensorAsOrtValue(ort_allocator, present_dims.data(), present_dims.size(), 
                ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &ort_present);
    fetches.emplace_back(ort_present);
  }
  
  while (current_length < parameters_.max_length) {
    //std::cout<<"Inside the beam search loop at iteration:" << iteration_counter+1<<std::endl;

    iteration_counter ++;

#ifdef DEBUG_BEAM_SEARCH
    auto cur_len = std::to_string(current_length);
    std::cout<<"CurrentLength:"<<cur_len<<std::endl;
#endif

#ifdef DEBUG_BEAM_SEARCH
  cpu_dumper_.Print("input_ids", ort_.GetTensorMutableData<float>(feeds[0]), parameters_.batch_size * parameters_.num_beams, current_length-past_seq_len);
  if (iteration_counter > 1) {
    cpu_dumper_.Print("past_0", ort_.GetTensorMutableData<float>(feeds[3]), 2 * batch_size * num_heads, current_length-1, head_size);
    cpu_dumper_.Print("past_1", ort_.GetTensorMutableData<float>(feeds[4]), 2 * batch_size * num_heads, current_length-1, head_size);
  }
#endif

    //TODO resuse the input and output buffers.
    // All of them are actually from stack, how is this working?? Is the CreateTensorWithDataAsValue moving them to heap??
    CUSTOMOP_RETURN_IF_ERROR(api_.Run(session, nullptr, input_names.data(), feeds.data(), 9, output_names.data(), 7, fetches.data()));

#ifdef DEBUG_BEAM_SEARCH
  cpu_dumper_.Print("logits", ort_.GetTensorMutableData<float>(fetches[0]), batch_size, seq_len, 50263);
  cpu_dumper_.Print("present_0", ort_.GetTensorMutableData<float>(fetches[1]), 2 * batch_size * num_heads, current_length, head_size);
  cpu_dumper_.Print("present_1", ort_.GetTensorMutableData<float>(fetches[2]), 2 * batch_size * num_heads, current_length, head_size);
#endif

    gsl::span<int32_t> beam_next_tokens;
    gsl::span<int32_t> beam_indices;

    const OrtValue& logits_internal = *(fetches[0]);

    CUSTOMOP_RETURN_IF_ERROR(GenerateNextToken(logits_internal, beam_next_tokens, beam_indices, beam_state, cpu_state, iteration_counter, ops_map));

    // When all batches are finished, stop earlier to avoid wasting computation.
    if (beam_scorer_->IsDone()) {
      break;
    }

    // Increase sequence length after a new token is generated.
    ++current_length;

    if (current_length < parameters_.max_length) {
      CUSTOMOP_RETURN_IF_ERROR(UpdateFeeds(ortmemoryinfo,
                                      fetches,
                                      feeds,
                                      current_length,
                                      position_ids,
                                      gsl::make_span(reinterpret_cast<const int32_t*>(beam_next_tokens.data()), parameters_.BatchBeamSize()),
                                      gsl::make_span(reinterpret_cast<const int32_t*>(beam_indices.data()), parameters_.BatchBeamSize())));

      //Only logits are released, all others are either released or used in next iteration as inputs
      api_.ReleaseValue(fetches[0]);
      fetches.clear();

      //TODO Get the new outputs ready
      past_seq_len += seq_len; 
      seq_len = 1;

      std::vector<int64_t> logits_dims{parameters_.batch_size, parameters_.num_beams, 50263};
      OrtValue* ortlogits;
      api_.CreateTensorAsOrtValue(ort_allocator, logits_dims.data(), logits_dims.size(),
                ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &ortlogits);
      fetches.emplace_back(ortlogits);

#if DEBUG_BEAM_SEARCH
      std::cout<<"Past seq len after iteration "<<iteration_counter<<":"<<past_seq_len<<std::endl;
      std::cout<<"Size of outputs :"<<fetches.size()<<std::endl;
#endif

      //std::vector<float> present_0_data(2 * batch_size * num_heads * (seq_len+past_seq_len) * head_size, 0);
      std::vector<int64_t> present_dims{2, batch_size, num_heads, seq_len+past_seq_len, head_size};
      for (int i = gpt_subgraph_first_present_output_idx_; i < gpt_subgraph_first_present_output_idx_+6; i++) {
        OrtValue* ort_present;
        api_.CreateTensorAsOrtValue(ort_allocator, present_dims.data(), present_dims.size(),
                              ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &ort_present);
        
        fetches.emplace_back(ort_present);
      }
    }
  }

  //Clean all feeds and fetches
  for (int i=0; i<feeds.size(); i++) {
    api_.ReleaseValue(feeds[i]);
  }

  for(int i=0;i<fetches.size();i++) {
    api_.ReleaseValue(fetches[i]);
  }

  gsl::span<const float> final_beam_scores(beam_state.beam_scores.data(), beam_state.beam_scores.size());
  // TODO For cuda, these scores will be copied over from cuda
  /*
  if (this->IsCuda()) {
    ORT_RETURN_IF_ERROR(this->device_copy_func_(cpu_state.final_beam_scores,
                                              final_beam_scores,
                                              nullptr,
                                              DeviceCopyDirection::deviceToHost));
    final_beam_scores = gsl::make_span<const float>(cpu_state.final_beam_scores.data(),
                                                  cpu_state.final_beam_scores.size());
  }
  */

  beam_scorer_->Finalize(
                api_,
                ort_,
                &(cpu_state.sequences),
                final_beam_scores,
                sequences,
                sequences_scores);

  // Output per token scores
  //TODO output scores is an optional third output that is not supported now
  /*if (output_scores != nullptr) {
      gsl::span<float> target = output_scores->MutableDataAsSpan<float>();
      gsl::span<const float> source = gsl::span<const float>(beam_state.scores.data(), beam_state.scores.size());
      assert(target.length() == source.length());
      ORT_RETURN_IF_ERROR(this->device_copy_func_(target, source, nullptr, DeviceCopyDirection::deviceToDevice));
    }
  }*/

  return nullptr;
}

OrtStatusPtr RunBeamSearchOnInternalSession(
      OrtKernelContext* context,
      OrtApi &api,
      Ort::CustomOpApi &ort,
      OrtSession *session,
      custombsop::BeamSearchParameters parameters,
      std::unordered_map<std::string, OrtOp*> &ops_map) {
  std::vector<OrtValue*> inputs;
  std::vector<const char*> input_names{"input_ids", "position_ids", "attention_mask", "past_0", "past_1", "past_2", "past_3", "past_4", "past_5"};

  // Both of the following should provide the same thing, one for memory info and other for ort allocator.
  OrtMemoryInfo *ortmemoryinfo;
  // Must be freed explicitly
  //api.CreateMemoryInfo("Cpu", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeCPU, &ortmemoryinfo);
  api.CreateMemoryInfo("Cpu", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemType::OrtMemTypeDefault, &ortmemoryinfo);

  OrtAllocator *ortallocator;
  //api.GetAllocatorWithDefaultOptions(&ortallocator);
  api.CreateAllocator(session, ortmemoryinfo, &ortallocator);

  //void* thread_pool = ort.KernelContext_GetThreadPool(context);
  void* thread_pool = nullptr;
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
                            ortallocator,
                            BeamSearchCpuDeviceHelper::CreateInputs,
                            BeamSearchCpuDeviceHelper::AddToFeeds,
                            //topk_func_ ? topk_func_ : BeamSearchCpuDeviceHelper::TopK,
                            BeamSearchCpuDeviceHelper::ProcessLogits<float>,
                            BeamSearchCpuDeviceHelper::InitBeamState<float>,
                            //device_copy_func_ ? device_copy_func_ : BeamSearchCpuDeviceHelper::DeviceCopy<float>,
                            BeamSearchCpuDeviceHelper::UpdateFeeds<float>};
  OrtStatusPtr status_ptr = impl.Initialize();
  if (status_ptr != nullptr) {
    //TODO handle better than abort
    std::cout<<" Beam search is not initialized properly : "<<api.GetErrorMessage(status_ptr)<<std::endl;
    abort();
  }

  const OrtValue* input_ids_tensor = ort.KernelContext_GetInput(context, 0);
  const int* input_ids = ort.GetTensorData<int>(input_ids_tensor);
  
  OrtStatusPtr status = impl.Execute(input_ids_tensor, ortallocator, ortmemoryinfo, session, ops_map);
  if (status != nullptr) {
    std::cout<<api.GetErrorMessage(status)<<std::endl;
  }

  return nullptr;
}

} //namespace custombsop