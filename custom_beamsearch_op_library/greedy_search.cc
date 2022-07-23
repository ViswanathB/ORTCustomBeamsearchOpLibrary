// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <cstdint>
#include <iostream>
#include <safeint.h>

#include "greedy_search.h"
#include "greedy_search_parameters.h"
#include "beam_search_device_helper.h"
#include "sequences.h"
#include "dump_tensor.h"
#include "logits_processor.h"
#include "utils.h"

using namespace std;
using namespace msl::utilities;
namespace custombsop {

template <typename T>
struct GreedySearchState : public custombsop::IGreedySearchState<T> {
  Sequences sequences;

  void Init(OrtAllocator* cpu_allocator,
            OrtAllocator*, /*allocator*/
            int batch_size,
            int vocab_size,
            int sequence_length,
            int max_length,
            bool /*is_cuda*/) {
    // below buffers are on cpu
    this->sequences_space = AllocateBufferUniquePtr<int32_t>(cpu_allocator,
                                                    sequences_space_buffer_,
                                                    SafeInt<size_t>(2) * batch_size * max_length);
    memset(this->sequences_space.data(), 0, this->sequences_space.size_bytes());
    this->sequences.Init(this->sequences_space, static_cast<int>(batch_size), sequence_length, max_length);

    this->sequence_lengths = AllocateBufferUniquePtr<int32_t>(cpu_allocator, sequence_lengths_buffer_, batch_size);
    this->eos_meet = AllocateBufferUniquePtr<bool>(cpu_allocator, eos_meet_buffer_, batch_size);
    memset(this->eos_meet.data(), 0, this->eos_meet.size_bytes());

    this->next_tokens_cpu = AllocateBufferUniquePtr<int64_t>(cpu_allocator,
                                                    next_tokens_cpu_buffer_,
                                                    SafeInt<size_t>(batch_size));
    this->next_tokens = AllocateBufferUniquePtr<int32_t>(cpu_allocator, next_tokens_buffer_, SafeInt<size_t>(batch_size));

    // below buffers are on cpu or cuda
    size_t next_token_size = SafeInt<size_t>(batch_size) * vocab_size;
    this->next_token_scores = AllocateBufferUniquePtr<T>(cpu_allocator, next_token_scores_buffer_, next_token_size);
    this->next_positions = AllocateBufferUniquePtr<int32_t>(cpu_allocator, next_positions_buffer_, batch_size);
  }

 private:
  BufferUniquePtr sequences_space_buffer_;
  BufferUniquePtr sequence_lengths_buffer_;
  BufferUniquePtr next_token_scores_buffer_;
  BufferUniquePtr next_tokens_buffer_;
  BufferUniquePtr next_tokens_cpu_buffer_;
  BufferUniquePtr next_positions_buffer_;
  BufferUniquePtr eos_meet_buffer_;
};

template <typename T>
class GreedySearchImpl {
 public:
  GreedySearchImpl(OrtApi &api,
                   Ort::CustomOpApi &ort,
                   OrtKernelContext* context,
                   //const SessionState& session_state,
                   //GptSubgraph& gpt_subgraph,
                   void* thread_pool,
                   void* cuda_stream,
                   //IConsoleDumper* cuda_dumper,
                   custombsop::GreedySearchParameters &params,
                   OrtAllocator *ort_allocator,
                   const BeamSearchDeviceHelper::CreateInputsFunc& create_inputs_func,
                   const BeamSearchDeviceHelper::AddToFeedsFunc& add_to_feeds_func,
                   //TODO next_thing_to_be_implemented.
                   //const BeamSearchDeviceHelper::TopkFunc& topk_func,
                   const BeamSearchDeviceHelper::GreedySearchProcessLogitsFunc<T>& process_logits_func,
                   const BeamSearchDeviceHelper::InitGreedyStateFunc<T>& init_greedy_state_func,
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
        init_greedy_state_func_(init_greedy_state_func),
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
      gsl::span<const int32_t> next_tokens);

  // Process logits and append next tokens to sequences.
  OrtStatusPtr GenerateNextToken(
      const OrtValue& logits,
      gsl::span<int32_t>& next_tokens,
      GreedySearchState<T>& greedy_state,
      int counter,
      int eos_token_id,
      std::unordered_map<std::string, OrtOp*> &ops_map);

  // Calculate scores from logits, then apply filtering and select next token for each beam.
  OrtStatusPtr ProcessLogits(
    const OrtValue& logits,  // logits output of subgraph
    GreedySearchState<T>& greedy_state,
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

  custombsop::GreedySearchParameters parameters_;

  LogitsProcessorList logits_processors_;

  OrtAllocator* cpu_allocator_;
  //AllocatorPtr temp_space_allocator_;

  // Device specific functions
  BeamSearchDeviceHelper::CreateInputsFunc create_inputs_func_;
  BeamSearchDeviceHelper::AddToFeedsFunc add_to_feeds_func_;
  //BeamSearchDeviceHelper::TopkFunc topk_func_;
  BeamSearchDeviceHelper::GreedySearchProcessLogitsFunc<T> process_logits_func_;
  BeamSearchDeviceHelper::InitGreedyStateFunc<T> init_greedy_state_func_;
  //BeamSearchDeviceHelper::DeviceCopyFunc<float> device_copy_func_;
  BeamSearchDeviceHelper::UpdateFeedsFunc<T> update_feeds_func_;
};

template <typename T>
OrtStatusPtr GreedySearchImpl<T>::ProcessLogits(
    const OrtValue& logits,
    GreedySearchState<T>& greedy_state,
    int counter,
    std::unordered_map<std::string, OrtOp*> &ops_map) {
  return process_logits_func_(context_, api_, ort_, logits, &greedy_state,
                              &(greedy_state.sequences), cpu_allocator_, thread_pool_, &logits_processors_,
                              &parameters_, counter, nullptr, GetConsoleDumper(),
                              ops_map);
}

template <typename T>
OrtStatusPtr GreedySearchImpl<T>::GenerateNextToken(
    const OrtValue& logits,
    gsl::span<int32_t>& next_tokens,
    GreedySearchState<T>& greedy_state,
    int counter,
    int eos_token_id,
    std::unordered_map<std::string, OrtOp*> &ops_map) {
  // Process logits to get next token scores
  CUSTOMOP_RETURN_IF_ERROR(ProcessLogits(logits, greedy_state, counter, ops_map));

  next_tokens = greedy_state.next_tokens;
    for (size_t i = 0; i < next_tokens.size(); i++) {
      next_tokens[i] = gsl::narrow_cast<int32_t>(greedy_state.next_tokens_cpu[i]);
    }

    gsl::span<bool>& eos_meet = greedy_state.eos_meet;
    for (size_t batch_id = 0; batch_id < next_tokens.size(); ++batch_id) {
      if (next_tokens[batch_id] == eos_token_id) {
        eos_meet[batch_id] = true;
        next_tokens[batch_id] = parameters_.pad_token_id;
      }
    }

    greedy_state.sequences.AppendNextTokenToSequences(next_tokens);

  #ifdef DEBUG_BEAM_SEARCH
    greedy_state.sequences.PrintSequences(&cpu_dumper_);
  #endif

 return nullptr;
}

template <typename T>
OrtStatusPtr GreedySearchImpl<T>::CheckInputs(const OrtKernelContext* context) {
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

  return nullptr;
}

template <typename T>
OrtStatusPtr GreedySearchImpl<T>::Initialize() {
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
OrtStatusPtr GreedySearchImpl<T>::UpdateFeeds(
    OrtMemoryInfo *ortmemoryinfo,
    std::vector<OrtValue*>& last_outputs,
    std::vector<OrtValue*>& next_inputs,
    int current_length,
    OrtValue* position_ids,
    gsl::span<const int32_t> next_tokens) {
  gsl::span<const int32_t> place_holder;
  return update_feeds_func_(api_, ort_, ortmemoryinfo, cpu_allocator_, nullptr, last_outputs, next_inputs, current_length, position_ids,
                            next_tokens, place_holder, parameters_.num_beams, gpt_subgraph_first_past_input_idx_,
                            gpt_subgraph_first_present_output_idx_, GetConsoleDumper());
}

template <typename T>
OrtStatusPtr GreedySearchImpl<T>::Execute(
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
  output1_dims.push_back(parameters_.max_length);
  OrtValue* sequences = ort_.KernelContext_GetOutput(context_, 0, output1_dims.data(), output1_dims.size());

  std::vector<OrtValue*> feeds;
  std::vector<OrtValue*> fetches;

  GreedySearchState<T> greedy_state;
  greedy_state.Init(ort_allocator,
                    ort_allocator, // not used
                    parameters_.batch_size,
                    parameters_.vocab_size,
                    parameters_.sequence_length,
                    parameters_.max_length,
                    this->IsCuda());

  OrtValue *expanded_inputs;
  OrtValue *expanded_position_ids;
  OrtValue *expanded_attention_mask;
  OrtStatusPtr status = create_inputs_func_(api_, ort_, original_input_ids,
            parameters_.num_beams, parameters_.pad_token_id, greedy_state.sequence_lengths, ort_allocator,
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


  gsl::span<const int32_t> input_ids = gsl::make_span(ort_.GetTensorData<int32_t>(expanded_inputs),
                                                      size_t(parameters_.batch_size) *  parameters_.sequence_length);

  init_greedy_state_func_(&greedy_state,
                          greedy_state.sequence_lengths,
                          parameters_.batch_size,
                          parameters_.sequence_length,
                          parameters_.max_length,
                          input_ids,
                          cuda_stream_);

  OrtValue *position_ids;
  std::vector<int64_t> position_ids_dims{parameters_.BatchBeamSize(), 1};
  api_.CreateTensorWithDataAsOrtValue(ortmemoryinfo, greedy_state.next_positions.data(), size_t(4)*parameters_.BatchBeamSize(), position_ids_dims.data(),
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

    iteration_counter++;

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

    gsl::span<int32_t> next_tokens;

    const OrtValue& logits_internal = *(fetches[0]);

    CUSTOMOP_RETURN_IF_ERROR(GenerateNextToken(logits_internal, next_tokens, greedy_state, iteration_counter, parameters_.eos_token_id, ops_map));

    gsl::span<bool>& eos_meet = greedy_state.eos_meet;
    size_t batch_id = 0;
    while (batch_id < eos_meet.size()) {
      if (eos_meet[batch_id] == false) {
        break;
      }
      ++batch_id;
    }
    if (batch_id == eos_meet.size()) {
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
                                      gsl::make_span(reinterpret_cast<const int32_t*>(next_tokens.data()), parameters_.BatchBeamSize())));

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

  int32_t* output_sequences_data = ort_.GetTensorMutableData<int32_t>(sequences);
  OrtTensorTypeAndShapeInfo* output_sequences_info = ort_.GetTensorTypeAndShape(sequences);
  std::vector<int64_t> output_sequences_shape = ort_.GetTensorShape(output_sequences_info);
  gsl::span<int32_t> output = gsl::make_span(output_sequences_data, SizeHelper(output_sequences_shape));

  for (int batch_id = 0; batch_id < parameters_.batch_size; ++batch_id) {
    auto batch_output = output.subspan(batch_id * parameters_.max_length,  parameters_.max_length);
    gsl::span<const int32_t> sequence_source = greedy_state.sequences.GetSequence(batch_id);
    gsl::copy(sequence_source, batch_output);
  }

  return nullptr;
}

OrtStatusPtr RunGreedySearchOnInternalSession(
      OrtKernelContext* context,
      OrtApi &api,
      Ort::CustomOpApi &ort,
      OrtSession *session,
      custombsop::GreedySearchParameters parameters,
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

  GreedySearchImpl<float>impl{api,
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
                              BeamSearchCpuDeviceHelper::GreedySearchProcessLogits<float>,
                              BeamSearchCpuDeviceHelper::InitGreedyState<float>,
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