#include "beam_search_device_helper.h"

#include <safeint.h>
#include <gsl/span>
#include <gsl/gsl>

using namespace std;
using namespace msl::utilities;

namespace BeamSearchCpuDeviceHelper
{

  // Copy present state to past state for GPT model
  template <typename T>
  void PickGptPastState(
      OrtApi &api,
      Ort::CustomOpApi &ort,
      std::vector<OrtValue *> &last_outputs,
      std::vector<OrtValue *> &next_inputs,
      gsl::span<const int32_t> &beam_indices,
      int gpt_subgraph_first_past_input_idx,
      int gpt_subgraph_first_present_output_idx,
      OrtAllocator *ort_allocator)
  {

    int num_present_tensors = static_cast<int>(last_outputs.size()) - gpt_subgraph_first_present_output_idx;
    for (int i = 0; i < num_present_tensors; ++i)
    {
      const OrtValue *present = last_outputs[gpt_subgraph_first_present_output_idx + i];
      OrtTensorTypeAndShapeInfo *tsinfo = ort.GetTensorTypeAndShape(present);
      std::vector<int64_t> past_shape = ort.GetTensorShape(tsinfo);
      int64_t block_size_per_beam = past_shape[2] * past_shape[3] * past_shape[4];
      int64_t past_key_size = past_shape[1] * past_shape[2] * past_shape[3] * past_shape[4];

      OrtValue *past;
      // TODO datatypeimpl hard coding past state to float, getting type from T which is float and then have
      //  a mapping to ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
      api.CreateTensorAsOrtValue(ort_allocator, past_shape.data(), past_shape.size(),
                                 ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &past);

      gsl::span<T> past_span = gsl::make_span<T>(ort.GetTensorMutableData<T>(past), SizeHelper(past_shape));
      gsl::span<const T> present_span = gsl::make_span<const T>(ort.GetTensorData<T>(present), SizeHelper(past_shape));
      for (gsl::index j = 0; j < beam_indices.size(); j++)
      {
        int32_t beam_index = beam_indices[j];
        gsl::span<const T> present_key = present_span.subspan(beam_index * block_size_per_beam, block_size_per_beam);
        gsl::span<const T> present_value = present_span.subspan(past_key_size + beam_index * block_size_per_beam,
                                                                block_size_per_beam);

        gsl::span<T> past_key = past_span.subspan(j * block_size_per_beam, block_size_per_beam);
        gsl::span<T> past_value = past_span.subspan(past_key_size + j * block_size_per_beam, block_size_per_beam);
        gsl::copy(present_key, past_key);
        gsl::copy(present_value, past_value);

        api.ReleaseValue(last_outputs[gpt_subgraph_first_present_output_idx + i]);
      }

      api.ReleaseValue(next_inputs[gpt_subgraph_first_past_input_idx + i]);
      next_inputs[gpt_subgraph_first_past_input_idx + i] = past;
    }
  }

  template <typename T>
  OrtStatusPtr UpdateFeeds(
      OrtApi &api,
      Ort::CustomOpApi &ort,
      OrtMemoryInfo *ortmemoryinfo,
      OrtAllocator *ort_allocator,
      void *stream,
      std::vector<OrtValue *> &last_outputs,
      std::vector<OrtValue *> &next_inputs,
      int current_length,
      OrtValue *position_ids,
      gsl::span<const int32_t> beam_next_tokens,
      gsl::span<const int32_t> beam_indices,
      int num_beams,
      int gpt_subgraph_first_past_input_idx,
      int gpt_subgraph_first_present_output_idx,
      const custombsop::IConsoleDumper *dumper)
  {

    int batch_beam_size = static_cast<int>(beam_next_tokens.size());
    std::vector<int64_t> input_dims = {batch_beam_size, 1};

    OrtValue *input_ids;
    // TODO Creating new inputs here using allocator, don't do this, reuse the buffer some way

    api.CreateTensorAsOrtValue(ort_allocator, input_dims.data(), input_dims.size(),
                               ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &input_ids);

    int32_t *input_ids_data = ort.GetTensorMutableData<int32_t>(input_ids);
    for (int i = 0; i < batch_beam_size; i++)
    {
      input_ids_data[i] = beam_next_tokens[i];
    }

    // TODO what happens to the original input_ids
    // since there were creating on the stack they just get deallocated properly.
    api.ReleaseValue(next_inputs[0]);
    next_inputs[0] = input_ids;

    int32_t *position_ids_data = ort.GetTensorMutableData<int32_t>(position_ids);
    for (int i = 0; i < batch_beam_size; i++)
    {
      position_ids_data[i]++;
    }
    next_inputs[1] = position_ids;

    // Update attention mask
    OrtValue *old_mask = next_inputs[2];
    int32_t *old_mask_data = ort.GetTensorMutableData<int32_t>(old_mask);

    OrtValue *new_mask;
    std::vector<int64_t> mask_dims = {batch_beam_size, current_length};
    api.CreateTensorAsOrtValue(ort_allocator, mask_dims.data(), mask_dims.size(),
                               ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &new_mask);

    int32_t *new_mask_data = ort.GetTensorMutableData<int32_t>(new_mask);
    for (int i = 0; i < batch_beam_size; i++)
    {
      for (int j = 0; j < current_length - 1; j++)
      {
        new_mask_data[i * current_length + j] = old_mask_data[i * (current_length - 1) + j];
      }
      new_mask_data[i * current_length + current_length - 1] = 1;
    }
    api.ReleaseValue(next_inputs[2]);
    next_inputs[2] = new_mask;

#ifdef DEBUG_BEAM_SEARCH
    dumper->Print("input_ids", input_ids_data, batch_beam_size, 1);
    dumper->Print("position_ids", position_ids_data, batch_beam_size, 1);
    dumper->Print("attention_mask", new_mask_data, batch_beam_size, current_length);
#endif

    // Update past state
    if (num_beams == 1)
    {
      // feed present_* output to past_* inputs one by one
      const int k = gpt_subgraph_first_past_input_idx - gpt_subgraph_first_present_output_idx;
      for (size_t i = gpt_subgraph_first_present_output_idx; i < last_outputs.size(); ++i)
      {
        api.ReleaseValue(next_inputs[i + k]);
        next_inputs[i + k] = last_outputs[i];
      }
    }
    else
    {
      PickGptPastState<T>(api, ort, last_outputs, next_inputs, beam_indices,
                          gpt_subgraph_first_past_input_idx,
                          gpt_subgraph_first_present_output_idx, ort_allocator);
    }

    return nullptr;
  }

  OrtStatusPtr AddToFeeds(OrtValue *input_ids,
                          OrtValue *position_ids,
                          OrtValue *attention_mask,
                          std::vector<OrtValue *> &feeds)
  {
    feeds.push_back(std::move(input_ids));
    feeds.push_back(std::move(position_ids));
    feeds.push_back(std::move(attention_mask));
    return nullptr;
  }

  OrtValue *ExpandInputs(OrtApi &api, Ort::CustomOpApi &ort, OrtValue *input, int num_beams, OrtAllocator *allocator, ONNXTensorElementDataType element_type)
  {
    // Input shape (batch_size, sequence_length)
    // Output shape (batch_size * num_beams, sequence_length)
    if (num_beams == 1)
      return input;

    const OrtTensorTypeAndShapeInfo *input_info = ort.GetTensorTypeAndShape(input);
    std::vector<int64_t> input_shape = ort.GetTensorShape(input_info);

    const int64_t &batch_size = input_shape[0];
    const int64_t &sequence_length = input_shape[1];

    vector<int64_t> dims{batch_size * num_beams, sequence_length};

    CUSTOMOP_ENFORCE(element_type == ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
                     "input_ids, position_ids and attention_mask is required to be int32 data type");

    OrtValue *expanded;
    api.CreateTensorAsOrtValue(allocator, dims.data(), dims.size(), element_type, &expanded);

    const int32_t *input_data = ort.GetTensorData<int32_t>(input);
    int32_t *expanded_data = ort.GetTensorMutableData<int32_t>(expanded);
    int32_t *target = expanded_data;
    for (int i = 0; i < batch_size; i++)
    {
      for (int j = 0; j < num_beams; j++)
      {
        memcpy(target, input_data + i * sequence_length, sizeof(int32_t) * sequence_length);
        target += sequence_length;
      }
    }

    return expanded;
  }

  template <typename T>
  OrtStatusPtr ProcessLogits(
      OrtKernelContext *context,
      OrtApi &api,
      Ort::CustomOpApi &ort,
      const OrtValue &logits,                              // logits output of subgraph
      custombsop::IBeamSearchState<T> *beam_state,         // state
      custombsop::IBeamSearchCpuState *cpu_state,          // state in CPU
      custombsop::ISequences *sequences,                   // sequences
      OrtAllocator *ort_allocator,                         // default allocator
      custombsop::ILogitsProcessorList *logits_processors, // logits processors
      custombsop::IBeamScorer *beam_scorer,                // beam scorer
      custombsop::IBeamSearchParameters *parameters,       // parameters
      int step,                                            // iteration counter
      const custombsop::IConsoleDumper *dumper,            // tensor dumper
      std::unordered_map<std::string, OrtOp *> &ops_map)
  {
    int batch_size = parameters->batch_size;
    int num_beams = parameters->num_beams;
    int vocab_size = parameters->vocab_size;
    bool output_scores = parameters->output_scores;

    int batch_beam_size = batch_size * num_beams;
    const T *logits_data = ort.GetTensorData<T>(&logits);

    // Logits has shape (batch_size * num_beams, input_length, vocab_size),
    // where input_length equals to parameters_->sequence_length for first subgraph call, and 1 for the remaining calls.
    OrtTensorTypeAndShapeInfo *logits_data_info = ort.GetTensorTypeAndShape(&logits);
    std::vector<int64_t> logits_shape = ort.GetTensorShape(logits_data_info);

    CUSTOMOP_ENFORCE(logits_shape.size() == 3);
    auto input_length = logits_shape[1];
    auto logits_batch_size = logits_shape[0];

#ifdef DEBUG_BEAM_SEARCH
    std::cout << "logits shape:" << logits_shape[0] << "," << logits_shape[1] << "," << logits_shape[2] << std::endl;
#endif

    // Get logits for the last token:
    //    next_token_logits = logits[:, -1, :], and the result shape is (batch_size * num_beams, vocab_size)
    // When input_length == 1, use logits directly in SoftmaxCPU below so it only need for input_length > 1.
    gsl::span<T> &next_token_logits = beam_state->next_token_logits;
    if (input_length > 1)
    {
      const T *current_logits = logits_data + (input_length - 1) * vocab_size;
      for (int i = 0; i < batch_beam_size; i++)
      {
        gsl::span<const T> source(current_logits, vocab_size);
        gsl::span<T> target = next_token_logits.subspan(SafeInt<gsl::index>(i) * vocab_size, static_cast<gsl::index>(vocab_size));
        gsl::copy(source, target);
        current_logits += input_length * vocab_size;
      }
    }
    else
    {
      const T *current_logits = logits_data;
      gsl::span<const T> source(current_logits, batch_beam_size * vocab_size);
      gsl::span<T> target = next_token_logits.subspan(0, batch_beam_size * vocab_size);
      gsl::copy(source, target);
    }

#ifdef DEBUG_BEAM_SEARCH
    std::cout << "Executing inside ProcessLogits" << std::endl;
    dumper->Print("next_token_logits", next_token_logits.data(), batch_size, num_beams, vocab_size);
#endif

    // Get scores for candidates of next token: next_token_scores = log_softmax(next_token_logits, dim=-1)
    gsl::span<T> &next_token_scores = beam_state->next_token_scores;

    OrtMemoryInfo *ortmemoryinfo;
    // Must be freed explicitly
    api.CreateMemoryInfo("Cpu", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeCPU, &ortmemoryinfo);

    OrtValue *softmax_input;
    std::vector<int64_t> softmax_input_dims{batch_size, num_beams * vocab_size};
    api.CreateTensorWithDataAsOrtValue(ortmemoryinfo, next_token_logits.data(), size_t(4) * batch_size * num_beams * vocab_size, softmax_input_dims.data(),
                                       softmax_input_dims.size(), ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &softmax_input);
    const OrtValue *softmax_inputs[1] = {reinterpret_cast<const OrtValue *>(softmax_input)};

    OrtValue *softmax_output;
    api.CreateTensorWithDataAsOrtValue(ortmemoryinfo, next_token_scores.data(), size_t(4) * batch_size * num_beams * vocab_size, softmax_input_dims.data(),
                                       softmax_input_dims.size(), ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &softmax_output);
    OrtValue *softmax_outputs[1] = {softmax_output};

    OrtOp *softmax = ops_map[std::string("softmax")];
    ort.InvokeOp(context, reinterpret_cast<const OrtOp *>(softmax), softmax_inputs, 1, softmax_outputs, 1);

#ifdef DEBUG_BEAM_SEARCH
    dumper->Print("next_token_scores after softmax", next_token_scores.data(), batch_size, num_beams, vocab_size);
#endif

    // Apply all score processors that updates scores
    logits_processors->Process(sequences, next_token_scores, step);

#ifdef DEBUG_BEAM_SEARCH
    dumper->Print("next_token_scores after logits processor", next_token_scores.data(), batch_size, num_beams, vocab_size);
#endif

    // Add beam score to next token scores. Corresponding python code is like:
    //    next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
    // TODO: use thread pool to parrellel
    int offset = 0;
    int batch_beam_index = 0;
    for (int i = 0; i < batch_size; i++)
    {
      for (int j = 0; j < num_beams; j++, batch_beam_index++)
      {
        for (int k = 0; k < vocab_size; k++, offset++)
        {
          next_token_scores[offset] += beam_state->beam_scores[batch_beam_index];
        }
      }
    }

#ifdef DEBUG_BEAM_SEARCH
    dumper->Print("batch beam_scores", beam_state->beam_scores.data(), batch_size, num_beams);
#endif

#ifdef DEBUG_BEAM_SEARCH
    dumper->Print("next_token_scores after adding beam_scores", next_token_scores.data(), batch_size, num_beams, vocab_size);
#endif

    if (output_scores)
    {
      // Append next token scores to the scores output.
      gsl::copy(next_token_scores, beam_state->remaining_scores);
      beam_state->remaining_scores = beam_state->remaining_scores.subspan(next_token_scores.size());
    }

    // Apply top-k selection like the following:
    //   next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
    //   next_token_scores, next_tokens = torch.topk(next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

    const int32_t top_k = static_cast<int32_t>(2 * num_beams);
    OrtValue *topk_input;
    api.CreateTensorWithDataAsOrtValue(ortmemoryinfo, next_token_scores.data(), size_t(4) * batch_size * num_beams * vocab_size, softmax_input_dims.data(),
                                       softmax_input_dims.size(), ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &topk_input);

    OrtValue *topk;
    std::vector<int64_t> topk_input_dims{1};
    std::vector<int64_t> topk_input_data{top_k};
    api.CreateTensorWithDataAsOrtValue(ortmemoryinfo, topk_input_data.data(), size_t(8) * 1, topk_input_dims.data(),
                                       topk_input_dims.size(), ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &topk);
    const OrtValue *topk_inputs[2] = {reinterpret_cast<const OrtValue *>(topk_input), reinterpret_cast<const OrtValue *>(topk)};

    std::vector<int64_t> topk_dims{batch_size, top_k};
    OrtValue *topk_scores;
    std::vector<float> topk_scores_data(batch_size * top_k, -1);
    api.CreateTensorWithDataAsOrtValue(ortmemoryinfo, topk_scores_data.data(), size_t(4) * batch_size * top_k, topk_dims.data(),
                                       topk_dims.size(), ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &topk_scores);

    OrtValue *topk_indices;
    std::vector<int64_t> topk_indices_data(batch_size * top_k, -1);
    api.CreateTensorWithDataAsOrtValue(ortmemoryinfo, topk_indices_data.data(), size_t(8) * batch_size * top_k, topk_dims.data(),
                                       topk_dims.size(), ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &topk_indices);
    OrtValue *topk_outputs[2] = {topk_scores, topk_indices};

    OrtOp *topk_op = ops_map[std::string("topk")];
    ort.InvokeOp(context, reinterpret_cast<const OrtOp *>(topk_op), topk_inputs, 2, topk_outputs, 2);

#ifdef DEBUG_BEAM_SEARCH
    dumper->Print("topk_scores", topk_scores_data.data(), batch_size, top_k);
    dumper->Print("topk_indices", topk_indices_data.data(), batch_size, top_k);
#endif

    // Convert indices in range [0, num_beams * vocab_size) to token ID of range [0, vocab_size) like the following:
    //   next_indices = (next_tokens / vocab_size).long()
    //   next_tokens = next_tokens % vocab_size
    // gsl::span<const int64_t> next_token_indices = topk_indices->DataAsSpan<int64_t>();
    gsl::span<const int64_t> next_token_indices = gsl::make_span(topk_indices_data.data(), batch_size * top_k);

    offset = 0;
    for (int i = 0; i < batch_size; i++)
    {
      for (unsigned int j = 0; j < top_k; j++, offset++)
      {
        beam_state->next_indices[offset] = gsl::narrow_cast<int32_t>(next_token_indices[offset] / vocab_size);
        beam_state->next_tokens[offset] = gsl::narrow_cast<int32_t>(next_token_indices[offset] % vocab_size);
      }
    }

    // gsl::span<const T> next_scores = topk_scores->DataAsSpan<T>();
    gsl::span<const T> next_scores = gsl::make_span(topk_scores_data.data(), batch_size * top_k);
    gsl::span<const int32_t> next_tokens(beam_state->next_tokens.data(), beam_state->next_tokens.size());
    gsl::span<const int32_t> next_indices(beam_state->next_indices.data(), beam_state->next_indices.size());

#ifdef DEBUG_BEAM_SEARCH
    dumper->Print("next_scores before scorer", next_scores.data(), batch_size, top_k);
    dumper->Print("next_tokens before scorer", next_tokens.data(), batch_size, top_k);
    dumper->Print("next_indices before scorer", next_indices.data(), batch_size, top_k);
#endif

    beam_scorer->Process(
        sequences,
        next_scores,
        next_tokens,
        next_indices);

    return nullptr;
  }

template <typename T>
OrtStatusPtr GreedySearchProcessLogits(
    OrtKernelContext* context,
    OrtApi &api,
    Ort::CustomOpApi &ort,
    const OrtValue& logits,                                     // logits output of subgraph
    custombsop::IGreedySearchState<T>* greedy_state,            // state
    custombsop::ISequences* sequences,                          // sequences
    OrtAllocator* allocator,                                    // default allocator
    void* thread_pool,                                          // thread pool (for CPU only)
    custombsop::ILogitsProcessorList* logits_processors,        // logits processors
    const custombsop::IBeamSearchParameters* parameters,        // parameters
    int step,                                                   // iteration counter
    void* stream,                                               // cuda stream (for CUDA only)
    const custombsop::IConsoleDumper* dumper,                   // tensor dumper
    std::unordered_map<std::string, OrtOp*> &ops_map)
{
  int batch_size = parameters->batch_size;
  int vocab_size = parameters->vocab_size;

  const T* logits_data = ort.GetTensorData<T>(&logits);

  // Logits has shape (batch_size * num_beams, input_length, vocab_size),
  // where input_length equals to parameters_->sequence_length for first subgraph call, and 1 for the remaining calls.
  OrtTensorTypeAndShapeInfo* logits_data_info = ort.GetTensorTypeAndShape(&logits);
  std::vector<int64_t> logits_shape = ort.GetTensorShape(logits_data_info);

  CUSTOMOP_ENFORCE(logits_shape.size() == 3);
  auto input_length = logits_shape[1];
  auto logits_batch_size = logits_shape[0];

#ifdef DEBUG_BEAM_SEARCH
  std::cout<<"logits shape:"<<logits_shape[0]<<","<<logits_shape[1]<<","<<logits_shape[2]<<std::endl;
#endif

  // Get logits for the last token:
  //    next_token_logits = logits[:, -1, :], and the result shape is (batch_size * num_beams, vocab_size)
  // When input_length == 1, use logits directly in SoftmaxCPU below so it only need for input_length > 1.
  gsl::span<T>& next_token_scores  = greedy_state->next_token_scores ;
  const T* current_logits = logits_data + (input_length - 1) * vocab_size;
  for (int i = 0; i < batch_size; i++) {
    gsl::span<const T> source(current_logits, vocab_size);
    gsl::span<T> target = next_token_scores .subspan(SafeInt<gsl::index>(i) * vocab_size, static_cast<gsl::index>(vocab_size));
    gsl::copy(source, target);
    current_logits += input_length * vocab_size;
  }

#ifdef DEBUG_BEAM_SEARCH
  std::cout<<"Executing inside ProcessLogits"<<std::endl;
  dumper->Print("next_token_scores ", next_token_scores .data(), batch_size, num_beams, vocab_size);
#endif

  // Apply all score processors that updates scores
  logits_processors->Process(sequences, next_token_scores, step);

#ifdef DEBUG_BEAM_SEARCH
  dumper->Print("next_token_scores after logits processor", next_token_scores.data(), batch_size, num_beams, vocab_size);
#endif

   OrtMemoryInfo *ortmemoryinfo;
  // Must be freed explicitly
  api.CreateMemoryInfo("Cpu", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeCPU, &ortmemoryinfo);

  // argmax
  const int32_t top_k = static_cast<int32_t>(1);
  OrtValue* topk_input;
  std::vector<int64_t> softmax_input_dims{batch_size, vocab_size};
  api.CreateTensorWithDataAsOrtValue(ortmemoryinfo, next_token_scores.data(), size_t(4)*batch_size*vocab_size, softmax_input_dims.data(),
      softmax_input_dims.size(), ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &topk_input);

  OrtValue* topk;
  std::vector<int64_t> topk_input_dims{1};
  std::vector<int64_t> topk_input_data{top_k};
  api.CreateTensorWithDataAsOrtValue(ortmemoryinfo, topk_input_data.data(), size_t(8)*1, topk_input_dims.data(),
      topk_input_dims.size(), ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &topk);
  const OrtValue* topk_inputs[2] = {reinterpret_cast<const OrtValue*>(topk_input), reinterpret_cast<const OrtValue*>(topk)};

  std::vector<int64_t> topk_dims{batch_size, top_k};
  OrtValue* topk_scores;
  std::vector<float> topk_scores_data(batch_size*top_k, -1);
  api.CreateTensorWithDataAsOrtValue(ortmemoryinfo, topk_scores_data.data(), size_t(4)*batch_size*top_k, topk_dims.data(),
      topk_dims.size(), ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &topk_scores);

  OrtValue* topk_indices;
  std::vector<int64_t> topk_indices_data(batch_size*top_k, -1);
  api.CreateTensorWithDataAsOrtValue(ortmemoryinfo, topk_indices_data.data(), size_t(8)*batch_size*top_k, topk_dims.data(),
      topk_dims.size(), ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &topk_indices);
  OrtValue* topk_outputs[2] = {topk_scores, topk_indices};

  OrtOp* topk_op = ops_map[std::string("topk")];
  ort.InvokeOp(context, reinterpret_cast<const OrtOp*>(topk_op), topk_inputs, 2, topk_outputs, 2);

#ifdef DEBUG_BEAM_SEARCH
  dumper->Print("topk_scores", topk_scores_data.data(), batch_size, top_k);
  dumper->Print("topk_indices", topk_indices_data.data(), batch_size, top_k);
#endif

  gsl::span<const int64_t> next_token_indices = gsl::make_span(topk_indices_data.data(), batch_size * top_k);
  gsl::copy(next_token_indices, greedy_state->next_tokens_cpu);

#ifdef DEBUG_BEAM_SEARCH
  gsl::span<const int64_t> next_tokens(greedy_state->next_tokens_cpu.data(), greedy_state->next_tokens_cpu.size());
  dumper->Print("next_tokens before scorer", next_tokens.data(), batch_size, top_k);
#endif

  return nullptr;
}

  template <typename T>
  void InitBeamState(custombsop::IBeamSearchState<T> *beam_state,
                     custombsop::IBeamSearchCpuState *cpu_state,
                     gsl::span<int32_t> &sequence_lengths,
                     int batch_size,
                     int num_beams,
                     gsl::span<const int32_t> input_ids_in_cpu,
                     int sequence_length,
                     int max_length)
  {
    memset(beam_state->beam_scores.data(), 0, beam_state->beam_scores.size_bytes());
    memset(beam_state->next_token_logits.data(), 0, beam_state->next_token_logits.size_bytes());
    memset(beam_state->next_token_scores.data(), 0, beam_state->next_token_scores.size_bytes());
    memset(beam_state->next_tokens.data(), 0, beam_state->next_tokens.size_bytes());
    memset(beam_state->next_indices.data(), 0, beam_state->next_indices.size_bytes());
    memset(beam_state->next_positions.data(), 0, beam_state->next_positions.size_bytes());

    // Initialize score of first beam of each group with 0 and the rest with -1e9.
    // This ensures that the beams in the same group don't produce same tokens every time.
    gsl::span<float> &beam_scores = beam_state->beam_scores;
    for (int i = 0; i < batch_size; i++)
    {
      for (int j = 1; j < num_beams; j++)
      {
        beam_scores[SafeInt<gsl::index>(i) * num_beams + j] = -1e9;
      }
    }

    gsl::copy(sequence_lengths, beam_state->next_positions);

    int *position_data_ptr = beam_state->next_positions.data();
    for (int i = 0; i < batch_size * num_beams; i++)
    {
      position_data_ptr[i] -= 1;
    }

    memset(cpu_state->sequences_space.data(), 0, cpu_state->sequences_space.size_bytes());

    // Copy input_ids to sequences[0].
    gsl::span<int32_t> sequences_0 = cpu_state->sequences_space;
    int batch_beam_size = batch_size * num_beams;
    for (int i = 0; i < batch_beam_size; i++)
    {
      for (int j = 0; j < sequence_length; j++)
      {
        sequences_0[SafeInt<gsl::index>(i) * max_length + j] = static_cast<int32_t>(input_ids_in_cpu[SafeInt<gsl::index>(i) * sequence_length + j]);
      }
    }
  }

  template<typename T>
  void InitGreedyState(
      custombsop::IGreedySearchState<T>* greedy_state,
      gsl::span<int32_t>& sequence_lengths,
      int batch_size,
      int sequence_length,
      int max_length,
      gsl::span<const int32_t> input_ids_in_cpu,
      void* /*stream*/) {
    memset(greedy_state->next_token_scores.data(), 0, greedy_state->next_token_scores.size_bytes());
    memset(greedy_state->next_tokens.data(), 0, greedy_state->next_tokens.size_bytes());
    memset(greedy_state->next_positions.data(), 0, greedy_state->next_positions.size_bytes());

    gsl::copy(sequence_lengths, greedy_state->next_positions);

    int* position_data_ptr = greedy_state->next_positions.data();
    for (int i = 0; i < batch_size; i++) {
      position_data_ptr[i] -= 1;
    }

    memset(greedy_state->sequences_space.data(), 0, greedy_state->sequences_space.size_bytes());

    // Copy input_ids to sequences[0].
    gsl::span<int32_t> sequences_0 = greedy_state->sequences_space;
    for (int i = 0; i < batch_size; i++) {
      for (int j = 0; j < sequence_length; j++) {
        sequences_0[SafeInt<gsl::index>(i) * max_length + j] = static_cast<int32_t>(input_ids_in_cpu[SafeInt<gsl::index>(i) * sequence_length + j]);
      }
    }
  }

  // Explicit template instantiations of functions
  template void InitBeamState<float>(
      custombsop::IBeamSearchState<float> *beam_state,
      custombsop::IBeamSearchCpuState *cpu_state,
      gsl::span<int32_t> &sequence_lengths,
      int batch_size,
      int num_beams,
      gsl::span<const int32_t> input_ids_in_cpu,
      int sequence_length,
      int max_length);

  template void InitGreedyState<float>(
    custombsop::IGreedySearchState<float>* greedy_state,
    gsl::span<int32_t>& sequence_lengths,
    int batch_size,
    int sequence_length,
    int max_length,
    gsl::span<const int32_t> input_ids_in_cpu,
    void* stream);

  template OrtStatusPtr ProcessLogits<float>(
      OrtKernelContext *context,
      OrtApi &api,
      Ort::CustomOpApi &ort,
      const OrtValue &logits,                              // logits output of subgraph
      custombsop::IBeamSearchState<float> *beam_state,     // state
      custombsop::IBeamSearchCpuState *cpu_state,          // state in CPU
      custombsop::ISequences *sequences,                   // sequences
      OrtAllocator *ort_allocator,                         // default allocator
      custombsop::ILogitsProcessorList *logits_processors, // logits processors
      custombsop::IBeamScorer *beam_scorer,                // beam scorer
      custombsop::IBeamSearchParameters *parameters,       // parameters
      int step,                                            // iteration counter
      const custombsop::IConsoleDumper *dumper,            // tensor dumper
      std::unordered_map<std::string, OrtOp *> &ops_map);

  template OrtStatusPtr GreedySearchProcessLogits<float>(
    OrtKernelContext* context,
    OrtApi &api,
    Ort::CustomOpApi &ort,
    const OrtValue& logits,                                     // logits output of subgraph
    custombsop::IGreedySearchState<float>* greedy_state,            // state
    custombsop::ISequences* sequences,                          // sequences
    OrtAllocator* allocator,                                    // default allocator
    void* thread_pool,                                          // thread pool (for CPU only)
    custombsop::ILogitsProcessorList* logits_processors,        // logits processors
    const custombsop::IBeamSearchParameters* parameters,        // parameters
    int step,                                                   // iteration counter
    void* stream,                                               // cuda stream (for CUDA only)
    const custombsop::IConsoleDumper* dumper,                   // tensor dumper
    std::unordered_map<std::string, OrtOp*> &ops_map);

  template OrtStatusPtr UpdateFeeds<float>(
      OrtApi &api,
      Ort::CustomOpApi &ort,
      OrtMemoryInfo *ortmemoryinfo,
      OrtAllocator *ort_allocator,
      void *stream,
      std::vector<OrtValue *> &last_outputs,
      std::vector<OrtValue *> &next_inputs,
      int current_length,
      OrtValue *position_ids,
      gsl::span<const int32_t> beam_next_tokens,
      gsl::span<const int32_t> beam_indices,
      int num_beams,
      int gpt_subgraph_first_past_input_idx,
      int gpt_subgraph_first_present_output_idx,
      const custombsop::IConsoleDumper *dumper);

  OrtStatusPtr CreateInputs(OrtApi &api,
                            Ort::CustomOpApi &ort,
                            const OrtValue *original_input_ids,
                            int num_beams,
                            int pad_token_id,
                            gsl::span<int32_t> &sequence_lengths,
                            OrtAllocator *ort_allocator,
                            OrtValue **expanded_input_ids,
                            OrtValue **expanded_position_ids,
                            OrtValue **expanded_attention_mask)
  {

    const OrtTensorTypeAndShapeInfo *original_input_ids_info = ort.GetTensorTypeAndShape(original_input_ids);
    std::vector<int64_t> original_input_ids_shape = ort.GetTensorShape(original_input_ids_info);

    CUSTOMOP_ENFORCE(original_input_ids_shape.size() == 2)

    const int64_t &batch_size = original_input_ids_shape[0];
    const int64_t &sequence_length = original_input_ids_shape[1];

    OrtValue *input_ids;
    api.CreateTensorAsOrtValue(ort_allocator, original_input_ids_shape.data(), original_input_ids_shape.size(),
                               ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &input_ids);

    OrtValue *position_ids;
    api.CreateTensorAsOrtValue(ort_allocator, original_input_ids_shape.data(), original_input_ids_shape.size(),
                               ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &position_ids);

    OrtValue *attention_mask;
    api.CreateTensorAsOrtValue(ort_allocator, original_input_ids_shape.data(), original_input_ids_shape.size(),
                               ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &attention_mask);

    // Set attention mask to be 0 for pad tokens, and 1 for all other tokens.
    // Set position id to be 0 for pad tokens, and accumulated sum of mask in a batch for other tokens
    int32_t *ids = ort.GetTensorMutableData<int32_t>(input_ids);
    const int32_t *orig_ids = ort.GetTensorData<int32_t>(original_input_ids);
    int32_t *mask_data = ort.GetTensorMutableData<int32_t>(attention_mask);
    int32_t *position_data = ort.GetTensorMutableData<int32_t>(position_ids);
    const int32_t *word_id = ort.GetTensorData<int32_t>(original_input_ids);
    int32_t *mask = mask_data;
    int32_t *position = position_data;

    for (int i = 0; i < batch_size; i++)
    {
      int32_t abs_position = 0;
      for (int j = 0; j < sequence_length; j++, word_id++, mask++, position++)
      {
        if (*word_id == pad_token_id)
        {
          *mask = 0;
          *position = 0;
        }
        else
        {
          *mask = 1;
          *position = abs_position;
          abs_position++;
        }
        *ids = *orig_ids;
        ids++;
        orig_ids++;
      }

      for (int k = 0; k < num_beams; k++)
      {
        // TODO use safeint here
        // sequence_lengths[SafeInt<gsl::index>(i) * num_beams + k] = abs_position;
        sequence_lengths[SafeInt<size_t>(i) * num_beams + k] = abs_position;
      }
    }

    *expanded_input_ids = ExpandInputs(api, ort, input_ids, num_beams, ort_allocator, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    *expanded_position_ids = ExpandInputs(api, ort, position_ids, num_beams, ort_allocator, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    *expanded_attention_mask = ExpandInputs(api, ort, attention_mask, num_beams, ort_allocator, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);

    return nullptr;
  }
}