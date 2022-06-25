#include "beam_search_device_helper.h"
#include "safeint.h"
#include "gsl/span"
#include "gsl/gsl"

using namespace std;
using namespace msl::utilities;

namespace BeamSearchCpuDeviceHelper {

OrtStatusPtr AddToFeeds(OrtValue* input_ids,
                        OrtValue* position_ids,
                        OrtValue* attention_mask,
                        std::vector<OrtValue*>& feeds) {
  feeds.push_back(std::move(input_ids));
  feeds.push_back(std::move(position_ids));
  feeds.push_back(std::move(attention_mask));
  return nullptr;
}

OrtValue* ExpandInputs(OrtApi &api, Ort::CustomOpApi &ort, OrtValue* input, int num_beams, OrtAllocator *allocator, ONNXTensorElementDataType element_type) {
  // Input shape (batch_size, sequence_length)
  // Output shape (batch_size * num_beams, sequence_length)
  if (num_beams == 1)
    return input;

  const OrtTensorTypeAndShapeInfo* input_info = ort.GetTensorTypeAndShape(input);
  std::vector<int64_t> input_shape = ort.GetTensorShape(input_info); 

  const int64_t& batch_size = input_shape[0];
  const int64_t& sequence_length = input_shape[1];

  vector<int64_t> dims{batch_size * num_beams, sequence_length};

  CUSTOMOP_ENFORCE(element_type == ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, \
            "input_ids, position_ids and attention_mask is required to be int32 data type");

  OrtValue *expanded;
  api.CreateTensorAsOrtValue(allocator, dims.data(), dims.size(), element_type, &expanded);

  const int32_t* input_data = ort.GetTensorData<int32_t>(input);
  int32_t* expanded_data = ort.GetTensorMutableData<int32_t>(expanded);
  int32_t* target = expanded_data;
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < num_beams; j++) {
      memcpy(target, input_data + i * sequence_length, sizeof(int32_t) * sequence_length);
      target += sequence_length;
    }
  }

  return expanded;
}

template<typename T>
void InitBeamState(custombsop::IBeamSearchState<T>* beam_state,
                   custombsop::IBeamSearchCpuState* cpu_state,
                   gsl::span<int32_t>& sequence_lengths,
                   int batch_size,
                   int num_beams,
                   gsl::span<const int32_t> input_ids_in_cpu,
                   int sequence_length,
                   int max_length,
                   void* /*stream*/) {
  memset(beam_state->beam_scores.data(), 0, beam_state->beam_scores.size_bytes());
  memset(beam_state->next_token_logits.data(), 0, beam_state->next_token_logits.size_bytes());
  memset(beam_state->next_token_scores.data(), 0, beam_state->next_token_scores.size_bytes());
  memset(beam_state->next_tokens.data(), 0, beam_state->next_tokens.size_bytes());
  memset(beam_state->next_indices.data(), 0, beam_state->next_indices.size_bytes());
  memset(beam_state->next_positions.data(), 0, beam_state->next_positions.size_bytes());

  // Initialize score of first beam of each group with 0 and the rest with -1e9.
  // This ensures that the beams in the same group don't produce same tokens every time.
  gsl::span<float>& beam_scores = beam_state->beam_scores;
  for (int i = 0; i < batch_size; i++) {
    for (int j = 1; j < num_beams; j++) {
      beam_scores[SafeInt<gsl::index>(i) * num_beams + j] = -1e9;
    }
  }

  gsl::copy(sequence_lengths, beam_state->next_positions);

  memset(cpu_state->sequences_space.data(), 0, cpu_state->sequences_space.size_bytes());

  // Copy input_ids to sequences[0].
  gsl::span<int32_t> sequences_0 = cpu_state->sequences_space;
  int batch_beam_size = batch_size * num_beams;
  for (int i = 0; i < batch_beam_size; i++) {
    for (int j = 0; j < sequence_length; j++) {
      sequences_0[SafeInt<gsl::index>(i) * max_length + j] = static_cast<int32_t>(input_ids_in_cpu[SafeInt<gsl::index>(i) * sequence_length + j]);
    }
  }
}

// Explicit template instantiations of functions
template void InitBeamState<float>(
    custombsop::IBeamSearchState<float>* beam_state,
    custombsop::IBeamSearchCpuState* cpu_state,
    gsl::span<int32_t>& sequence_lengths,
    int batch_size,
    int num_beams,
    gsl::span<const int32_t> input_ids_in_cpu,
    int sequence_length,
    int max_length,
    void* stream);

OrtStatusPtr CreateInputs(OrtApi &api,
                          Ort::CustomOpApi &ort,
                          const OrtValue* original_input_ids,
                          int num_beams,
                          int pad_token_id,
                          gsl::span<int32_t>& sequence_lengths,
                          OrtAllocator *ort_allocator,
                          OrtValue** expanded_input_ids,
                          OrtValue** expanded_position_ids,
                          OrtValue** expanded_attention_mask) {
  
  const OrtTensorTypeAndShapeInfo* original_input_ids_info = ort.GetTensorTypeAndShape(original_input_ids);
  std::vector<int64_t> original_input_ids_shape = ort.GetTensorShape(original_input_ids_info); 

  CUSTOMOP_ENFORCE(original_input_ids_shape.size() == 2)

  const int64_t& batch_size = original_input_ids_shape[0];
  const int64_t& sequence_length = original_input_ids_shape[1];

  OrtValue *input_ids;
  api.CreateTensorAsOrtValue(ort_allocator, original_input_ids_shape.data(), original_input_ids_shape.size(), \
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &input_ids);

  OrtValue *position_ids;
  api.CreateTensorAsOrtValue(ort_allocator, original_input_ids_shape.data(), original_input_ids_shape.size(), \
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &position_ids);

  OrtValue *attention_mask;
  api.CreateTensorAsOrtValue(ort_allocator, original_input_ids_shape.data(), original_input_ids_shape.size(), \
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &attention_mask);

  // Set attention mask to be 0 for pad tokens, and 1 for all other tokens.
  // Set position id to be 0 for pad tokens, and accumulated sum of mask in a batch for other tokens
  int32_t* ids =  ort.GetTensorMutableData<int32_t>(input_ids);
  const int32_t* orig_ids = ort.GetTensorData<int32_t>(original_input_ids);
  int32_t* mask_data = ort.GetTensorMutableData<int32_t>(attention_mask);
  int32_t* position_data = ort.GetTensorMutableData<int32_t>(position_ids);
  const int32_t* word_id = ort.GetTensorData<int32_t>(original_input_ids);
  int32_t* mask = mask_data;
  int32_t* position = position_data;

  for (int i = 0; i < batch_size; i++) {
    int32_t abs_position = 0;
    for (int j = 0; j < sequence_length; j++, word_id++, mask++, position++) {
      if (*word_id == pad_token_id) {
        *mask = 0;
        *position = 0;
      } else {
        *mask = 1;
        *position = abs_position;
        abs_position++;
      }
      *ids = *orig_ids;
      ids++;
      orig_ids++;
    }

    for (int k = 0; k < num_beams; k++) {
      //TODO use safeint here
      //sequence_lengths[SafeInt<gsl::index>(i) * num_beams + k] = abs_position;
      sequence_lengths[i * num_beams + k] = abs_position;
    }
  }

  // Expand (batch_size, sequence_length) to (batch_size * num_beams, sequence_length) for input_ids, position_ids and attention_mask
  // TODO: Try expand outputs after first subgraph call instead. That may get better performance, but more complex to implement.
  *expanded_input_ids = ExpandInputs(api, ort, input_ids, num_beams, ort_allocator, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
  *expanded_position_ids = ExpandInputs(api, ort, position_ids, num_beams, ort_allocator, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
  *expanded_attention_mask = ExpandInputs(api, ort, attention_mask, num_beams, ort_allocator, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);

  return nullptr;
  //return api.CreateStatus(OrtErrorCode::ORT_OK, "success");
}

}