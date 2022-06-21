#include "beam_search_device_helper.h"
#include "safeint.h"
#include "gsl/gsl"

namespace BeamSearchCpuDeviceHelper {

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

OrtStatusPtr CreateInputs(
    OrtApi &api,
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

  return api.CreateStatus(OrtErrorCode::ORT_OK, "success");
}

}