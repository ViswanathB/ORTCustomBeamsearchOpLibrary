// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "beam_search_parameters.h"

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

namespace custombsop {

constexpr int kMaxSequenceLength = 4096;
constexpr int kMaxNumBeams = 128;

OrtStatusPtr BeamSearchParameters::Validate(OrtApi &api) const {
  CUSTOMOP_ENFORCE(eos_token_id > 0, "invalid eos_token_id")
  CUSTOMOP_ENFORCE(pad_token_id > 0, "invalid pad_token_id")
  CUSTOMOP_ENFORCE(min_length < max_length, "min_length must be less than max length")

  // TODO there is no point in returning statuses since at any point inside custom kernel
  // failure, we don't want a dependency in ORT to fall back.
  return nullptr;
  //return api.CreateStatus(OrtErrorCode::ORT_OK, "OK");
}

void BeamSearchParameters::ParseFromAttributes(Ort::CustomOpApi &ort, const OrtKernelInfo* info) {
  // TODO This is for GPT2 only now
  // model_type = static_cast<int>(info.GetAttrOrDefault<int64_t>("model_type", 0));

  early_stopping = static_cast<bool>(ort.KernelInfoGetAttribute<int64_t>(info, "early_stopping"));  
  eos_token_id = static_cast<int>(ort.KernelInfoGetAttribute<int64_t>(info, "eos_token_id"));
  pad_token_id = static_cast<int>(ort.KernelInfoGetAttribute<int64_t>(info, "pad_token_id"));
  no_repeat_ngram_size = static_cast<int>(ort.KernelInfoGetAttribute<int64_t>(info, "no_repeat_ngram_size"));
}

void BeamSearchParameters::ParseFromInputs(OrtKernelContext* context, Ort::CustomOpApi &ort) {
  CUSTOMOP_ENFORCE(context != nullptr, "context is null");

  const OrtValue* input_ids_tensor = ort.KernelContext_GetInput(context, 0);
  OrtTensorTypeAndShapeInfo* input_ids_info = ort.GetTensorTypeAndShape(input_ids_tensor);
  std::vector<int64_t> tensor_shape = ort.GetTensorShape(input_ids_info);

  CUSTOMOP_ENFORCE(tensor_shape.size() == 2, "input_ids shall have 2 dimensions. Got ", tensor_shape.size());
  batch_size = static_cast<int>(tensor_shape[0]);
  sequence_length = static_cast<int>(tensor_shape[1]);

  const OrtValue* max_length_tensor = ort.KernelContext_GetInput(context, 1);
  max_length = max_length_tensor ? static_cast<int>(*ort.GetTensorData<int>(max_length_tensor)) : kMaxSequenceLength;
  CUSTOMOP_ENFORCE(max_length > sequence_length, "max_length (", max_length, ") shall be greater than input sequence length (", sequence_length, ")");
  CUSTOMOP_ENFORCE(max_length <= kMaxSequenceLength, "max_length (", max_length, ") shall be no more than ", kMaxSequenceLength);

  const OrtValue* min_length_tensor = ort.KernelContext_GetInput(context, 2);  
  min_length = min_length_tensor ? static_cast<int>(*ort.GetTensorData<int>(min_length_tensor)) : 0;

  const OrtValue* num_beams_tensor = ort.KernelContext_GetInput(context, 3);
  num_beams = num_beams_tensor ? static_cast<int>(*ort.GetTensorData<int>(num_beams_tensor)) : 1;
  CUSTOMOP_ENFORCE(num_beams >= 1 && num_beams <= kMaxNumBeams, "num_beams shall be a positive integer no more than ", kMaxNumBeams, ", got ", num_beams);

  const OrtValue* num_return_sequences_tensor = ort.KernelContext_GetInput(context, 4);
  num_return_sequences = num_return_sequences_tensor ? static_cast<int>(*ort.GetTensorData<int>(num_return_sequences_tensor)) : 1;
  CUSTOMOP_ENFORCE(num_return_sequences >= 1, "num_return_sequences shall be a positive integer, got ", num_return_sequences);
  CUSTOMOP_ENFORCE(num_beams >= num_return_sequences, "num_return_sequences (", num_return_sequences, ") shall be be no more than num_beams (", num_beams, ")");

  const OrtValue* temperature_tensor = ort.KernelContext_GetInput(context, 5);
  temperature = temperature_tensor ? static_cast<float>(*ort.GetTensorData<int>(temperature_tensor)) : 1.0f;
  CUSTOMOP_ENFORCE(temperature > 0.0f, "temperature shall be greater than 0, got ", temperature);

  const OrtValue* length_penalty_tensor = ort.KernelContext_GetInput(context, 6);
  length_penalty = length_penalty_tensor ? static_cast<float>(*ort.GetTensorData<int>(length_penalty_tensor)) : 1.0f;

  const OrtValue* repetition_penalty_tensor = ort.KernelContext_GetInput(context, 6);
  repetition_penalty = repetition_penalty_tensor ? static_cast<float>(*ort.GetTensorData<int>(repetition_penalty_tensor)) : 1.0f;
  CUSTOMOP_ENFORCE(repetition_penalty > 0.0f, "repetition_penalty shall be greater than 0, got ", repetition_penalty);

  temperature = 1.0f;
  length_penalty = 1.0f;
  repetition_penalty = 1.0f;
}

/* TODO remove these parts, subgraph doesn't exist anymore
void BeamSearchParameters::SetSubgraphParameters(int vocabulary_size, int heads, int hidden_size_per_head, int layers) {
  vocab_size = vocabulary_size;
  num_heads = heads;
  head_size = hidden_size_per_head;
  num_layers = layers;
}
*/
}  // namespace custombsop