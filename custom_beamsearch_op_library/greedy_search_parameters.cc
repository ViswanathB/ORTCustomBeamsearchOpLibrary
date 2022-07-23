// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "greedy_search_parameters.h"

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

namespace custombsop {

constexpr int kMaxSequenceLength = 4096;

void GreedySearchParameters::ParseFromAttributes(Ort::CustomOpApi &ort, const OrtKernelInfo* info) {
  // TODO This is for GPT2 only now
  // model_type = static_cast<int>(info.GetAttrOrDefault<int64_t>("model_type", 0));

  early_stopping = static_cast<bool>(ort.KernelInfoGetAttribute<int64_t>(info, "early_stopping"));
  eos_token_id = static_cast<int>(ort.KernelInfoGetAttribute<int64_t>(info, "eos_token_id"));
  pad_token_id = static_cast<int>(ort.KernelInfoGetAttribute<int64_t>(info, "pad_token_id"));
  no_repeat_ngram_size = static_cast<int>(0);
}

void GreedySearchParameters::ParseFromInputs(OrtKernelContext* context, Ort::CustomOpApi &ort) {
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

  const OrtValue* repetition_penalty_tensor = ort.KernelContext_GetInput(context, 3);
  repetition_penalty = repetition_penalty_tensor ? static_cast<float>(*ort.GetTensorData<int>(repetition_penalty_tensor)) : 1.0f;
  CUSTOMOP_ENFORCE(repetition_penalty > 0.0f, "repetition_penalty shall be greater than 0, got ", repetition_penalty);

  num_beams = 1;
}

}  // namespace custombsop