// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "beam_search_parameters.h"

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

namespace custombsop
{

  constexpr int kMaxSequenceLength = 4096;
  constexpr int kMaxNumBeams = 128;

  void BeamSearchParameters::Validate() const
  {
    CUSTOMOP_ENFORCE(eos_token_id > 0, "invalid eos_token_id")
    CUSTOMOP_ENFORCE(pad_token_id > 0, "invalid pad_token_id")
    CUSTOMOP_ENFORCE(min_length < max_length, "min_length must be less than max length");
  }

  void BeamSearchParameters::ParseFromAttributes(Ort::CustomOpApi &ort, const OrtKernelInfo *info)
  {
    early_stopping = static_cast<bool>(ort.KernelInfoGetAttribute<int64_t>(info, "early_stopping"));
    eos_token_id = static_cast<int>(ort.KernelInfoGetAttribute<int64_t>(info, "eos_token_id"));
    pad_token_id = static_cast<int>(ort.KernelInfoGetAttribute<int64_t>(info, "pad_token_id"));
    no_repeat_ngram_size = static_cast<int>(ort.KernelInfoGetAttribute<int64_t>(info, "no_repeat_ngram_size"));
  }

  void BeamSearchParameters::ParseFromInputs(OrtKernelContext *context, Ort::CustomOpApi &ort)
  {
    CUSTOMOP_ENFORCE(context != nullptr, "context is null");

    const OrtValue *input_ids_tensor = ort.KernelContext_GetInput(context, 0);
    OrtTensorTypeAndShapeInfo *input_ids_info = ort.GetTensorTypeAndShape(input_ids_tensor);
    std::vector<int64_t> tensor_shape = ort.GetTensorShape(input_ids_info);

    CUSTOMOP_ENFORCE(tensor_shape.size() == 2, "input_ids shall have 2 dimensions. Got ", tensor_shape.size());
    batch_size = static_cast<int>(tensor_shape[0]);
    sequence_length = static_cast<int>(tensor_shape[1]);

    const OrtValue *max_length_tensor = ort.KernelContext_GetInput(context, 1);
    max_length = max_length_tensor ? static_cast<int>(*ort.GetTensorData<int>(max_length_tensor)) : 1;
    CUSTOMOP_ENFORCE(max_length > sequence_length, "max_length (", max_length, ") shall be greater than input sequence length (", sequence_length, ")");
    CUSTOMOP_ENFORCE(max_length <= kMaxSequenceLength, "max_length (", max_length, ") shall be no more than ", kMaxSequenceLength);
  }
} // namespace custombsop