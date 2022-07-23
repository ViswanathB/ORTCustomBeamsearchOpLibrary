// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "beam_search_shared.h"
#include "beam_search_parameters.h"

namespace custombsop {

struct GreedySearchParameters : public BeamSearchParameters {
  int BatchBeamSize() const { return batch_size; }

  void ParseFromAttributes(Ort::CustomOpApi &ort, const OrtKernelInfo* info);

  void ParseFromInputs(OrtKernelContext* context, Ort::CustomOpApi &ort);
};
}  // namespace custombsop
