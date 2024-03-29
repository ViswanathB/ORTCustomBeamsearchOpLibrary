// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "beam_search_shared.h"

namespace custombsop
{

  struct BeamSearchParameters : public IBeamSearchParameters
  {
    void BeamSearchParameters::Validate() const;

    int BatchBeamSize() const { return batch_size * num_beams; }

    void ParseFromAttributes(Ort::CustomOpApi &ort, const OrtKernelInfo *info);

    void ParseFromInputs(OrtKernelContext *context, Ort::CustomOpApi &ort);
  };
} // namespace custombsop
