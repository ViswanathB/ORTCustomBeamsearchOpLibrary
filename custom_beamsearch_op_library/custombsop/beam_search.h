// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "beam_search_parameters.h"
#include "beam_search_shared.h"
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

using namespace std;

namespace custombsop
{
    OrtStatusPtr RunBeamSearchOnInternalSession(
        OrtKernelContext *context,
        const OrtApi *api,
        Ort::CustomOpApi &ort,
        OrtSession *session,
        custombsop::BeamSearchParameters parameters,
        std::unordered_map<std::string, OrtOp *> &ops_map);
}