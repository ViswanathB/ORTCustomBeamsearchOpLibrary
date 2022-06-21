// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
using namespace std;

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "beam_search_parameters.h"

//#define PRINT_TO_CONSOLE
void RunBeamSearchOnInternalSession(OrtKernelContext* context, OrtApi &api, Ort::CustomOpApi &ort, OrtSession *session, custombsop::BeamSearchParameters parameters);