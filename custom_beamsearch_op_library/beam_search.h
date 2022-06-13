// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using namespace std;

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

//#define PRINT_TO_CONSOLE 1

void RunBeamSearchOnInternalSession(OrtKernelContext* context, OrtApi &api, Ort::CustomOpApi &ort, OrtSession *session);