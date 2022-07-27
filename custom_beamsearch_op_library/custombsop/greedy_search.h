// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "greedy_search_parameters.h"
#include "beam_search_shared.h"
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

using namespace std;

//#define PRINT_TO_CONSOLE
namespace custombsop {
    OrtStatusPtr RunGreedySearchOnInternalSession(
                            OrtKernelContext* context,
                            OrtApi &api,
                            Ort::CustomOpApi &ort,
                            OrtSession *session,
                            custombsop::GreedySearchParameters parameters,
                            std::unordered_map<std::string, OrtOp*> &ops_map);
}