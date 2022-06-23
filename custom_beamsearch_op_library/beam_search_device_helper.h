#pragma once
#include "beam_search_shared.h"
#include <functional>

// TODO Use cxx api instead of calling c api's directly. 
using namespace std;

namespace BeamSearchDeviceHelper {
using AddToFeedsFunc = std::function<OrtStatusPtr(
    OrtValue* input_ids,
    OrtValue* position_ids,
    OrtValue* attention_mask,
    std::vector<OrtValue*>& feeds)>;

// Create subgraph inputs: input_ids, position_ids and attention_mask
using CreateInputsFunc = std::function<OrtStatusPtr(
    OrtApi &api,
    Ort::CustomOpApi &ort,
    const OrtValue* original_input_ids,
    int num_beams,
    int pad_token_id,
    gsl::span<int32_t>& sequence_lengths,
    OrtAllocator *ort_allocator,
    OrtValue** expanded_input_ids,
    OrtValue** expanded_position_ids,
    OrtValue** expanded_attention_mask)>;
}

// These are CPU specific device helper implementations
namespace BeamSearchCpuDeviceHelper {
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
    OrtValue** expanded_attention_mask);

OrtStatusPtr AddToFeeds(
    OrtValue* input_ids,
    OrtValue* position_ids,
    OrtValue* attention_mask,
    std::vector<OrtValue*>& feeds);
} // namespace BeamSearchCpuDeviceHelper