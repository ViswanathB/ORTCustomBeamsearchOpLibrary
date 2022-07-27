#pragma once
#include <functional>
#include "beam_search_shared.h"

using namespace std;

namespace BeamSearchDeviceHelper
{
    // Create subgraph inputs: input_ids, position_ids and attention_mask
    using CreateInputsFunc = std::function<OrtStatusPtr(
        const OrtApi *api,
        Ort::CustomOpApi &ort,
        const OrtValue *original_input_ids,
        int num_beams,
        int pad_token_id,
        gsl::span<int32_t> &sequence_lengths,
        OrtAllocator *ort_allocator,
        OrtValue **expanded_input_ids,
        OrtValue **expanded_position_ids,
        OrtValue **expanded_attention_mask)>;

    template <typename T>
    using ProcessLogitsFunc = std::function<OrtStatusPtr(
        OrtKernelContext *context,
        const OrtApi *api,
        Ort::CustomOpApi &ort,
        const OrtValue &logits,                              // logits output of subgraph
        custombsop::IBeamSearchState<T> *beam_state,         // state
        custombsop::IBeamSearchCpuState *cpu_state,          // state in CPU
        custombsop::ISequences *sequences,                   // sequences
        OrtAllocator *ort_allocator,                         // default allocator
        custombsop::ILogitsProcessorList *logits_processors, // logits processors
        custombsop::IBeamScorer *beam_scorer,                // beam scorer
        custombsop::IBeamSearchParameters *parameters,       // parameters
        int step,                                            // iteration counter
        const custombsop::IConsoleDumper *dumper,            // tensor dumper
        std::unordered_map<std::string, OrtOp *> &ops_map)>;

    template <typename T>
    using UpdateFeedsFunc = std::function<OrtStatusPtr(
        const OrtApi *api,
        Ort::CustomOpApi &ort,
        OrtMemoryInfo *ortmemoryinfo,
        OrtAllocator *ort_allocator,
        void *stream,
        std::vector<OrtValue *> &last_outputs,
        std::vector<OrtValue *> &next_inputs,
        int current_length,
        OrtValue *position_ids,
        gsl::span<const int32_t> beam_next_tokens,
        gsl::span<const int32_t> beam_indices,
        int num_beams,
        int gpt_subgraph_first_past_input_idx,
        int gpt_subgraph_first_present_output_idx,
        const custombsop::IConsoleDumper *dumper)>;

    using AddToFeedsFunc = std::function<OrtStatusPtr(
        OrtValue *input_ids,
        OrtValue *position_ids,
        OrtValue *attention_mask,
        std::vector<OrtValue *> &feeds)>;

    template <typename T>
    using InitBeamStateFunc = std::function<void(
        custombsop::IBeamSearchState<T> *beam_state,
        custombsop::IBeamSearchCpuState *cpu_state,
        gsl::span<int32_t> &sequence_lengths,
        int batch_size,
        int num_beams,
        gsl::span<const int32_t> input_ids_in_cpu,
        int sequence_length,
        int max_length)>;
}

// These are CPU specific device helper implementations
namespace BeamSearchCpuDeviceHelper
{
    OrtStatusPtr CreateInputs(
        const OrtApi *api,
        Ort::CustomOpApi &ort,
        const OrtValue *original_input_ids,
        int num_beams,
        int pad_token_id,
        gsl::span<int32_t> &sequence_lengths,
        OrtAllocator *ort_allocator,
        OrtValue **expanded_input_ids,
        OrtValue **expanded_position_ids,
        OrtValue **expanded_attention_mask);

    template <typename T>
    OrtStatusPtr ProcessLogits(
        OrtKernelContext *context,
        const OrtApi *api,
        Ort::CustomOpApi &ort,
        const OrtValue &logits,                              // logits output of subgraph
        custombsop::IBeamSearchState<T> *beam_state,         // state
        custombsop::IBeamSearchCpuState *cpu_state,          // state in CPU
        custombsop::ISequences *sequences,                   // sequences
        OrtAllocator *ort_allocator,                         // default allocator
        custombsop::ILogitsProcessorList *logits_processors, // logits processors
        custombsop::IBeamScorer *beam_scorer,                // beam scorer
        custombsop::IBeamSearchParameters *parameters,       // parameters
        int step,                                            // iteration counter
        const custombsop::IConsoleDumper *dumper,            // tensor dumper
        std::unordered_map<std::string, OrtOp *> &ops_map);

    OrtStatusPtr AddToFeeds(
        OrtValue *input_ids,
        OrtValue *position_ids,
        OrtValue *attention_mask,
        std::vector<OrtValue *> &feeds);

    template <typename T>
    OrtStatusPtr UpdateFeeds(
        const OrtApi *api,
        Ort::CustomOpApi &ort,
        OrtMemoryInfo *ortmemoryinfo,
        OrtAllocator *ort_allocator,
        void *stream,
        std::vector<OrtValue *> &last_outputs,
        std::vector<OrtValue *> &next_inputs,
        int current_length,
        OrtValue *position_ids,
        gsl::span<const int32_t> beam_next_tokens,
        gsl::span<const int32_t> beam_indices,
        int num_beams,
        int gpt_subgraph_first_past_input_idx,
        int gpt_subgraph_first_present_output_idx,
        const custombsop::IConsoleDumper *dumper);

    template <typename T>
    void InitBeamState(custombsop::IBeamSearchState<T> *beam_state,
                       custombsop::IBeamSearchCpuState *cpu_state,
                       gsl::span<int32_t> &sequence_lengths,
                       int batch_size,
                       int num_beams,
                       gsl::span<const int32_t> input_ids_in_cpu,
                       int sequence_length,
                       int max_length);
}