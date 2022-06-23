#pragma once
#include "gsl/gsl"

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "utils.h"

namespace custombsop {

template <typename T>
struct IBeamSearchState {
  gsl::span<T> next_token_logits;      // shape (batch_size * num_beams, vocab_size)
  gsl::span<float> next_token_scores;  // shape (batch_size, num_beams * vocab_size)
  gsl::span<int32_t> next_tokens;      // shape (batch_size, 2 * num_beams)
  gsl::span<int32_t> next_indices;     // shape (batch_size, 2 * num_beams)
  gsl::span<int32_t> next_positions;   // shape (batch_size, num_beams). Next position value for position_ids.
  gsl::span<float> beam_scores;        // shape (batch_size, num_beams)
  gsl::span<float> scores;             // shape (max_length - sequence_length + 1, batch_size, num_beams * vocab_size)
  gsl::span<float> remaining_scores;   // portion of scores that is avaiable for appending next token scores.
};

struct IBeamSearchCpuState {
  gsl::span<int32_t> sequence_lengths;  // shape (batch_size, num_beams), initial sequence length
  gsl::span<int32_t> sequences_space;   // shape (2, batch_size, num_beams, max_seq_length)

  // The following are used only by CUDA operator for data copied from device.
  gsl::span<float> topk_scores;        // shape (batch_size, 2*num_beams), scores of topk candidates (K=2*num_beams).
  gsl::span<int32_t> topk_tokens;      // shape (batch_size, 2*num_beams), tokens of topk candidates.
  gsl::span<int32_t> topk_indices;     // shape (batch_size, 2*num_beams), beam indices of topk candidates.
  gsl::span<float> final_beam_scores;  // shape (batch_size, num_beams)
};

class ISequences {
 public:
  virtual ~ISequences() {}
  virtual gsl::span<const int32_t> GetSequence(int beam_index) const = 0;
  virtual int GetSequenceLength() const = 0;
};

/*
class ILogitsProcessorList {
 public:
  virtual ~ILogitsProcessorList() {}
  virtual void Process(const ISequences* sequences, gsl::span<float>& next_token_scores, int step) = 0;
};
*/

// Interface for all scorers for beam search or beam sample.
class IBeamScorer {
 public:
  virtual ~IBeamScorer() {}

  virtual void Initialize(OrtAllocator* allocator, int sequence_length) = 0;

  virtual void Process(ISequences* sequences,
                       gsl::span<const float>& next_scores,
                       gsl::span<const int32_t>& next_tokens,
                       gsl::span<const int32_t>& next_indices) = 0;

  virtual void Finalize(ISequences* sequences,
                        gsl::span<const float>& final_beam_scores,
                        OrtValue* output_sequences,
                        OrtValue* output_sequence_scores) = 0;
};

struct IBeamSearchParameters {
  // Parameters from node attributes
  int model_type;
  int eos_token_id;
  int pad_token_id;
  int no_repeat_ngram_size;
  bool early_stopping;

  // Parameters from inputs
  int min_length;
  int max_length;
  int num_beams;
  int num_return_sequences;
  float temperature;
  float length_penalty;
  float repetition_penalty;
  int batch_size;       // deduce from first dimension of input_ids
  int sequence_length;  // deduce from second dimension of input_ids

  gsl::span<const int32_t> vocab_mask;
  gsl::span<const int32_t> prefix_vocab_mask;

  // Parameters from outputs.
  bool output_scores;  // whether scores existed in output

  // Parameters from subgraph.
  int vocab_size;
  int num_heads;
  int head_size;
  int num_layers;
};

}  // namespace custombsop