#pragma once
#include "gsl/gsl"

namespace custombsop {

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