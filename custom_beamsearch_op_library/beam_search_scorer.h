#pragma once
#include <queue>

#include "utils.h"
#include "beam_search_shared.h"

using namespace std;

namespace custombsop {

struct HypothesisScore {
  HypothesisScore(gsl::span<const int32_t>& _hypothesis, float _score)
      : hypothesis(_hypothesis), score(_score) {}

  gsl::span<const int32_t> hypothesis;
  float score;
};

class HypothesisScoreCompare {
 public:
  bool operator()(const HypothesisScore& a, const HypothesisScore& b) {
    return a.score > b.score;
  }
};

class BeamHypotheses {
 public:
  BeamHypotheses(int num_beams,
                 float length_penalty,
                 bool early_stopping);
                 //onnxruntime::OrtStlAllocator<HypothesisScore>& hypothesis_score_allocator);

  // Number of hypotheses
  int Size() { return static_cast<int>(beams_.size()); }

  // Add a new hypothesis
  void Add(gsl::span<const int32_t>& hypothesis, float sum_logprobs);

  bool IsDone(float best_sum_logprobs, int current_length);

  // Output results. Note that it will clear all beams.
  void Output(int top_k,                            // number of sequences to return
              int max_length,                       // max sequence length
              gsl::span<int32_t>& sequences,        // buffer filled with pad token ID, with shape (num_return_sequences, max_length)
              gsl::span<float>& sequences_scores);  // buffer for sequence scores, with shape (num_return_sequences)

 private:
  int num_beams_;
  float length_penalty_;
  bool early_stopping_;
  float worst_score_;
  std::priority_queue<HypothesisScore, std::vector<HypothesisScore>, HypothesisScoreCompare> beams_;  // min-heap for top k
};

class BeamSearchScorer : public IBeamScorer {
 public:
  BeamSearchScorer(size_t batch_size,
                   size_t num_beams,
                   size_t max_length,
                   float length_penalty,
                   bool early_stopping,
                   size_t num_return_sequences,
                   int pad_token_id,
                   int eos_token_id);
                   //onnxruntime::OrtStlAllocator<HypothesisScore>& hypothesis_score_allocator,
                   //onnxruntime::OrtStlAllocator<BeamHypotheses>& beam_hyps_allocator);

  void Initialize(OrtAllocator* ort_allocator, int sequence_length) override;

  void Process(ISequences* sequences,
               gsl::span<const float>& next_scores,
               gsl::span<const int32_t>& next_tokens,
               gsl::span<const int32_t>& next_indices) override;

  void Finalize(OrtApi &api,
                Ort::CustomOpApi &ort,
                ISequences* sequences,
                gsl::span<const float>& final_beam_scores,
                OrtValue* output_sequences,
                OrtValue* output_sequence_scores) override;

  bool IsDone();

  gsl::span<float>& GetNextScores() { return next_beam_scores_; }
  gsl::span<int32_t>& GetNextTokens() { return next_beam_tokens_; }
  gsl::span<int32_t>& GetNextIndices() { return next_beam_indices_; }

 private:
  size_t batch_size_;
  size_t num_beams_;
  size_t max_length_;
  size_t num_beam_hyps_to_keep_;
  int pad_token_id_;
  int eos_token_id_;


  //std::unique_ptr<bool> done_ptr_;      // Allocated buffer for done_
  BufferUniquePtr done_ptr_;
  gsl::span<bool> done_;                // List of flags indicates whether each batch is finished or not. Its shape is (batch_size).

  BufferUniquePtr next_beam_scores_ptr_;
  gsl::span<float> next_beam_scores_;

  BufferUniquePtr next_beam_tokens_ptr_;
  gsl::span<int32_t> next_beam_tokens_;

  BufferUniquePtr next_beam_indices_ptr_;
  gsl::span<int32_t> next_beam_indices_;

  BufferUniquePtr hypothesis_buffer_ptr_;      // Allocated buffer to hold all hypotheses
  gsl::span<int32_t> hypothesis_buffer_;                // Span of the allocated buffer
  size_t hypothesis_buffer_length_;                     // Total number of elements
  size_t hypothesis_buffer_offset_;                     // Offset of avaiable buffer, or length of used buffer.

  //onnxruntime::FastAllocVector<BeamHypotheses> beam_hyps_;
  //TODO do this with fast vector
  std::vector<BeamHypotheses> beam_hyps_;
};

}