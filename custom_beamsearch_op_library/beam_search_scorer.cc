// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "beam_search_scorer.h"
#include "utils.h"

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

namespace custombsop {

BeamHypotheses::BeamHypotheses(int num_beams,
                               float length_penalty,
                               bool early_stopping)
                               //onnxruntime::OrtStlAllocator<HypothesisScore>& hypothesis_score_allocator)
    : num_beams_(num_beams),
      length_penalty_(length_penalty),
      early_stopping_(early_stopping),
      worst_score_(1e9) {
      //beams_(hypothesis_score_allocator) {
}

void BeamHypotheses::Add(gsl::span<const int32_t>& hypothesis, float sum_logprobs) {
  auto length = hypothesis.size();
  float score = sum_logprobs / pow(static_cast<float>(length), length_penalty_);

  if (this->Size() < num_beams_ || score > worst_score_) {
    HypothesisScore item(hypothesis, score);
    beams_.push(item);
    if (this->Size() > num_beams_) {
      beams_.pop();
    }
    worst_score_ = beams_.top().score;
  }
}

bool BeamHypotheses::IsDone(float best_sum_logprobs, int current_length) {
  // If there are enough hypotheses and that none of the hypotheses being generated can become better
  // than the worst one in the heap, then we are done with this sentence.

  if (Size() < num_beams_)
    return false;

  if (early_stopping_)
    return true;

  float current_score = best_sum_logprobs / pow(static_cast<float>(current_length), length_penalty_);
  return worst_score_ >= current_score;
}


BeamSearchScorer::BeamSearchScorer(size_t batch_size,
                                   size_t num_beams,
                                   size_t max_length,
                                   float length_penalty,
                                   bool early_stopping,
                                   size_t num_return_sequences,
                                   int pad_token_id,
                                   int eos_token_id)
                                   //onnxruntime::OrtStlAllocator<HypothesisScore>& hypothesis_score_allocator,
                                   //onnxruntime::OrtStlAllocator<BeamHypotheses>& beam_hyps_allocator)
    : batch_size_(batch_size),
      num_beams_(num_beams),
      max_length_(max_length),
      num_beam_hyps_to_keep_(num_return_sequences),
      pad_token_id_(pad_token_id),
      eos_token_id_(eos_token_id),
      hypothesis_buffer_length_(0),
      hypothesis_buffer_offset_(0) {
      //beam_hyps_(beam_hyps_allocator) {
        for (size_t i = 0; i < batch_size_; i++) {
            //beam_hyps_.push_back(BeamHypotheses(num_beams, length_penalty, early_stopping, hypothesis_score_allocator));
            beam_hyps_.push_back(BeamHypotheses(num_beams, length_penalty, early_stopping));
        }
    }    

void BeamSearchScorer::Initialize(OrtAllocator* ort_allocator, int sequence_length) {
  CUSTOMOP_ENFORCE(next_beam_scores_.empty());  // Make sure this is called only once.

  size_t batch_beam_size = batch_size_ * num_beams_;
  constexpr bool no_fill = false;  // do not fill values after allocation

  void* temp_ptr;
  done_ = AllocateBuffer<bool>(ort_allocator, &temp_ptr, batch_size_, no_fill);
  std::fill_n(done_.data(), done_.size(), false);
  done_ptr_ = std::move(std::unique_ptr<bool>(reinterpret_cast<bool*>(temp_ptr)));

  next_beam_scores_ = AllocateBuffer<float>(ort_allocator, &temp_ptr, batch_beam_size, no_fill);
  next_beam_scores_ptr_ = std::move(std::unique_ptr<float>(reinterpret_cast<float*>(temp_ptr)));
  next_beam_tokens_ = AllocateBuffer<int32_t>(ort_allocator, &temp_ptr, batch_beam_size, no_fill);
  next_beam_tokens_ptr_ = std::move(std::unique_ptr<int32_t>(reinterpret_cast<int32_t*>(temp_ptr)));
  next_beam_indices_ = AllocateBuffer<int32_t>(ort_allocator, &temp_ptr, batch_beam_size, no_fill);
  next_beam_indices_ptr_=  std::move(std::unique_ptr<int32_t>(reinterpret_cast<int32_t*>(temp_ptr)));

  // Space to store intermediate sequence with length sequence_length, sequence_length + 1, ..., max_sequence_length.
  size_t buffer_per_beam = (static_cast<size_t>(max_length_) * (max_length_ + 1) - static_cast<size_t>(sequence_length - 1) * sequence_length) / 2;
  hypothesis_buffer_length_ = batch_beam_size * buffer_per_beam;

  hypothesis_buffer_ = AllocateBuffer<int32_t>(ort_allocator, &temp_ptr, hypothesis_buffer_length_, no_fill);
  hypothesis_buffer_ptr_ = std::move(std::unique_ptr<int32_t>(reinterpret_cast<int32_t*>(temp_ptr)));
}

void BeamSearchScorer::Process(ISequences* sequences,
                       gsl::span<const float>& next_scores,
                       gsl::span<const int32_t>& next_tokens,
                       gsl::span<const int32_t>& next_indices) {

}

void BeamSearchScorer::Finalize(ISequences* sequences,
                        gsl::span<const float>& final_beam_scores,
                        OrtValue* output_sequences,
                        OrtValue* output_sequence_scores) {

}
} //namespace custombsop