#include <cstdint>
#include "beam_search.h"
#include <iostream>

using namespace std;

void BeamSearchCPU(const int size,
  const float* input1,
  const float* input2,
  int32_t* output) {
    for (int64_t i = 0; i < size; i++) {
        output[i] = static_cast<int>(input1[i] + (*input2));
    }
  }