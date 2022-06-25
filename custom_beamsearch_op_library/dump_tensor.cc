// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "dump_tensor.h"

namespace custombsop {

#ifdef DEBUG_BEAM_SEARCH
constexpr int64_t kDefaultSnippetEdgeItems = 10;

// Skip non edge items in last dimension
#define SKIP_NON_EDGE_ITEMS_LAST_DIM(dim_size, index, edge_items)                          \
  if (dim_size > 2 * edge_items && index >= edge_items && index + edge_items < dim_size) { \
    if (index == edge_items) {                                                             \
      std::cout << ", ... ";                                                               \
    }                                                                                      \
    continue;                                                                              \
  }

// Skip non edge items in other dimensions except the last dimension
#define SKIP_NON_EDGE_ITEMS(dim_size, index, edge_items)                                   \
  if (dim_size > 2 * edge_items && index >= edge_items && index + edge_items < dim_size) { \
    if (index == edge_items) {                                                             \
      std::cout << "..." << std::endl;                                                     \
    }                                                                                      \
    continue;                                                                              \
  }

template <typename T>
void DumpCpuTensor(const char* name, const T* tensor, int dim0, int dim1, int64_t edge_items) {
  if (nullptr != name) {
    std::cout << std::string(name) << std::endl;
  }

  for(int i=0;i<dim0;i++) {
      SKIP_NON_EDGE_ITEMS(dim0, i, edge_items);
      std::cout<<tensor[i*dim1];
      for (int j=1;j<dim1;j++){
          SKIP_NON_EDGE_ITEMS_LAST_DIM(dim0, i, edge_items);
          std::cout<<","<<tensor[i*dim1+j];
      }
      std::cout<<std::endl;
  }
}

// Print snippet of 3D tensor with shape (dim0, dim1, dim2)
template <typename T>
void DumpCpuTensor(const char* name, const T* tensor, int64_t dim0, int64_t dim1, int64_t dim2, int64_t edge_items) {
  if (nullptr != name) {
    std::cout << std::string(name) << std::endl;
  }

  for (int64_t i = 0; i < dim0; i++) {
    SKIP_NON_EDGE_ITEMS(dim0, i, edge_items);
    for (int64_t j = 0; j < dim1; j++) {
      SKIP_NON_EDGE_ITEMS(dim1, j, edge_items);
      std::cout<<tensor[i * dim1 * dim2 + j * dim2];
      for (int64_t k = 1; k < dim2; k++) {
        SKIP_NON_EDGE_ITEMS_LAST_DIM(dim2, k, edge_items);
        std::cout << ", "<<tensor[i * dim1 * dim2 + j * dim2 + k];
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void CpuTensorConsoleDumper::Print(const char* name, const float* tensor, int dim0, int dim1) const {
  DumpCpuTensor<float>(name, tensor, dim0, dim1, kDefaultSnippetEdgeItems);
}

void CpuTensorConsoleDumper::Print(const char* name, const int64_t* tensor, int dim0, int dim1) const {
  DumpCpuTensor<int64_t>(name, tensor, dim0, dim1, kDefaultSnippetEdgeItems);
}

void CpuTensorConsoleDumper::Print(const char* name, const int32_t* tensor, int dim0, int dim1) const {
  DumpCpuTensor<int32_t>(name, tensor, dim0, dim1, kDefaultSnippetEdgeItems);
}

void CpuTensorConsoleDumper::Print(const char* name, const float* tensor, int dim0, int dim1, int dim2) const {
  DumpCpuTensor<float>(name, tensor, dim0, dim1, dim2, kDefaultSnippetEdgeItems);
}

void CpuTensorConsoleDumper::Print(const char* name, const int64_t* tensor, int dim0, int dim1, int dim2) const {
  DumpCpuTensor<int64_t>(name, tensor, dim0, dim1, dim2, kDefaultSnippetEdgeItems);
}

void CpuTensorConsoleDumper::Print(const char* name, const int32_t* tensor, int dim0, int dim1, int dim2) const {
  DumpCpuTensor<int32_t>(name, tensor, dim0, dim1, dim2, kDefaultSnippetEdgeItems);
}
#else

void CpuTensorConsoleDumper::Print(const char* name, const float* tensor, int dim0, int dim1) const {
}

void CpuTensorConsoleDumper::Print(const char* name, const int64_t* tensor, int dim0, int dim1) const {
}

void CpuTensorConsoleDumper::Print(const char* name, const int32_t* tensor, int dim0, int dim1) const {
}

void CpuTensorConsoleDumper::Print(const char* name, const float* tensor, int dim0, int dim1, int dim2) const {
}

void CpuTensorConsoleDumper::Print(const char* name, const int64_t* tensor, int dim0, int dim1, int dim2) const {
}

void CpuTensorConsoleDumper::Print(const char* name, const int32_t* tensor, int dim0, int dim1, int dim2) const {
}

#endif

} // namespace custombsop
