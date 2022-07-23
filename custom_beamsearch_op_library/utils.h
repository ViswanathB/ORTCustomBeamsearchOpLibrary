// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <iostream>
#include <ostream>
#include <sstream>
#include <vector>
#include <stdlib.h>
#include <stdint.h>

#include <gsl/gsl>
#include <safeint.h>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

using namespace std;

//TODO Move all the functions to utils.cc and keep only declaration here.
static void print_assertion() {
    std::cout << std::endl;
}

template<typename First, typename... Rest>
static void print_assertion(First first, Rest&&... rest)
{
    std::cout << first;
    print_assertion(std::forward<Rest>(rest)...);
}

//TODO ORT_CXX_API_THROW can be used instead of abort.
#define CUSTOMOP_ENFORCE(condition, ...)                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            print_assertion("Assertion failed ", #condition,                    \
            " in file: ", __FILE__, " in line: ", __LINE__,":" , __VA_ARGS__);  \
            abort();                                                            \
        }                                                                       \
    } while (false);

#define CUSTOMOP_RETURN_IF_ERROR(expr)                                      \
    do {                                                                    \
        auto status = (expr);                                               \
        if (status != nullptr) {                                            \
            return status;                                                  \
        }                                                                   \
    } while (0)


inline void MakeStringImpl(std::ostringstream& /*ss*/) noexcept {
}

template <typename T>
inline void MakeStringImpl(std::ostringstream& ss, const T& t) noexcept {
ss << t;
}

template <typename T, typename... Args>
inline void MakeStringImpl(std::ostringstream& ss, const T& t, const Args&... args) noexcept {
MakeStringImpl(ss, t);
MakeStringImpl(ss, args...);
}

template <typename ...Args>
static const char* MakeString(const Args& ...args) noexcept {
  std::ostringstream ss;
  MakeStringImpl(ss, args...);
  return ss.str().c_str();
}

template <typename T>
gsl::span<T> AllocateBuffer(OrtAllocator *allocator,
                            void **buffer,
                            size_t elements,
                            bool fill = false,
                            T fill_value = T{}) {

  //size_t bytes = SafeInt<size_t>(sizeof(T)) * elements;
  size_t bytes = sizeof(T) * elements;

  *buffer = malloc(bytes);

  T* first = reinterpret_cast<T*>(*buffer);
  auto span = gsl::make_span(first, elements);

  if (fill) {
    std::fill_n(first, elements, fill_value);
  }

  return span;
}

class BufferDeleter {
 public:
  BufferDeleter(OrtAllocator* allocator): allocator_(allocator) {}
  BufferDeleter(): allocator_(nullptr) {}
  
  void operator()(void* p) {
    allocator_->Free(allocator_, p);
  }

  private:
  //TODO is a shared pointer better?
  OrtAllocator* allocator_;
};

//TODO Update this to use ort allocator
using BufferUniquePtr = std::unique_ptr<void, BufferDeleter>;

template <typename T>
gsl::span<T> AllocateBufferUniquePtr(OrtAllocator *allocator,
                                    BufferUniquePtr &buffer,
                                    size_t elements,
                                    bool fill = false,
                                    T fill_value = T{}) {
  size_t bytes = sizeof(T) * elements;

  //buffer = BufferUniquePtr(malloc(bytes), BufferDeleter());
  buffer = BufferUniquePtr(allocator->Alloc(allocator, bytes), BufferDeleter(allocator));

  T* first = reinterpret_cast<T*>(buffer.get());
  auto span = gsl::make_span(first, elements);

  if (fill) {
    std::fill_n(first, elements, fill_value);
  }

  return span;
}


static int64_t SizeHelper(std::vector<int64_t> &array) {
  int64_t total_size = 1;
  for (size_t i=0; i<array.size(); i++) {
    CUSTOMOP_ENFORCE(array[i] >= 0)
    total_size *= array[i];
  }

  return total_size;
}

uint64_t GetTimeMs64();