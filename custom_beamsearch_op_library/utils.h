// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <iostream>
#include <ostream>
#include <sstream>

using namespace std;

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
  *buffer = allocator->Alloc(allocator, bytes);

  //void* data = allocator->Alloc(bytes);
  //BufferUniquePtr temp_buffer(data, BufferDeleter(allocator));
  //buffer = std::move(temp_buffer);

  T* first = reinterpret_cast<T*>(*buffer);
  auto span = gsl::make_span(first, elements);

  if (fill) {
    std::fill_n(first, elements, fill_value);
  }

  return span;
}