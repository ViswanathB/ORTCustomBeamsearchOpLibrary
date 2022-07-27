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

inline void MakeStringImpl(std::ostringstream & /*ss*/) noexcept
{
}

template <typename T>
inline void MakeStringImpl(std::ostringstream &ss, const T &t) noexcept
{
  ss << t;
}

template <typename T, typename... Args>
inline void MakeStringImpl(std::ostringstream &ss, const T &t, const Args &...args) noexcept
{
  MakeStringImpl(ss, t);
  MakeStringImpl(ss, args...);
}

template <typename... Args>
static const char *MakeString(const Args &...args) noexcept
{
  std::ostringstream ss;
  MakeStringImpl(ss, args...);
  return ss.str().c_str();
}

#define CUSTOMOP_ENFORCE(condition, ...)                                                         \
  do                                                                                             \
  {                                                                                              \
    if (!(condition))                                                                            \
    {                                                                                            \
      std::cout << MakeString("Assertion failed in beam search custom op dll ", #condition,      \
                              " in file: ", __FILE__, " in line: ", __LINE__, ":", __VA_ARGS__); \
      abort();                                                                                   \
    }                                                                                            \
  } while (false);

#define CUSTOMOP_RETURN_IF_ERROR(expr) \
  do                                   \
  {                                    \
    auto status = (expr);              \
    if (status != nullptr)             \
    {                                  \
      return status;                   \
    }                                  \
  } while (0)

class BufferDeleter
{
public:
  BufferDeleter(OrtAllocator *allocator) : allocator_(allocator) {}
  BufferDeleter() : allocator_(nullptr) {}

  void operator()(void *p)
  {
    allocator_->Free(allocator_, p);
  }

private:
  OrtAllocator *allocator_;
};
using BufferUniquePtr = std::unique_ptr<void, BufferDeleter>;

/* Utility functions
 */
int64_t SizeHelper(std::vector<int64_t> &array);
uint64_t GetTimeMs64();

template <typename T>
gsl::span<T> AllocateBufferUniquePtr(OrtAllocator *allocator,
                                     BufferUniquePtr &buffer,
                                     size_t elements,
                                     bool fill = false,
                                     T fill_value = T{});
