#include "utils.h"

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

/* Returns the amount of milliseconds elapsed since the UNIX epoch. Works on both
 * windows and linux. */
uint64_t GetTimeMs64()
{
#ifdef _WIN32
    /* Windows */
    FILETIME ft;
    LARGE_INTEGER li;

    /* Get the amount of 100 nano seconds intervals elapsed since January 1, 1601 (UTC) and copy it
     * to a LARGE_INTEGER structure. */
    GetSystemTimeAsFileTime(&ft);
    li.LowPart = ft.dwLowDateTime;
    li.HighPart = ft.dwHighDateTime;

    uint64_t ret = li.QuadPart;
    ret -= 116444736000000000LL; /* Convert from file time to UNIX epoch time. */
    ret /= 10000;                /* From 100 nano seconds (10^-7) to 1 millisecond (10^-3) intervals */

    return ret;
#else
    /* Linux */
    struct timeval tv;

    gettimeofday(&tv, NULL);

    uint64_t ret = tv.tv_usec;
    /* Convert from micro seconds (10^-6) to milliseconds (10^-3) */
    ret /= 1000;

    /* Adds the seconds (10^0) after converting them to milliseconds (10^-3) */
    ret += (tv.tv_sec * 1000);

    return ret;
#endif
}

int64_t SizeHelper(std::vector<int64_t> &array)
{
    int64_t total_size = 1;
    for (size_t i = 0; i < array.size(); i++)
    {
        CUSTOMOP_ENFORCE(array[i] >= 0)
        total_size *= array[i];
    }

    return total_size;
}

template <typename T>
gsl::span<T> AllocateBufferUniquePtr(OrtAllocator *allocator,
                                     BufferUniquePtr &buffer,
                                     size_t elements,
                                     bool fill,
                                     T fill_value)
{
    size_t bytes = sizeof(T) * elements;

    // buffer = BufferUniquePtr(malloc(bytes), BufferDeleter());
    buffer = BufferUniquePtr(allocator->Alloc(allocator, bytes), BufferDeleter(allocator));

    T *first = reinterpret_cast<T *>(buffer.get());
    auto span = gsl::make_span(first, elements);

    if (fill)
    {
        std::fill_n(first, elements, fill_value);
    }

    return span;
}

template gsl::span<float> AllocateBufferUniquePtr(OrtAllocator *allocator, BufferUniquePtr &buffer, size_t elements, bool fill, float fill_value);
template gsl::span<int> AllocateBufferUniquePtr(OrtAllocator *allocator, BufferUniquePtr &buffer, size_t elements, bool fill, int fill_value);
template gsl::span<bool> AllocateBufferUniquePtr(OrtAllocator *allocator, BufferUniquePtr &buffer, size_t elements, bool fill, bool fill_value);
