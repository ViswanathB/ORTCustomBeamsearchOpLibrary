// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <iostream>

using namespace std;

void print_assertion() {
    std::cout << std::endl;
}

template<typename First, typename... Rest>
void print_assertion(First first, Rest&&... rest)
{
    std::cout << first << std::endl;
    print_assertion(std::forward<Rest>(rest)...);
}

#define CUSTOMOP_ENFORCE(condition, ...)                                    \
    do {                                                                    \
        if (!(condition)) {                                                 \
            print_assertion("Assertion failed ", #condition,                \
            " in file: ", __FILE__, " in line: ", __LINE__, __VA_ARGS__);   \
        }                                                                   \
        abort();                                                            \
    } while (false);