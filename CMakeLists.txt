# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Minimum CMake required
cmake_minimum_required(VERSION 3.20)

project(customBeamsearchOPLib CXX CUDA)

set(CUSTOMBEAMSEARCHOPLIB_ROOT ${PROJECT_SOURCE_DIR}/custom_beamsearch_op_library)
set(ORTPACKAGE_ROOT ${CMAKE_CURRENT_BINARY_DIR}/ort_package)
set(ORTPACKAGE_INCLUDE ${ORTPACKAGE_ROOT}/build/native/include)

# bugbug: use ort 1.12
set (LOCAL_HEADER_SOURCE
      "onnxruntime\\include\\onnxruntime\\core\\session\\")
set (INCLUDE_SRC
      "${LOCAL_HEADER_SOURCE}/onnxruntime_c_api.h"
      "${LOCAL_HEADER_SOURCE}/onnxruntime_cxx_api.h"
      "${LOCAL_HEADER_SOURCE}/onnxruntime_cxx_inline.h")
file(COPY ${INCLUDE_SRC} DESTINATION ${ORTPACKAGE_INCLUDE})

set(ORT_HEADER_SRC
    "${ORTPACKAGE_INCLUDE}/onnxruntime_c_api.h"
    "${ORTPACKAGE_INCLUDE}/onnxruntime_cxx_api.h"
    "${ORTPACKAGE_INCLUDE}/onnxruntime_cxx_inline.h")

file(GLOB CUSTOMBEAMSEARCHOPLIB_SRC
    "${CUSTOMBEAMSEARCHOPLIB_ROOT}/*.h"
    "${CUSTOMBEAMSEARCHOPLIB_ROOT}/*.cc"
    "${CUSTOMBEAMSEARCHOPLIB_ROOT}/*.def"
)

file(COPY ${ORT_HEADER_SRC} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CUSTOMBEAMSEARCHOPLIB_SRC} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

file(GLOB CUSTOMBEAMSEARCHOPCPUIMPL_SRC
     "${CMAKE_CURRENT_BINARY_DIR}/utils.h"
     "${CMAKE_CURRENT_BINARY_DIR}/utils.cc"
     "${CMAKE_CURRENT_BINARY_DIR}/beam_search_shared.h"
     "${CMAKE_CURRENT_BINARY_DIR}/beam_search_parameters.h"
     "${CMAKE_CURRENT_BINARY_DIR}/beam_search_parameters.cc"
     "${CMAKE_CURRENT_BINARY_DIR}/beam_search_scorer.h"
     "${CMAKE_CURRENT_BINARY_DIR}/beam_search_scorer.cc"
     "${CMAKE_CURRENT_BINARY_DIR}/sequences.h"
     "${CMAKE_CURRENT_BINARY_DIR}/sequences.cc"
     "${CMAKE_CURRENT_BINARY_DIR}/dump_tensor.h"
     "${CMAKE_CURRENT_BINARY_DIR}/dump_tensor.cc"
     "${CMAKE_CURRENT_BINARY_DIR}/beam_search_device_helper.h"
     "${CMAKE_CURRENT_BINARY_DIR}/beam_search_device_helper.cc"
     "${CMAKE_CURRENT_BINARY_DIR}/beam_search.h"
     "${CMAKE_CURRENT_BINARY_DIR}/beam_search.cc"
     "${CMAKE_CURRENT_BINARY_DIR}/logits_processor.h"
     "${CMAKE_CURRENT_BINARY_DIR}/logits_processor.cc"
     )

include(FetchContent)
FetchContent_Declare(GSL
    GIT_REPOSITORY "https://github.com/microsoft/GSL"
    GIT_TAG "v3.1.0"
)
FetchContent_MakeAvailable(GSL)

add_library(custom_beamsearch_op_library_cpu SHARED
    ${CMAKE_CURRENT_BINARY_DIR}/custom_beamsearch_op_library_cpu.cc
    ${CUSTOMBEAMSEARCHOPCPUIMPL_SRC}
)
target_link_libraries(custom_beamsearch_op_library_cpu PRIVATE GSL)

set(ONNXRUNTIME_CUSTOM_BEAMSEARCH_OP_LIB_LINK_FLAG "-DEF:${CUSTOMBEAMSEARCHOPLIB_ROOT}/custom_beamsearch_op_library.def")

set_property(
  TARGET custom_beamsearch_op_library_cpu
  APPEND_STRING PROPERTY LINK_FLAGS ${ONNXRUNTIME_CUSTOM_BEAMSEARCH_OP_LIB_LINK_FLAG}
)