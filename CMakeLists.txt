# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Minimum CMake required
cmake_minimum_required(VERSION 3.20)

project(customBeamsearchOPLib CXX CUDA)

set(CUSTOMBEAMSEARCHOPLIB_ROOT ${PROJECT_SOURCE_DIR}/custom_beamsearch_op_library)
set(ORTPACKAGE_ROOT ${CMAKE_CURRENT_BINARY_DIR}/ort_package)
set(ORTPACKAGE_INCLUDE ${ORTPACKAGE_ROOT}/build/native/include)
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
     "${CMAKE_CURRENT_BINARY_DIR}/beam_search.h"
     "${CMAKE_CURRENT_BINARY_DIR}/beam_search.cc")

add_library(custom_beamsearch_op_library_cpu SHARED
    ${CMAKE_CURRENT_BINARY_DIR}/custom_beamsearch_op_library_cpu.cc
    ${CUSTOMBEAMSEARCHOPCPUIMPL_SRC}
)

set(ONNXRUNTIME_CUSTOM_BEAMSEARCH_OP_LIB_LINK_FLAG "-DEF:${CUSTOMBEAMSEARCHOPLIB_ROOT}/custom_beamsearch_op_library.def")

set_property(
  TARGET custom_beamsearch_op_library_cpu
  APPEND_STRING PROPERTY LINK_FLAGS ${ONNXRUNTIME_CUSTOM_BEAMSEARCH_OP_LIB_LINK_FLAG}
)