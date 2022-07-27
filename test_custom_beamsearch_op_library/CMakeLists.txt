# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Minimum CMake required
cmake_minimum_required(VERSION 3.20)

project(cc_app CXX)

set(CC_APP_ROOT ${PROJECT_SOURCE_DIR}/cc_app)
set(ORTPACKAGE_ROOT ${CMAKE_CURRENT_BINARY_DIR}/ort_package)
set(DEBUG_ROOT ${CMAKE_CURRENT_BINARY_DIR}/Debug)
set(ORTPACKAGE_INCLUDE ${ORTPACKAGE_ROOT}/include)
set(ORTPACKAGE_LIB ${ORTPACKAGE_ROOT}/lib)

set(LOCAL_HEADER_SOURCE
    "D:\\ai\\onnxruntime\\include\\onnxruntime\\core\\session\\")
set(INCLUDE_SRC
    "${LOCAL_HEADER_SOURCE}/onnxruntime_c_api.h"
    "${LOCAL_HEADER_SOURCE}/onnxruntime_cxx_api.h"
    "${LOCAL_HEADER_SOURCE}/onnxruntime_cxx_inline.h")
file(COPY ${INCLUDE_SRC} DESTINATION ${ORTPACKAGE_INCLUDE})

set(LOCAL_DLL_SOURCE
    "D:\\ai\\onnxruntime\\build\\Windows\\RelWithDebInfo\\RelWithDebInfo\\")
set(DLLS_SRC
    "${LOCAL_DLL_SOURCE}/onnxruntime.dll"
    "${LOCAL_DLL_SOURCE}/onnxruntime.lib"
    "${LOCAL_DLL_SOURCE}/onnxruntime.pdb"
    "${LOCAL_DLL_SOURCE}/onnxruntime_providers_shared.dll"
    "${LOCAL_DLL_SOURCE}/onnxruntime_providers_shared.lib"
    "${LOCAL_DLL_SOURCE}/onnxruntime_providers_shared.pdb")
file(COPY ${DLLS_SRC} DESTINATION ${ORTPACKAGE_LIB})

set(ORT_HEADER_SRC
    "${ORTPACKAGE_INCLUDE}/onnxruntime_c_api.h"
    "${ORTPACKAGE_INCLUDE}/onnxruntime_cxx_api.h"
    "${ORTPACKAGE_INCLUDE}/onnxruntime_cxx_inline.h")

file(GLOB APP_SRC
    "${CC_APP_ROOT}/*.cc")

set(ORT_DLL_SOURCE
    "${ORTPACKAGE_LIB}/onnxruntime.dll")

file(COPY ${ORT_HEADER_SRC} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${APP_HEADER_SRC} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${APP_SRC} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${ORT_DLL_SOURCE} DESTINATION ${DEBUG_ROOT})

add_executable(cc_app ${CMAKE_CURRENT_BINARY_DIR}/cc_app.cc)
target_include_directories(cc_app PRIVATE ${ORTPACKAGE_ROOT}/lib)
target_include_directories(cc_app PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(cc_app PRIVATE ${ORTPACKAGE_ROOT}/lib/onnxruntime.lib)
