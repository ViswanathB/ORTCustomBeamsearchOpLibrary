# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Minimum CMake required
cmake_minimum_required(VERSION 3.20)

project(cc_app CXX)

set(CC_APP_ROOT ${PROJECT_SOURCE_DIR}/cc_app)
set(ORTPACKAGE_DOWNLOAD_ROOT ${PROJECT_SOURCE_DIR}/../build/ort_package)
set(ORTPACKAGE_DOWNLOAD_INCLUDE ${ORTPACKAGE_DOWNLOAD_ROOT}/build/native/include)
set(ORTPACKAGE_ROOT ${CMAKE_CURRENT_BINARY_DIR}/ort_package)
set(ORTPACKAGE_LIB ${ORTPACKAGE_ROOT}/lib)

set(DEBUG_ROOT ${CMAKE_CURRENT_BINARY_DIR}/Debug)

set(INCLUDE_SRC
    "${ORTPACKAGE_DOWNLOAD_INCLUDE}/onnxruntime_c_api.h"
    "${ORTPACKAGE_DOWNLOAD_INCLUDE}/onnxruntime_cxx_api.h"
    "${ORTPACKAGE_DOWNLOAD_INCLUDE}/onnxruntime_cxx_inline.h")
file(COPY ${INCLUDE_SRC} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

set(LOCAL_DLL_SOURCE ${ORTPACKAGE_DOWNLOAD_ROOT}/runtimes/win-x64/native)
set(DLLS_SRC
    "${LOCAL_DLL_SOURCE}/onnxruntime.dll"
    "${LOCAL_DLL_SOURCE}/onnxruntime.lib"
    "${LOCAL_DLL_SOURCE}/onnxruntime_providers_shared.dll"
    "${LOCAL_DLL_SOURCE}/onnxruntime_providers_shared.lib"
)
file(COPY ${DLLS_SRC} DESTINATION ${ORTPACKAGE_LIB})

file(GLOB APP_SRC
    "${CC_APP_ROOT}/*.cc")

set(ORT_DLL_SOURCE
    "${ORTPACKAGE_LIB}/onnxruntime.dll")

file(COPY ${APP_SRC} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${ORT_DLL_SOURCE} DESTINATION ${DEBUG_ROOT})

add_executable(cc_app ${CMAKE_CURRENT_BINARY_DIR}/cc_app.cc)
target_include_directories(cc_app PRIVATE ${ORTPACKAGE_ROOT}/lib)
target_include_directories(cc_app PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(cc_app PRIVATE ${ORTPACKAGE_LIB}/onnxruntime.lib)
