cmake_minimum_required(VERSION 3.21)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Check if HIP/ROCm is available — GPU targets are optional
set(HAS_HIP FALSE)
if(NOT DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH "/opt/rocm" CACHE PATH "ROCm installation path")
else()
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "ROCm installation path")
endif()

if(EXISTS "${ROCM_PATH}/bin/hipcc")
    set(HAS_HIP TRUE)
    list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES "gfx942" CACHE STRING "GPU architectures")
    endif()
    project(quantum LANGUAGES CXX HIP)
    find_package(hip REQUIRED)
    message(STATUS "ROCm/HIP found — building GPU simulator + decoder")
else()
    project(quantum LANGUAGES CXX)
    message(STATUS "ROCm/HIP not found — building decoder only (CPU)")
endif()

# ===== GPU Simulator (requires HIP) =====
if(HAS_HIP)
    add_library(quantum SHARED
        src/statevec.hip
        src/kernels/single_qubit.hip
        src/kernels/two_qubit.hip
        src/kernels/diagonal.hip
        src/kernels/measure.hip
        src/circuit.hip
        src/fusion.hip
        src/noise.hip
    )
    target_include_directories(quantum PUBLIC include)
    target_link_libraries(quantum hip::device)

    find_package(pybind11 REQUIRED)
    set_source_files_properties(python/pyquantum.cpp PROPERTIES LANGUAGE HIP)
    pybind11_add_module(pyquantum python/pyquantum.cpp)
    target_link_libraries(pyquantum PRIVATE quantum)

    set_source_files_properties(tests/test_statevec.cpp PROPERTIES LANGUAGE HIP)
    add_executable(test_statevec tests/test_statevec.cpp)
    target_link_libraries(test_statevec quantum)
endif()

# ===== Decoder (pure CPU C++ — builds everywhere) =====
add_library(decoder SHARED
    decoder/src/types.cpp
    decoder/src/union_find.cpp
)
target_include_directories(decoder PUBLIC decoder/include)

# Decoder Python bindings
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    pybind11_add_module(pydecoder decoder/python/pydecoder.cpp)
    target_link_libraries(pydecoder PRIVATE decoder)
    message(STATUS "pybind11 found — building pydecoder Python module")
else()
    message(STATUS "pybind11 not found — skipping pydecoder (use pip install pybind11)")
endif()
