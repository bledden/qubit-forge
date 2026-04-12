cmake_minimum_required(VERSION 3.21)

# Ensure ROCm cmake modules are discoverable
if(NOT DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH "/opt/rocm" CACHE PATH "ROCm installation path")
else()
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "ROCm installation path")
endif()
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")

# Default to MI300X (gfx942); override with -DCMAKE_HIP_ARCHITECTURES=...
# Must be set BEFORE project() so CMake uses it during HIP language init.
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "gfx942" CACHE STRING "GPU architectures")
endif()

project(quantum LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(hip REQUIRED)

# Core library (sources added incrementally)
add_library(quantum SHARED
    src/statevec.hip
    src/kernels/single_qubit.hip
    src/kernels/two_qubit.hip
    src/kernels/diagonal.hip
    src/kernels/measure.hip
    src/circuit.hip
    src/fusion.hip
)
target_include_directories(quantum PUBLIC include)
target_link_libraries(quantum hip::device)

find_package(pybind11 REQUIRED)
# Compile as HIP so hipcc handles these (avoids g++ getting --offload-arch)
set_source_files_properties(python/pyquantum.cpp PROPERTIES LANGUAGE HIP)
pybind11_add_module(pyquantum python/pyquantum.cpp)
target_link_libraries(pyquantum PRIVATE quantum)

# Tests
set_source_files_properties(tests/test_statevec.cpp PROPERTIES LANGUAGE HIP)
add_executable(test_statevec tests/test_statevec.cpp)
target_link_libraries(test_statevec quantum)
