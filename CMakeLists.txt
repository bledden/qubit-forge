cmake_minimum_required(VERSION 3.21)
project(quantum LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(hip REQUIRED)

# Core library (sources added incrementally)
add_library(quantum SHARED
    src/statevec.hip
    src/kernels/single_qubit.hip
)
target_include_directories(quantum PUBLIC include)
target_link_libraries(quantum hip::device)

# Tests
add_executable(test_statevec tests/test_statevec.cpp)
target_link_libraries(test_statevec quantum)
