cmake_minimum_required(VERSION 4.2)
project(SoA CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)


add_subdirectory(tests)
add_subdirectory(benchmarks)
add_subdirectory(examples)