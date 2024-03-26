cmake_minimum_required(VERSION 3.15)
project(cublasLt_profile LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE CUDA::cudart CUDA::cublasLt)
