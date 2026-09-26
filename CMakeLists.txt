cmake_minimum_required(VERSION 3.24)

# Must be set BEFORE project(), or CMake picks its own default.
# 120 = sm_120 (RTX PRO 6000 Blackwell). Override for other GPUs:
#   cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86   (A6000)
#   cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=80   (A100)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

project(KernelForge LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Shared settings every KernelForge target links against.
add_library(kf_common INTERFACE)
target_include_directories(kf_common INTERFACE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(kf_common INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -Xcompiler=-Wall,-Wextra>)
target_compile_definitions(kf_common INTERFACE KF_BUILD_TYPE="${CMAKE_BUILD_TYPE}")

# Print per-kernel registers, spills, and shared memory at compile time.
option(KF_PTXAS_VERBOSE "Pass -Xptxas -v to nvcc" OFF)
if(KF_PTXAS_VERBOSE)
  target_compile_options(kf_common INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
endif()

function(kf_add_bench name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE kf_common)
endfunction()

kf_add_bench(device_info src/device_info.cu)
kf_add_bench(vector_add src/vector_add.cu)
kf_add_bench(saxpy src/saxpy.cu)
kf_add_bench(copy src/copy.cu)
kf_add_bench(mem_patterns src/mem_patterns.cu)
kf_add_bench(reduce src/reduce.cu)

find_package(CUDAToolkit REQUIRED)
kf_add_bench(sgemm src/sgemm.cu)
target_link_libraries(sgemm PRIVATE CUDA::cublas)

kf_add_bench(cublas_fp32_check src/cublas_fp32_check.cu)
target_link_libraries(cublas_fp32_check PRIVATE CUDA::cublas)
