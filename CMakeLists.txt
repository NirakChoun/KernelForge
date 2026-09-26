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

add_executable(device_info src/device_info.cu)
target_link_libraries(device_info PRIVATE kf_common)
