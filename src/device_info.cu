#include <cstdio>
#include "kf/cuda_check.cuh"

// Deliberately trivial: one thread writes one value.
// Purpose is only to prove the CPU -> GPU -> CPU path works.
__global__ void smoke(int* out) { *out = 42; }

int main() {
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));

  int mem_clk_khz = 0, bus_bits = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&mem_clk_khz, cudaDevAttrMemoryClockRate, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev));
  // Nominal peak = 2 (double data rate) * clock * bus width in bytes.
  // Treat this as a hypothesis to test against measured bandwidth, not truth.
  const double peak_gbs = 2.0 * mem_clk_khz * 1e3 * (bus_bits / 8.0) / 1e9;

  std::printf("device:                 %s\n", p.name);
  std::printf("compute capability:     %d.%d\n", p.major, p.minor);
  std::printf("SMs:                    %d\n", p.multiProcessorCount);
  std::printf("global memory:          %.1f GiB\n", p.totalGlobalMem / double(1 << 30));
  std::printf("L2 cache:               %.1f MiB\n", p.l2CacheSize / double(1 << 20));
  std::printf("shared mem / block:     %zu KiB\n", p.sharedMemPerBlock / 1024);
  std::printf("shared mem / SM:        %zu KiB\n", p.sharedMemPerMultiprocessor / 1024);
  std::printf("registers / SM:         %d\n", p.regsPerMultiprocessor);
  std::printf("max threads / SM:       %d\n", p.maxThreadsPerMultiProcessor);
  std::printf("max threads / block:    %d\n", p.maxThreadsPerBlock);
  std::printf("warp size:              %d\n", p.warpSize);
  std::printf("memory bus width:       %d bits\n", bus_bits);
  std::printf("memory clock:           %d MHz\n", mem_clk_khz / 1000);
  std::printf("nominal peak bandwidth: %.1f GB/s\n", peak_gbs);

  int* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(int)));
  smoke<<<1, 1>>>(d_out);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaDeviceSynchronize());
  int h_out = 0;
  CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_out));

  std::printf("smoke test:             %s\n", h_out == 42 ? "PASS" : "FAIL");
  return h_out == 42 ? 0 : 1;
}
