#pragma once
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include "kf/cuda_check.cuh"

namespace kf {

struct Stats {
  double min_ms, median_ms, mean_ms, stddev_ms;
  int n;
};

inline Stats summarize(std::vector<float> t) {
  std::sort(t.begin(), t.end());
  const int n = static_cast<int>(t.size());
  const double mean = std::accumulate(t.begin(), t.end(), 0.0) / n;
  double var = 0.0;
  for (float x : t) var += (x - mean) * (x - mean);
  const double median = (n % 2) ? t[n / 2] : 0.5 * (t[n / 2 - 1] + t[n / 2]);
  return {t.front(), median, mean, n > 1 ? std::sqrt(var / (n - 1)) : 0.0, n};
}

// Times fn() on the GPU using CUDA events.
// fn must enqueue its GPU work on `stream` (default stream = 0).
// Warm-up runs are executed but not recorded.
template <typename F>
std::vector<float> time_gpu(F&& fn, int warmup, int reps, cudaStream_t stream = 0) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (int i = 0; i < warmup; ++i) fn();
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<float> times(reps);
  for (int i = 0; i < reps; ++i) {
    CUDA_CHECK(cudaEventRecord(start, stream));
    fn();
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
  }
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return times;
}

}  // namespace kf
