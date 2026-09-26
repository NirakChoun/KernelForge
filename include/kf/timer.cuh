#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
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

namespace detail {
// Value changes every call so no pass can be skipped as redundant.
static __global__ void l2_flush_kernel(unsigned* buf, size_t n, unsigned v) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
    buf[i] = v ^ static_cast<unsigned>(i);
}
}  // namespace detail

// Evicts benchmark data from L2 by writing a scratch buffer of
// max(min_bytes, 2 * L2 size). Default min is 256 MiB.
class L2Flush {
 public:
  explicit L2Flush(size_t min_bytes = size_t(256) << 20) {
    int dev = 0, l2 = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev));
    bytes_ = std::max(min_bytes, 2 * static_cast<size_t>(l2));
    CUDA_CHECK(cudaMalloc(&buf_, bytes_));
  }
  ~L2Flush() { cudaFree(buf_); }
  L2Flush(const L2Flush&) = delete;
  L2Flush& operator=(const L2Flush&) = delete;

  void operator()(cudaStream_t stream) const {
    detail::l2_flush_kernel<<<1024, 256, 0, stream>>>(buf_, bytes_ / sizeof(unsigned), ++calls_);
    CUDA_CHECK_LAST();
  }
  size_t bytes() const { return bytes_; }

 private:
  unsigned* buf_ = nullptr;
  size_t bytes_ = 0;
  mutable unsigned calls_ = 0;
};

struct TimeOptions {
  int warmup = 10;
  int reps = 100;
  const L2Flush* flush = nullptr;  // if set, runs before each rep, outside the timed region
  cudaStream_t stream = 0;
};

// Times fn() on the GPU using CUDA events. fn must enqueue its work on opt.stream.
// Warm-up runs are executed but not recorded.
template <typename F>
std::vector<float> time_gpu(F&& fn, const TimeOptions& opt) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (int i = 0; i < opt.warmup; ++i) fn();
  CUDA_CHECK(cudaStreamSynchronize(opt.stream));

  std::vector<float> times(opt.reps);
  for (int i = 0; i < opt.reps; ++i) {
    // Stream order guarantees the flush finishes before the start event.
    if (opt.flush) (*opt.flush)(opt.stream);
    CUDA_CHECK(cudaEventRecord(start, opt.stream));
    fn();
    CUDA_CHECK(cudaEventRecord(stop, opt.stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
  }
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return times;
}

template <typename F>
std::vector<float> time_gpu(F&& fn, int warmup, int reps, cudaStream_t stream = 0) {
  TimeOptions opt;
  opt.warmup = warmup;
  opt.reps = reps;
  opt.stream = stream;
  return time_gpu(fn, opt);
}

}  // namespace kf
