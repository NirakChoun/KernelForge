#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include "kf/csv.hpp"
#include "kf/cuda_check.cuh"
#include "kf/timer.cuh"

namespace kf {

// Per-launch median below this is marked launch-overhead dominated.
constexpr double kLaunchBoundMs = 0.020;

struct BenchArgs {
  size_t n = 0;
  bool flush = false;
  int launches = 1;
  int warmup = 10;
  int reps = 100;
  std::string csv;
  std::vector<std::string> rest;  // binary-specific flags, parsed by the caller
};

// Common CLI: <n> [--flush | --launches K] [--warmup W] [--reps R] [--csv PATH | --no-csv]
// [binary flags...]. W >= 10 and R >= 100 (project minimums); defaults 10 and 100.
inline BenchArgs parse_bench_args(int argc, char** argv, const std::string& default_csv,
                                  const char* extra_usage = "") {
  auto usage = [&] {
    std::fprintf(stderr,
                 "usage: %s <n> [--flush | --launches K] [--warmup W] [--reps R] [--csv PATH | --no-csv] %s\n",
                 argv[0], extra_usage);
    std::exit(2);
  };
  if (argc < 2) usage();
  BenchArgs a;
  a.n = std::strtoull(argv[1], nullptr, 10);
  a.csv = default_csv;
  if (a.n == 0) usage();
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--csv") && i + 1 < argc) a.csv = argv[++i];
    else if (!std::strcmp(argv[i], "--no-csv")) a.csv.clear();
    else if (!std::strcmp(argv[i], "--flush")) a.flush = true;
    else if (!std::strcmp(argv[i], "--launches") && i + 1 < argc) a.launches = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc) a.warmup = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) a.reps = std::atoi(argv[++i]);
    else a.rest.emplace_back(argv[i]);
  }
  if (a.launches < 1 || a.warmup < 10 || a.reps < 100) usage();
  if (a.flush && a.launches > 1) {
    // Only the first launch of each rep would be cold, so the result would be neither.
    std::fprintf(stderr, "--flush requires --launches 1\n");
    std::exit(2);
  }
  return a;
}

// Static resource use and theoretical occupancy from the runtime API. Needs no
// performance counters.
struct KernelInfo {
  int regs = -1;         // registers per thread
  int local_bytes = -1;  // local memory per thread (spills and stack arrays)
  int static_smem = -1;  // static shared memory per block
  int blocks_per_sm = -1;
  double occupancy = -1;  // active warps / max warps per SM
};

template <typename Kernel>
KernelInfo kernel_info(Kernel kernel, int block_threads, size_t dyn_smem = 0) {
  KernelInfo k;
  cudaFuncAttributes attr{};
  CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
  k.regs = attr.numRegs;
  k.local_bytes = static_cast<int>(attr.localSizeBytes);
  k.static_smem = static_cast<int>(attr.sharedSizeBytes);
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&k.blocks_per_sm, kernel,
                                                           block_threads, dyn_smem));
  int dev = 0, max_threads = 0, warp = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&max_threads, cudaDevAttrMaxThreadsPerMultiProcessor, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&warp, cudaDevAttrWarpSize, dev));
  const int warps_per_block = (block_threads + warp - 1) / warp;
  k.occupancy = double(k.blocks_per_sm * warps_per_block) / (max_threads / warp);
  return k;
}

template <typename F>
Stats bench(F&& fn, const BenchArgs& a) {
  std::unique_ptr<L2Flush> flusher;
  if (a.flush) flusher = std::make_unique<L2Flush>();
  TimeOptions opt;
  opt.warmup = a.warmup;
  opt.reps = a.reps;
  opt.launches_per_rep = a.launches;
  opt.flush = flusher.get();
  return summarize(time_gpu(fn, opt));
}

// One measured configuration. bytes and flops are per launch; the metric uses one of them.
struct Record {
  std::string kernel;
  std::string variant = "-";  // configuration detail, e.g. "stride=4" or "tile=32"
  size_t n = 0;
  double bytes = 0;
  double flops = 0;
  std::string grid, block;  // launch configuration, "-" when not a kernel launch
  size_t dyn_smem = 0;
  KernelInfo ki;
  Stats s{};
  std::string unit = "GB/s";  // "GB/s" or "GFLOP/s"

  double metric() const {
    const double sec = s.median_ms * 1e-3;
    return (unit == "GFLOP/s" ? flops : bytes) / sec / 1e9;
  }
  bool launch_bound() const { return s.median_ms < kLaunchBoundMs; }
};

inline CsvRow to_row(const Record& r, const BenchArgs& a) {
  CsvRow row;
  row.add("kernel", r.kernel)
      .add("variant", r.variant)
      .add("n", r.n)
      .add("bytes", r.bytes, 0)
      .add("flops", r.flops, 0)
      .add("grid", r.grid)
      .add("block", r.block)
      .add("dyn_smem", r.dyn_smem)
      .add("regs", r.ki.regs)
      .add("local_bytes", r.ki.local_bytes)
      .add("static_smem", r.ki.static_smem)
      .add("blocks_per_sm", r.ki.blocks_per_sm)
      .add("theo_occupancy", r.ki.occupancy, 3)
      .add("launches_per_rep", a.launches)
      .add("l2_flush", a.flush ? 1 : 0)
      .add("warmup", a.warmup)
      .add("reps", a.reps)
      .add("median_ms", r.s.median_ms, 6)
      .add("min_ms", r.s.min_ms, 6)
      .add("stddev_ms", r.s.stddev_ms, 6)
      .add("metric_value", r.metric(), 1)
      .add("metric_unit", r.unit)
      .add("launch_bound", r.launch_bound() ? 1 : 0);
  return row;
}

// Prints one line and appends to the CSV unless --no-csv. extra adds benchmark-specific
// columns after the common ones.
inline void report(const Record& r, const BenchArgs& a, const CsvRow& extra = CsvRow()) {
  std::printf(
      "RESULT kernel=%s variant=%s n=%zu grid=%s block=%s regs=%d local=%d smem=%d occ=%.3f "
      "K=%d flush=%d median_ms=%.6f min_ms=%.6f stddev_ms=%.6f %s=%.1f launch_bound=%d\n",
      r.kernel.c_str(), r.variant.c_str(), r.n, r.grid.c_str(), r.block.c_str(), r.ki.regs,
      r.ki.local_bytes, r.ki.static_smem, r.ki.occupancy, a.launches, a.flush ? 1 : 0,
      r.s.median_ms, r.s.min_ms, r.s.stddev_ms, r.unit.c_str(), r.metric(),
      r.launch_bound() ? 1 : 0);
  if (!a.csv.empty()) {
    CsvRow row = to_row(r, a);
    for (const auto& [k, v] : extra.cols()) row.add(k, v);
    append_csv(a.csv, run_info(), row);
  }
}

}  // namespace kf
