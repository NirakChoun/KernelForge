#pragma once
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include "kf/cuda_check.cuh"

// Set by CMake so every row records how the binary was built.
#ifndef KF_BUILD_TYPE
#define KF_BUILD_TYPE "unknown"
#endif

namespace kf {

// Where and with what a result was produced.
struct RunInfo {
  std::string gpu, job_id, date, cuda_runtime, cuda_driver, build_type;
};

inline std::string cuda_version_string(int v) {
  return std::to_string(v / 1000) + "." + std::to_string((v % 1000) / 10);
}

inline RunInfo run_info() {
  RunInfo r;
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  r.gpu = p.name;
  const char* job = std::getenv("SLURM_JOB_ID");
  r.job_id = job ? job : "none";
  char buf[32];
  const std::time_t t = std::time(nullptr);
  std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%S%z", std::localtime(&t));
  r.date = buf;
  int rt = 0, drv = 0;
  CUDA_CHECK(cudaRuntimeGetVersion(&rt));
  CUDA_CHECK(cudaDriverGetVersion(&drv));
  r.cuda_runtime = cuda_version_string(rt);
  r.cuda_driver = cuda_version_string(drv);
  r.build_type = KF_BUILD_TYPE;
  return r;
}

// One benchmark configuration. Times are per launch.
struct ResultRow {
  std::string kernel;
  size_t n = 0;              // problem size in elements
  size_t bytes = 0;          // bytes moved per launch, basis of the derived metric
  unsigned grid = 0, block = 0;
  int launches_per_rep = 1;  // launches between one event pair
  bool l2_flush = false;     // L2 flushed before each event pair
  int warmup = 0, reps = 0;
  double median_ms = 0, min_ms = 0, stddev_ms = 0;
  double metric_value = 0;
  std::string metric_unit;   // "GB/s" or "GFLOP/s"
  bool launch_bound = false; // per-launch median under 20 us
};

inline const std::string& csv_header() {
  static const std::string h =
      "kernel,gpu,job_id,date,cuda_runtime,cuda_driver,build_type,n,bytes,grid,block,"
      "launches_per_rep,l2_flush,warmup,reps,median_ms,min_ms,stddev_ms,metric_value,"
      "metric_unit,launch_bound";
  return h;
}

inline std::string csv_field(const std::string& s) {
  if (s.find_first_of(",\"\n") == std::string::npos) return s;
  std::string q = "\"";
  for (char c : s) q += (c == '"') ? std::string("\"\"") : std::string(1, c);
  return q + "\"";
}

// Appends one row, writing the header if the file is new. Refuses to append to a
// file with a different header so schema changes never mix in one CSV.
// Not safe for concurrent writers: run configurations sequentially within one job.
inline void append_csv(const std::string& path, const RunInfo& info, const ResultRow& r) {
  namespace fs = std::filesystem;
  const fs::path p(path);
  if (p.has_parent_path()) fs::create_directories(p.parent_path());
  bool need_header = true;
  if (fs::exists(p) && fs::file_size(p) > 0) {
    std::ifstream in(p);
    std::string first;
    std::getline(in, first);
    if (first != csv_header()) {
      std::fprintf(stderr, "csv header mismatch in %s\n", path.c_str());
      std::exit(EXIT_FAILURE);
    }
    need_header = false;
  }
  std::ofstream out(p, std::ios::app);
  if (!out) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(EXIT_FAILURE);
  }
  if (need_header) out << csv_header() << '\n';
  char nums[256];
  std::snprintf(nums, sizeof nums, "%zu,%zu,%u,%u,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.1f", r.n, r.bytes,
                r.grid, r.block, r.launches_per_rep, r.l2_flush ? 1 : 0, r.warmup, r.reps,
                r.median_ms, r.min_ms, r.stddev_ms, r.metric_value);
  out << csv_field(r.kernel) << ',' << csv_field(info.gpu) << ',' << csv_field(info.job_id) << ','
      << info.date << ',' << info.cuda_runtime << ',' << info.cuda_driver << ','
      << csv_field(info.build_type) << ',' << nums << ',' << csv_field(r.metric_unit) << ','
      << (r.launch_bound ? 1 : 0) << '\n';
  if (!out) {
    std::fprintf(stderr, "write failed for %s\n", path.c_str());
    std::exit(EXIT_FAILURE);
  }
}

}  // namespace kf
