#pragma once
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <cuda_runtime.h>
#include "kf/cuda_check.cuh"

// Set by CMake so every row records how the binary was built.
#ifndef KF_BUILD_TYPE
#define KF_BUILD_TYPE "unknown"
#endif

namespace kf {

// Where and with what a result was produced. Written as the first columns of every row.
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

// Ordered (column, value) pairs. Each benchmark defines its own columns after the
// RunInfo columns; the header check in append_csv keeps one schema per file.
class CsvRow {
 public:
  CsvRow& add(const std::string& key, const std::string& v) {
    cols_.emplace_back(key, v);
    return *this;
  }
  CsvRow& add(const std::string& key, const char* v) { return add(key, std::string(v)); }
  template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
  CsvRow& add(const std::string& key, T v) {
    return add(key, std::to_string(v));
  }
  CsvRow& add(const std::string& key, double v, int precision) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.*f", precision, v);
    return add(key, std::string(buf));
  }
  const std::vector<std::pair<std::string, std::string>>& cols() const { return cols_; }

 private:
  std::vector<std::pair<std::string, std::string>> cols_;
};

inline std::string csv_field(const std::string& s) {
  if (s.find_first_of(",\"\n") == std::string::npos) return s;
  std::string q = "\"";
  for (char c : s) q += (c == '"') ? std::string("\"\"") : std::string(1, c);
  return q + "\"";
}

// Appends one row, writing the header if the file is new. Refuses to append to a
// file with a different header so schema changes never mix in one CSV.
// Not safe for concurrent writers: run configurations sequentially within one job.
inline void append_csv(const std::string& path, const RunInfo& info, const CsvRow& row) {
  namespace fs = std::filesystem;
  std::string header = "gpu,job_id,date,cuda_runtime,cuda_driver,build_type";
  std::string line = csv_field(info.gpu) + ',' + csv_field(info.job_id) + ',' + info.date + ',' +
                     info.cuda_runtime + ',' + info.cuda_driver + ',' + csv_field(info.build_type);
  for (const auto& [k, v] : row.cols()) {
    header += ',' + k;
    line += ',' + csv_field(v);
  }
  const fs::path p(path);
  if (p.has_parent_path()) fs::create_directories(p.parent_path());
  bool need_header = true;
  if (fs::exists(p) && fs::file_size(p) > 0) {
    std::ifstream in(p);
    std::string first;
    std::getline(in, first);
    if (first != header) {
      std::fprintf(stderr, "csv header mismatch in %s\n  file: %s\n  row:  %s\n", path.c_str(),
                   first.c_str(), header.c_str());
      std::exit(EXIT_FAILURE);
    }
    need_header = false;
  }
  std::ofstream out(p, std::ios::app);
  if (!out) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(EXIT_FAILURE);
  }
  if (need_header) out << header << '\n';
  out << line << '\n';
  if (!out) {
    std::fprintf(stderr, "write failed for %s\n", path.c_str());
    std::exit(EXIT_FAILURE);
  }
}

}  // namespace kf
