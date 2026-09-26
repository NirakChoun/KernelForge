// Stage 3: row-major FP32 C = A * B (M x K times K x N). One version per --version;
// version 0 is cuBLAS. See docs/stage3.md.
#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <vector>
#include <cublas_v2.h>
#include "kf/bench.cuh"

#define CUBLAS_CHECK(call)                                                        \
  do {                                                                            \
    cublasStatus_t st_ = (call);                                                  \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                           \
      std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", int(st_), __FILE__, __LINE__); \
      std::exit(EXIT_FAILURE);                                                    \
    }                                                                             \
  } while (0)

static int cdiv(int a, int b) { return (a + b - 1) / b; }

// v1: one thread per output. threadIdx.x selects the row, so the 32 threads of a warp
// read A from 32 different rows and write C with a stride of N floats.
__global__ void sgemm_naive(int M, int N, int K, const float* __restrict__ A,
                            const float* __restrict__ B, float* __restrict__ C) {
  const int row = blockIdx.x * 32 + threadIdx.x;
  const int col = blockIdx.y * 32 + threadIdx.y;
  if (row < M && col < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
  }
}

struct Plan {
  std::string name, variant, grid, block;
  kf::KernelInfo ki;
  std::function<void()> run;
};

struct Problem {
  int M, N, K;
  const float *A, *B;
  float* C;
};

static std::string dims(dim3 d) {
  return d.y > 1 ? std::to_string(d.x) + "x" + std::to_string(d.y) : std::to_string(d.x);
}

template <typename Kern>
static Plan kernel_plan(const std::string& name, const std::string& variant, Kern k, dim3 grid,
                        dim3 block, const Problem& p) {
  Plan pl;
  pl.name = name;
  pl.variant = variant;
  pl.grid = dims(grid);
  pl.block = dims(block);
  pl.ki = kf::kernel_info(k, block.x * block.y);
  pl.run = [=] { k<<<grid, block>>>(p.M, p.N, p.K, p.A, p.B, p.C); };
  return pl;
}

static Plan make_plan(int version, const std::string& cfg, const Problem& p, cublasHandle_t h) {
  auto bad_cfg = [&]() -> Plan {
    std::fprintf(stderr, "unknown --cfg '%s' for version %d\n", cfg.c_str(), version);
    std::exit(2);
  };
  switch (version) {
    case 0: {
      Plan pl;
      pl.name = "cublas";
      pl.variant = "cublasSgemm";
      pl.grid = pl.block = "-";
      // Row-major C = A * B is column-major C^T = B^T * A^T: pass B first.
      pl.run = [=] {
        const float one = 1.0f, zero = 0.0f;
        CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, p.N, p.M, p.K, &one, p.B, p.N, p.A,
                                 p.K, &zero, p.C, p.N));
      };
      return pl;
    }
    case 1:
      if (!cfg.empty()) return bad_cfg();
      return kernel_plan("v1_naive", "-", sgemm_naive, dim3(cdiv(p.M, 32), cdiv(p.N, 32)),
                         dim3(32, 32), p);
    default:
      std::fprintf(stderr, "unknown --version %d\n", version);
      std::exit(2);
  }
}

// Per-element check: |got - ref| <= tol * (|A||B|)_ij. The factor 16 sqrt(K) 2^-24 is
// far above the typical rounding error of either summation order (about 2^-24 relative
// to |A||B|) and below the error from one wrong or missing product term for K <= 8192.
static bool check(const char* what, const std::vector<float>& got, const std::vector<double>& ref,
                  const std::vector<double>& absprod, int K) {
  const double tol = 16.0 * std::sqrt(double(K)) * std::ldexp(1.0, -24);
  size_t errors = 0;
  double worst = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double e = std::fabs(double(got[i]) - ref[i]);
    const double ratio = absprod[i] > 0 ? e / absprod[i] : e;
    if (!(ratio <= tol)) {
      if (errors < 5)
        std::fprintf(stderr, "%s mismatch at %zu: got %.9g ref %.9g\n", what, i, got[i], ref[i]);
      ++errors;
    }
    if (ratio > worst || std::isnan(ratio)) worst = ratio;
  }
  std::printf("correctness %s: %s (errors=%zu, max_err/(|A||B|)=%.3g, tol=%.3g)\n", what,
              errors ? "FAIL" : "PASS", errors, worst, tol);
  return errors == 0;
}

int main(int argc, char** argv) {
  const kf::BenchArgs args = kf::parse_bench_args(
      argc, argv, "results/stage3/sgemm.csv", "--version V [--cfg C] [--ncols N] [--k K]");
  int version = -1;
  std::string cfg;
  int M = static_cast<int>(args.n), N = M, K = M;
  for (size_t i = 0; i < args.rest.size(); ++i) {
    const std::string& a = args.rest[i];
    const bool has = i + 1 < args.rest.size();
    if (a == "--version" && has) version = std::stoi(args.rest[++i]);
    else if (a == "--cfg" && has) cfg = args.rest[++i];
    else if (a == "--ncols" && has) N = std::stoi(args.rest[++i]);
    else if (a == "--k" && has) K = std::stoi(args.rest[++i]);
    else {
      std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
      return 2;
    }
  }
  if (M <= 0 || N <= 0 || K <= 0 || double(M) * K > 2e9 || double(K) * N > 2e9 ||
      double(M) * N > 2e9) {
    std::fprintf(stderr, "bad dimensions\n");
    return 2;
  }
  const size_t sa = size_t(M) * K, sb = size_t(K) * N, sc = size_t(M) * N;
  std::vector<float> hA(sa), hB(sb), hAabs(sa), hBabs(sb), hC(sc), hRef(sc), hAbsProd(sc);
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < sa; ++i) hAabs[i] = std::fabs(hA[i] = dist(rng));
  for (size_t i = 0; i < sb; ++i) hBabs[i] = std::fabs(hB[i] = dist(rng));

  float *dA, *dB, *dC, *dAabs, *dBabs, *dRef, *dAbsProd;
  for (float** p : {&dA, &dAabs}) CUDA_CHECK(cudaMalloc(p, sa * sizeof(float)));
  for (float** p : {&dB, &dBabs}) CUDA_CHECK(cudaMalloc(p, sb * sizeof(float)));
  for (float** p : {&dC, &dRef, &dAbsProd}) CUDA_CHECK(cudaMalloc(p, sc * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), sa * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), sb * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dAabs, hAabs.data(), sa * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dBabs, hBabs.data(), sb * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dC, 0xFF, sc * sizeof(float)));  // NaN sentinel

  cublasHandle_t h;
  CUBLAS_CHECK(cublasCreate(&h));
  CUBLAS_CHECK(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));  // FP32, no TF32

  // Reference and |A||B| from cuBLAS.
  make_plan(0, "", Problem{M, N, K, dA, dB, dRef}, h).run();
  make_plan(0, "", Problem{M, N, K, dAabs, dBabs, dAbsProd}, h).run();
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(hRef.data(), dRef, sc * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hAbsProd.data(), dAbsProd, sc * sizeof(float), cudaMemcpyDeviceToHost));

  const Plan plan = make_plan(version, cfg, Problem{M, N, K, dA, dB, dC}, h);
  std::printf("version=%s variant=%s M=%d N=%d K=%d grid=%s block=%s\n", plan.name.c_str(),
              plan.variant.c_str(), M, N, K, plan.grid.c_str(), plan.block.c_str());
  plan.run();
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(hC.data(), dC, sc * sizeof(float), cudaMemcpyDeviceToHost));

  bool pass = true;
  // CPU reference in double for small problems; checks cuBLAS and the kernel.
  if (double(M) * N * K <= double(1 << 30)) {
    std::vector<double> cpu(sc, 0.0), cpu_abs(sc, 0.0);
    for (int r = 0; r < M; ++r)
      for (int k = 0; k < K; ++k) {
        const double a = hA[size_t(r) * K + k], aa = std::fabs(a);
        const float* brow = &hB[size_t(k) * N];
        double* crow = &cpu[size_t(r) * N];
        double* arow = &cpu_abs[size_t(r) * N];
        for (int c = 0; c < N; ++c) {
          crow[c] += a * brow[c];
          arow[c] += aa * std::fabs(brow[c]);
        }
      }
    pass &= check("cublas_vs_cpu", hRef, cpu, cpu_abs, K);
    pass &= check("kernel_vs_cpu", hC, cpu, cpu_abs, K);
  }
  {
    std::vector<double> ref(hRef.begin(), hRef.end()), absprod(hAbsProd.begin(), hAbsProd.end());
    pass &= check("kernel_vs_cublas", hC, ref, absprod, K);
  }

  if (pass) {
    kf::Record r;
    r.kernel = plan.name;
    r.variant = plan.variant;
    r.n = static_cast<size_t>(M);
    r.bytes = 4.0 * (double(sa) + sb + sc);  // compulsory traffic: read A, B, write C
    r.flops = 2.0 * M * N * K;
    r.unit = "GFLOP/s";
    r.grid = plan.grid;
    r.block = plan.block;
    r.ki = plan.ki;
    r.s = kf::bench(plan.run, args);
    kf::CsvRow extra;
    extra.add("M", M).add("N", N).add("K", K);
    kf::report(r, args, extra);
  }
  CUBLAS_CHECK(cublasDestroy(h));
  for (float* p : {dA, dB, dC, dAabs, dBabs, dRef, dAbsProd}) CUDA_CHECK(cudaFree(p));
  return pass ? 0 : 1;
}
