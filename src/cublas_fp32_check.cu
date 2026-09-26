// Verifies that the Stage 3 cuBLAS baseline computes in FP32, not TF32.
// For each size: cublasSgemm with the default math mode (as in src/sgemm.cu) and with
// TF32 tensor-op math explicitly enabled, both compared with a cublasDgemm reference on
// the same inputs. Error is max_ij |C - C_fp64| / (|A||B|)_ij. FP32 accumulation gives
// errors near 2^-24 per term; TF32 inputs (10-bit mantissa) give errors near 2^-11.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <cublas_v2.h>
#include "kf/cuda_check.cuh"

#define CUBLAS_CHECK(call)                                                        \
  do {                                                                            \
    cublasStatus_t st_ = (call);                                                  \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                           \
      std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", int(st_), __FILE__, __LINE__); \
      std::exit(EXIT_FAILURE);                                                    \
    }                                                                             \
  } while (0)

static const char* env(const char* k) {
  const char* v = std::getenv(k);
  return v ? v : "(unset)";
}

static double max_rel_err(const std::vector<float>& c, const std::vector<double>& ref,
                          const std::vector<double>& absprod) {
  double worst = 0;
  for (size_t i = 0; i < c.size(); ++i) worst = std::fmax(worst, std::fabs(c[i] - ref[i]) / absprod[i]);
  return worst;
}

int main(int argc, char** argv) {
  std::printf("NVIDIA_TF32_OVERRIDE=%s\n", env("NVIDIA_TF32_OVERRIDE"));
  std::printf("CUBLAS_EMULATION_STRATEGY=%s\n", env("CUBLAS_EMULATION_STRATEGY"));
  cublasHandle_t h;
  CUBLAS_CHECK(cublasCreate(&h));
  cublasMath_t mode;
  CUBLAS_CHECK(cublasGetMathMode(h, &mode));
  std::printf("math mode after cublasCreate: %d (CUBLAS_DEFAULT_MATH=%d, CUBLAS_TF32_TENSOR_OP_MATH=%d)\n",
              int(mode), int(CUBLAS_DEFAULT_MATH), int(CUBLAS_TF32_TENSOR_OP_MATH));
  int ver = 0;
  CUBLAS_CHECK(cublasGetVersion(h, &ver));
  std::printf("cuBLAS version: %d\n", ver);
  for (int a = 1; a < argc; ++a) {
    const int n = std::atoi(argv[a]);
    const size_t sz = size_t(n) * n;
    std::vector<float> hA(sz), hB(sz), hC(sz);
    std::vector<double> dA(sz), dB(sz), dAa(sz), dBa(sz), ref(sz), absprod(sz);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < sz; ++i) {
      hA[i] = dist(rng);
      dA[i] = hA[i];
      dAa[i] = std::fabs(dA[i]);
    }
    for (size_t i = 0; i < sz; ++i) {
      hB[i] = dist(rng);
      dB[i] = hB[i];
      dBa[i] = std::fabs(dB[i]);
    }
    float *fA, *fB, *fC;
    double *gA, *gB, *gC;
    CUDA_CHECK(cudaMalloc(&fA, sz * 4));
    CUDA_CHECK(cudaMalloc(&fB, sz * 4));
    CUDA_CHECK(cudaMalloc(&fC, sz * 4));
    CUDA_CHECK(cudaMalloc(&gA, sz * 8));
    CUDA_CHECK(cudaMalloc(&gB, sz * 8));
    CUDA_CHECK(cudaMalloc(&gC, sz * 8));
    CUDA_CHECK(cudaMemcpy(fA, hA.data(), sz * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(fB, hB.data(), sz * 4, cudaMemcpyHostToDevice));
    const double one_d = 1, zero_d = 0;
    // Row-major C = A B as column-major C^T = B^T A^T, as in src/sgemm.cu.
    auto dgemm = [&](const std::vector<double>& x, const std::vector<double>& y, std::vector<double>& out) {
      CUDA_CHECK(cudaMemcpy(gA, x.data(), sz * 8, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(gB, y.data(), sz * 8, cudaMemcpyHostToDevice));
      CUBLAS_CHECK(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one_d, gB, n, gA, n, &zero_d, gC, n));
      CUDA_CHECK(cudaMemcpy(out.data(), gC, sz * 8, cudaMemcpyDeviceToHost));
    };
    dgemm(dA, dB, ref);
    dgemm(dAa, dBa, absprod);
    const float one = 1, zero = 0;
    for (cublasMath_t m : {CUBLAS_DEFAULT_MATH, CUBLAS_TF32_TENSOR_OP_MATH}) {
      CUBLAS_CHECK(cublasSetMathMode(h, m));
      CUDA_CHECK(cudaMemset(fC, 0xFF, sz * 4));
      CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, fB, n, fA, n, &zero, fC, n));
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(hC.data(), fC, sz * 4, cudaMemcpyDeviceToHost));
      std::printf("CHECK n=%d math=%s max_err/(|A||B|)=%.3e\n", n,
                  m == CUBLAS_DEFAULT_MATH ? "default" : "tf32_tensor_op", max_rel_err(hC, ref, absprod));
    }
    CUBLAS_CHECK(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));
    for (void* p : {(void*)fA, (void*)fB, (void*)fC, (void*)gA, (void*)gB, (void*)gC}) CUDA_CHECK(cudaFree(p));
  }
  CUBLAS_CHECK(cublasDestroy(h));
  return 0;
}
