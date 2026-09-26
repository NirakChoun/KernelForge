// Stage 1: effective bandwidth of global memory access patterns.
// All kernels move 32-bit words and write their output contiguously, so only the
// read pattern changes. Data is uint32 so outputs are checked exactly.
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <vector>
#include "kf/bench.cuh"

using u32 = uint32_t;

__global__ void contiguous(const u32* __restrict__ in, u32* __restrict__ out, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i];
}

// Reads every input element exactly once, S elements apart within a pass: output i
// reads row (i >> row_shift) of an S-column view, column-major. n and S are powers of
// two so the index is shifts and masks, identical instruction count for every S.
__global__ void strided(const u32* __restrict__ in, u32* __restrict__ out, size_t n,
                        size_t row_mask, int row_shift, int log2_stride) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[((i & row_mask) << log2_stride) | (i >> row_shift)];
}

__global__ void gather(const u32* __restrict__ in, const u32* __restrict__ idx,
                       u32* __restrict__ out, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[idx[i]];
}

__global__ void offset_read(const u32* __restrict__ in, u32* __restrict__ out, size_t n,
                            int offset) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i + offset];
}

// 16-byte struct with 4-byte alignment, so each field is a separate 32-bit load.
struct Quad {
  u32 x, y, z, w;
};

__global__ void aos_x(const Quad* __restrict__ in, u32* __restrict__ out, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i].x;
}
__global__ void soa_x(const u32* __restrict__ x, u32* __restrict__ out, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = x[i];
}
__global__ void aos_sum(const Quad* __restrict__ in, u32* __restrict__ out, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const Quad q = in[i];
    out[i] = q.x + q.y + q.z + q.w;
  }
}
__global__ void soa_sum(const u32* __restrict__ x, const u32* __restrict__ y,
                        const u32* __restrict__ z, const u32* __restrict__ w,
                        u32* __restrict__ out, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = x[i] + y[i] + z[i] + w[i];
}

static int log2_exact(size_t v) {
  int s = 0;
  while ((size_t(1) << s) < v) ++s;
  return (size_t(1) << s) == v ? s : -1;
}

template <typename T>
static T* upload(const std::vector<T>& h) {
  T* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

static bool check_exact(const std::vector<u32>& got, const std::vector<u32>& ref) {
  size_t errors = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    if (got[i] != ref[i]) {
      if (errors < 5) std::fprintf(stderr, "mismatch at %zu: got %u expected %u\n", i, got[i], ref[i]);
      ++errors;
    }
  }
  std::printf("correctness: %s (errors=%zu of %zu)\n", errors ? "FAIL" : "PASS", errors, ref.size());
  return errors == 0;
}

int main(int argc, char** argv) {
  const kf::BenchArgs args = kf::parse_bench_args(
      argc, argv, "results/stage1/patterns.csv",
      "--pattern contiguous|strided|gather|offset|aos_x|soa_x|aos_sum|soa_sum [--stride S] [--offset O]");
  std::string pattern;
  size_t stride = 1;
  int offset = 0;
  for (size_t i = 0; i < args.rest.size(); ++i) {
    const std::string& a = args.rest[i];
    const bool has = i + 1 < args.rest.size();
    if (a == "--pattern" && has) pattern = args.rest[++i];
    else if (a == "--stride" && has) stride = std::stoull(args.rest[++i]);
    else if (a == "--offset" && has) offset = std::stoi(args.rest[++i]);
    else {
      std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
      return 2;
    }
  }
  // n is the number of output elements (and, except for AoS/SoA, input elements).
  const size_t n = args.n;
  constexpr int kBlock = 256;
  constexpr int kMaxOffset = 32;
  const unsigned grid = static_cast<unsigned>((n + kBlock - 1) / kBlock);

  kf::Record r;
  r.kernel = pattern;
  r.n = n;
  r.grid = std::to_string(grid);
  r.block = std::to_string(kBlock);
  std::vector<u32> ref(n), got(n);
  std::vector<void*> to_free;
  u32* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(u32)));
  CUDA_CHECK(cudaMemset(d_out, 0xFF, n * sizeof(u32)));
  std::function<void()> launch;

  if (pattern == "contiguous" || pattern == "strided" || pattern == "gather") {
    std::vector<u32> h_in(n);
    for (size_t j = 0; j < n; ++j) h_in[j] = static_cast<u32>(j);
    u32* d_in = upload(h_in);
    to_free.push_back(d_in);
    if (pattern == "contiguous") {
      for (size_t i = 0; i < n; ++i) ref[i] = static_cast<u32>(i);
      r.bytes = 2.0 * n * sizeof(u32);
      r.ki = kf::kernel_info(contiguous, kBlock);
      launch = [=] { contiguous<<<grid, kBlock>>>(d_in, d_out, n); };
    } else if (pattern == "strided") {
      const int ls = log2_exact(stride), ln = log2_exact(n);
      if (ls < 0 || ln < 0 || stride > n) {
        std::fprintf(stderr, "strided needs power-of-two n and stride <= n\n");
        return 2;
      }
      const int row_shift = ln - ls;
      const size_t row_mask = (size_t(1) << row_shift) - 1;
      for (size_t i = 0; i < n; ++i) ref[i] = static_cast<u32>(((i & row_mask) << ls) | (i >> row_shift));
      r.variant = "stride=" + std::to_string(stride);
      r.bytes = 2.0 * n * sizeof(u32);
      r.ki = kf::kernel_info(strided, kBlock);
      launch = [=] { strided<<<grid, kBlock>>>(d_in, d_out, n, row_mask, row_shift, ls); };
    } else {
      // Random permutation, so each input element is read once, in random order.
      std::vector<u32> h_idx(n);
      for (size_t i = 0; i < n; ++i) h_idx[i] = static_cast<u32>(i);
      std::mt19937_64 rng(12345);
      for (size_t i = n - 1; i > 0; --i) std::swap(h_idx[i], h_idx[rng() % (i + 1)]);
      ref = h_idx;
      u32* d_idx = upload(h_idx);
      to_free.push_back(d_idx);
      r.variant = "random_permutation";
      r.bytes = 3.0 * n * sizeof(u32);  // index read, data read, output write
      r.ki = kf::kernel_info(gather, kBlock);
      launch = [=] { gather<<<grid, kBlock>>>(d_in, d_idx, d_out, n); };
    }
  } else if (pattern == "offset") {
    if (offset < 0 || offset > kMaxOffset) {
      std::fprintf(stderr, "--offset must be 0..%d\n", kMaxOffset);
      return 2;
    }
    // Input has kMaxOffset extra elements so every offset reads the same count.
    std::vector<u32> h_in(n + kMaxOffset);
    for (size_t j = 0; j < h_in.size(); ++j) h_in[j] = static_cast<u32>(j);
    u32* d_in = upload(h_in);
    to_free.push_back(d_in);
    for (size_t i = 0; i < n; ++i) ref[i] = static_cast<u32>(i + offset);
    r.variant = "offset=" + std::to_string(offset);
    r.bytes = 2.0 * n * sizeof(u32);
    r.ki = kf::kernel_info(offset_read, kBlock);
    launch = [=] { offset_read<<<grid, kBlock>>>(d_in, d_out, n, offset); };
  } else if (pattern == "aos_x" || pattern == "aos_sum") {
    std::vector<Quad> h_in(n);
    for (size_t i = 0; i < n; ++i) {
      const u32 b = static_cast<u32>(4 * i);
      h_in[i] = {b, b + 1, b + 2, b + 3};
      ref[i] = pattern == "aos_x" ? b : b + (b + 1) + (b + 2) + (b + 3);
    }
    Quad* d_in = upload(h_in);
    to_free.push_back(d_in);
    // Useful bytes: fields actually consumed plus the output.
    r.bytes = (pattern == "aos_x" ? 2.0 : 5.0) * n * sizeof(u32);
    if (pattern == "aos_x") {
      r.ki = kf::kernel_info(aos_x, kBlock);
      launch = [=] { aos_x<<<grid, kBlock>>>(d_in, d_out, n); };
    } else {
      r.ki = kf::kernel_info(aos_sum, kBlock);
      launch = [=] { aos_sum<<<grid, kBlock>>>(d_in, d_out, n); };
    }
  } else if (pattern == "soa_x" || pattern == "soa_sum") {
    std::vector<u32> hx(n), hy(n), hz(n), hw(n);
    for (size_t i = 0; i < n; ++i) {
      const u32 b = static_cast<u32>(4 * i);
      hx[i] = b, hy[i] = b + 1, hz[i] = b + 2, hw[i] = b + 3;
      ref[i] = pattern == "soa_x" ? b : b + (b + 1) + (b + 2) + (b + 3);
    }
    u32 *dx = upload(hx), *dy = upload(hy), *dz = upload(hz), *dw = upload(hw);
    to_free.insert(to_free.end(), {dx, dy, dz, dw});
    r.bytes = (pattern == "soa_x" ? 2.0 : 5.0) * n * sizeof(u32);
    if (pattern == "soa_x") {
      r.ki = kf::kernel_info(soa_x, kBlock);
      launch = [=] { soa_x<<<grid, kBlock>>>(dx, d_out, n); };
    } else {
      r.ki = kf::kernel_info(soa_sum, kBlock);
      launch = [=] { soa_sum<<<grid, kBlock>>>(dx, dy, dz, dw, d_out, n); };
    }
  } else {
    std::fprintf(stderr, "unknown --pattern '%s'\n", pattern.c_str());
    return 2;
  }

  launch();
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(got.data(), d_out, n * sizeof(u32), cudaMemcpyDeviceToHost));
  std::printf("pattern=%s variant=%s n=%zu\n", pattern.c_str(), r.variant.c_str(), n);
  const bool pass = check_exact(got, ref);
  if (pass) {
    r.s = kf::bench(launch, args);
    kf::report(r, args);
  }
  for (void* p : to_free) CUDA_CHECK(cudaFree(p));
  CUDA_CHECK(cudaFree(d_out));
  return pass ? 0 : 1;
}
