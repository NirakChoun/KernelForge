#pragma once
#include <cmath>
#include <cstdio>
#include <vector>

namespace kf {

struct CheckResult {
  size_t errors = 0;
  double max_abs_err = 0;
  bool pass() const { return errors == 0; }
};

// Element-wise |got - ref| <= atol + rtol * |ref|. NaN in got counts as an error.
template <typename T, typename R>
CheckResult check_close(const std::vector<T>& got, const std::vector<R>& ref, double rtol,
                        double atol, const char* what = "") {
  CheckResult c;
  for (size_t i = 0; i < got.size(); ++i) {
    const double g = static_cast<double>(got[i]), r = static_cast<double>(ref[i]);
    const double err = std::fabs(g - r);
    if (!(err <= atol + rtol * std::fabs(r))) {
      if (c.errors < 5)
        std::fprintf(stderr, "%s mismatch at %zu: got %.9g expected %.9g\n", what, i, g, r);
      ++c.errors;
    }
    if (err > c.max_abs_err || std::isnan(err)) c.max_abs_err = err;
  }
  return c;
}

inline void print_check(const char* what, const CheckResult& c) {
  std::printf("correctness %s: %s (errors=%zu, max_abs_err=%.3g)\n", what,
              c.pass() ? "PASS" : "FAIL", c.errors, c.max_abs_err);
}

}  // namespace kf
