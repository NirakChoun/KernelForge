#!/bin/bash
# Counter-free static analysis for one or more targets. Runs on the login node
# (compile and disassemble only, no GPU work).
#   results/<stage>/ptxas_<target>.txt     registers, spills, smem per kernel (-Xptxas -v)
#   results/<stage>/sass_ops_<target>.csv  static SASS opcode counts per kernel
#   results/<stage>/sass_<target>.txt      full SASS, only with SAVE_SASS=1
# usage: [SAVE_SASS=1] scripts/kernel_report.sh <stage> <target>...
set -euo pipefail
cd ~/KernelForge
source scripts/modules.sh >/dev/null 2>&1
out=results/$1
shift
mkdir -p "$out"
cmake -S . -B build-ptxas -DKF_PTXAS_VERBOSE=ON >/dev/null
for t in "$@"; do
  # Force a recompile of this target so ptxas prints its report.
  find build-ptxas/CMakeFiles/"$t".dir -name '*.o' -delete 2>/dev/null || true
  cmake --build build-ptxas --target "$t" 2>&1 \
    | grep -E "ptxas info|Compiling entry|spill" | c++filt > "$out/ptxas_$t.txt"
  cuobjdump -sass "build-ptxas/$t" | c++filt > "build-ptxas/$t.sass"
  python3 scripts/sass_stats.py < "build-ptxas/$t.sass" > "$out/sass_ops_$t.csv"
  if [ "${SAVE_SASS:-0}" = 1 ]; then cp "build-ptxas/$t.sass" "$out/sass_$t.txt"; fi
  echo "$t: $(grep -c 'Compiling entry' "$out/ptxas_$t.txt") kernels"
done
