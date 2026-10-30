#!/usr/bin/env bash
set -uo pipefail
if [[ ${1:-} == --help ]]; then
  echo 'One visible CUDA GPU per container. Logs: /results. MEM_PASSES=1 MEM_ITERATIONS=100 MEM_TIMEOUT=1800 BURN_SECONDS=900 BURN_MEMORY=90% SMOKE=0'
  exit 0
fi
mkdir -p /results
status=FAILED
finish() {
  rc=$?
  printf '%s\n' "$status" > /results/status.txt
  date -Is > /results/end.txt
  exit "$rc"
}
trap finish EXIT
for key in MEM_PASSES MEM_ITERATIONS MEM_TIMEOUT BURN_SECONDS; do
  val=${!key:-}
  [[ -z $val || $val =~ ^[1-9][0-9]*$ ]] || { echo "Invalid $key" >&2; exit 2; }
done
date -Is > /results/start.txt
cp /opt/video-test/architectures.txt /opt/video-test/matrix-size.txt /results/
args=(--device 0 --num_passes "${MEM_PASSES:-1}" --num_iterations "${MEM_ITERATIONS:-100}" --exit_on_error)
if [[ ${SMOKE:-0} == 1 ]]; then
  args=(--device 0 --stress --num_passes 1 --num_iterations 1 --max_num_blocks 64)
fi
timeout "${MEM_TIMEOUT:-1800}" stdbuf -oL -eL /opt/cuda_memtest/cuda_memtest "${args[@]}" > /results/memtest.log 2>&1
rc=$?
echo "$rc" > /results/memtest.exit
if [[ $rc != 0 ]] || grep -Eiq '(^|[^[:alpha:]])(error|fail|failed)([^[:alpha:]]|$)' /results/memtest.log || ! grep -q 'main thread: Program exits' /results/memtest.log; then
  echo MEMORY_TEST_FAILED_OR_INCOMPLETE >&2
  exit 1
fi
cd /opt/gpu-burn
timeout "$(( ${BURN_SECONDS:-900} + 60 ))" stdbuf -oL -eL ./gpu_burn -i 0 -m "${BURN_MEMORY:-90%}" "${BURN_SECONDS:-900}" > /results/gpu-burn.log 2>&1
rc=$?
echo "$rc" > /results/gpu-burn.exit
if [[ $rc != 0 ]] || grep -Eiq 'FAULTY|failed|aborting|(^|[^[:alpha:]])error([^[:alpha:]]|$)|errors:[[:space:]]*[1-9]' /results/gpu-burn.log || ! grep -Eq 'GPU 0: OK' /results/gpu-burn.log; then
  echo GPU_BURN_FAILED_OR_INCOMPLETE >&2
  exit 1
fi
status=PASSED
