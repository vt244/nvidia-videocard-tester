#!/usr/bin/env bash
set -euo pipefail
archs=${1:?CUDA_ARCHS required}
matrix=${2:-8192}
[[ $archs =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Invalid CUDA_ARCHS' >&2; exit 2; }
[[ $matrix =~ ^(1024|2048|4096|8192)$ ]] || exit 2
flags=()
IFS=, read -ra targets <<< "$archs"
for arch in "${targets[@]}"; do
  # A native cubin per architecture: no reliance on forward PTX JIT.
  virtual=$arch
  [[ $arch != 21 ]] || virtual=20
  flags+=("-gencode=arch=compute_${virtual},code=sm_${arch}")
done
# Keep adaptations in the image; upstream source directories stay intact.
sed -i \
  -e 's/^#define SIZE 8192ul$/#define SIZE MATRIX_SIZE/' \
  -e 's/^#define OPS_PER_MUL 1100048498688ul.*/#define OPS_PER_MUL (2.0 * SIZE * SIZE * SIZE)/' \
  -e 's/(size_t)useBytes < 3 \* d_resultSize/(size_t)useBytes < 4 * d_resultSize/' \
  -e '/        if (d_tensors)/i #if CUDA_VERSION >= 9000' \
  -e '/checkError(cublasSetMathMode.*CUBLAS_TENSOR_OP_MATH));/a #endif' \
  -e '/        checkError(cuParamSetSize/,/                   "set block size");/d' \
  -e '/        checkError(cuLaunchGridAsync/,/                   "Launch grid");/c\
        void *args[] = {\&d_Cdata, \&d_faultyElemData, \&d_iters};\
        checkError(cuLaunchKernel(d_function, SIZE / g_blockSize, SIZE / g_blockSize, 1,\
                                 g_blockSize, g_blockSize, 1, 0, 0, args, nullptr), "Launch kernel");' \
  /opt/gpu-burn/gpu_burn-drv.cpp
make -C /opt/gpu-burn clean
make -C /opt/gpu-burn -j4 COMPUTE= NVCCFLAGS="${flags[*]}" CFLAGS="-DMATRIX_SIZE=${matrix}ul"
cd /opt/cuda_memtest
nvcc -O2 -std=c++11 "${flags[@]}" -DENABLE_NVML=0 -Xcompiler -pthread \
  -x cu cuda_memtest.cpp tests.cpp misc.cpp -o cuda_memtest
printf '%s\n' "$archs" > /opt/video-test/architectures.txt
printf '%s\n' "$matrix" > /opt/video-test/matrix-size.txt
