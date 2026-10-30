ARG CUDA_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM ${CUDA_IMAGE}
ARG CUDA_ARCHS=75
ARG MATRIX_SIZE=8192
COPY gpu-burn /opt/gpu-burn
COPY cuda_memtest /opt/cuda_memtest
COPY build-tools.sh /opt/video-test/build-tools.sh
COPY container-tests.sh /opt/video-test/container-tests.sh
RUN bash /opt/video-test/build-tools.sh "${CUDA_ARCHS}" "${MATRIX_SIZE}"
WORKDIR /opt/gpu-burn
ENTRYPOINT ["bash", "/opt/video-test/container-tests.sh"]
