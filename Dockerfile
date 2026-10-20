# syntax=docker/dockerfile:1.7
# ══════════════════════════════════════════════════════════════════════════
# llama-cpp-turboquant tqp-v0.4.0 + ROCm 6.3.3 for gfx1010
# (RX 5700 XT + 2x RX 5600 XT) on a CPU without AVX (e.g. Celeron 3855U)
#
# Build:
#   docker build -t llama-tqp040:rocm633 -f Dockerfile .
#
# ROCm 6.3.0 variant (apt repo "6.3") if 6.3.3 hangs with multi-GPU:
#   docker build --build-arg ROCM_APT=6.3 --build-arg ROCM_TAG=6.3.0 \
#     -t llama-tqp040:rocm630 -f Dockerfile .
#
# Another turboquant tag/commit:
#   docker build --build-arg TURBOQUANT_REF=<tag-or-sha> ...
# ══════════════════════════════════════════════════════════════════════════

# ── STAGE 1: builder ───────────────────────────────────────────────────────
FROM ubuntu:22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive

ARG ROCM_APT=6.3.3
ARG ROCM_TAG=6.3.3
ARG TURBOQUANT_REF=tqp-v0.4.0

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    wget gnupg2 ca-certificates git cmake build-essential ninja-build \
    python3 python3-pip python3-venv python3-yaml libmsgpack-dev \
    libnuma-dev gfortran

# ROCm apt repo pinned at priority 1000 (avoids mixing versions from other repos)
RUN mkdir -p /etc/apt/keyrings && \
    wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key \
    | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg && \
    printf 'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/%s jammy main\n' "${ROCM_APT}" \
    > /etc/apt/sources.list.d/rocm.list && \
    printf 'Package: *\nPin: origin repo.radeon.com\nPin-Priority: 1000\n' \
    > /etc/apt/preferences.d/rocm-pin

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    rocm-dev hip-dev rocm-cmake hipblaslt-dev hipblas-dev rocsparse-dev

ENV PATH=/opt/rocm/bin:$PATH
ENV PYTHONPATH=/opt/rocm/lib/python3.10/site-packages
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install joblib msgpack PyYAML

# -- rocBLAS from source: gfx1010 kernels + host code without AVX ------------
RUN git clone --depth=1 https://github.com/ROCm/rocBLAS.git -b rocm-${ROCM_TAG} && \
    cd rocBLAS && mkdir build && cd build && \
    cmake .. \
        -GNinja \
        -DCMAKE_CXX_COMPILER=amdclang++ \
        -DCMAKE_C_COMPILER=amdclang \
        -DAMDGPU_TARGETS="gfx1010" \
        -DGPU_TARGETS="gfx1010" \
        -DCMAKE_CXX_FLAGS="-march=x86-64 -mno-avx -mno-avx2 -mno-fma -mno-bmi2" \
        -DCMAKE_C_FLAGS="-march=x86-64 -mno-avx -mno-avx2 -mno-fma -mno-bmi2" \
        -DCMAKE_INSTALL_PREFIX=/opt/rocm \
        -DBUILD_CLIENTS_TESTS=OFF \
        -DBUILD_CLIENTS_BENCHMARKS=OFF \
        -DBUILD_CLIENTS_SAMPLES=OFF && \
    ninja install && \
    cd / && rm -rf rocBLAS

# -- llama-cpp-turboquant (tag or SHA), shallow fetch ------------------------
WORKDIR /app
RUN git init . && \
    git remote add origin https://github.com/TheTom/llama-cpp-turboquant.git && \
    git fetch --depth 1 origin ${TURBOQUANT_REF} && \
    git checkout FETCH_HEAD

# -- gfx1010 patch: the D=512 flash-attn VEC kernel with V=turbo2/3/4 exceeds the
# per-thread local memory limit (65792 > 65536) and breaks the HIP build. Only
# that path is removed (head_dim 512 + K=q8_0 + V=turbo*). Models with head_dim
# 128/256 (e.g. Qwen3.x) do not use it. See patches/ for the equivalent diff.
RUN F=ggml/src/ggml-cuda/fattn.cu && \
    sed -i '/FATTN_VEC_CASE_D512(GGML_TYPE_Q8_0, GGML_TYPE_TURBO[234]_0)/d' $F && \
    sed -i 's/V->type == GGML_TYPE_BF16 || V->type == GGML_TYPE_TURBO2_0 ||/V->type == GGML_TYPE_BF16 ||/; s/V->type == GGML_TYPE_TURBO3_0 || V->type == GGML_TYPE_TURBO4_0;/false;/' $F && \
    sed -i '/DECL_FATTN_VEC_CASE_D512(GGML_TYPE_Q8_0, GGML_TYPE_TURBO[234]_0);/d' ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q8_0-turbo*.cu && \
    sed -i '/extern DECL_FATTN_VEC_CASE_D512(GGML_TYPE_Q8_0, GGML_TYPE_TURBO[234]_0);/d' ggml/src/ggml-cuda/fattn-vec.cuh && \
    ! grep -rn "D512(GGML_TYPE_Q8_0, GGML_TYPE_TURBO" ggml/src/ggml-cuda

RUN mkdir build && cd build && \
    HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake .. \
        -GNinja \
        -DGGML_HIP=ON \
        -DAMDGPU_TARGETS=gfx1010 \
        -DGPU_TARGETS=gfx1010 \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=OFF \
        -DGGML_AVX=OFF \
        -DGGML_AVX2=OFF \
        -DGGML_FMA=OFF \
        -DGGML_F16C=OFF \
        -DGGML_BMI2=OFF \
        -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DCMAKE_PREFIX_PATH="/opt/rocm;/opt/rocm/hip" \
        -DCMAKE_CXX_FLAGS="-march=x86-64 -mno-avx -mno-avx2 -mno-fma -mno-bmi2" \
        -DCMAKE_C_FLAGS="-march=x86-64 -mno-avx -mno-avx2 -mno-fma -mno-bmi2" && \
    ninja -k 0 -j$(nproc) && \
    ls -la bin/ && \
    test -x bin/llama-server && \
    (strip --strip-unneeded bin/llama-cli bin/llama-server 2>/dev/null || true)

# ── STAGE 2: runtime ───────────────────────────────────────────────────────
FROM ubuntu:22.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive

ARG ROCM_APT=6.3.3

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    wget gnupg2 ca-certificates python3 libnuma1 libgomp1 && \
    mkdir -p /etc/apt/keyrings && \
    wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key \
    | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg && \
    printf 'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/%s jammy main\n' "${ROCM_APT}" \
    > /etc/apt/sources.list.d/rocm.list && \
    printf 'Package: *\nPin: origin repo.radeon.com\nPin-Priority: 1000\n' \
    > /etc/apt/preferences.d/rocm-pin && \
    apt-get update && apt-get install -y --no-install-recommends \
    rocm-hip-runtime hipblaslt hipblas rocsparse && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# Own rocBLAS (no AVX, gfx1010 kernels): overrides the apt one
COPY --from=builder /opt/rocm/lib/librocblas.so* /opt/rocm/lib/
COPY --from=builder /opt/rocm/lib/rocblas /opt/rocm/lib/rocblas

# Binaries + every .so produced by the build (libllama*, libggml*, libmtmd...)
COPY --from=builder /app/build/bin/ /app/build/bin/

ENV HSA_OVERRIDE_GFX_VERSION=10.1.0
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64

EXPOSE 8080
ENTRYPOINT ["/app/build/bin/llama-server"]
