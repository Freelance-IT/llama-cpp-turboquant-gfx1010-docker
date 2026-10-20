# llama.cpp (turboquant) on AMD RDNA1 (gfx1010) + CPUs without AVX — Docker

🇪🇸 [Leer en español](README.es.md)

Docker recipes to run [llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)
`tqp-v0.4.0` on **AMD RDNA1 GPUs (gfx1010: RX 5700 XT / RX 5600 XT)** with **ROCm 6.3.3**, on a
machine whose **CPU has no AVX** (tested on a Celeron 3855U). A Vulkan fallback image is included.

> **Status:** the ROCm image builds, detects the 3 GPUs and runs benchmarks (0.8B and 14B).
> Sustained server use, 27B models and the turbo KV cache types on this image are **not tested yet**
> (see [Not tested yet](#not-tested-yet)). Treat this as a working base, not a finished product.

## Tested hardware

| Item | Value |
|---|---|
| GPUs | 1× RX 5700 XT (8 GB) + 2× RX 5600 XT (6 GB) — all gfx1010, ~20 GB VRAM total |
| CPU | Celeron 3855U (2 threads, **no AVX/AVX2/FMA/BMI2**) |
| RAM | 7.7 GB |
| Host | Proxmox VE, Docker |
| Image base | Ubuntu 22.04 (jammy) + ROCm 6.3.3 |

## Why these choices

- **Ubuntu 22.04 + ROCm 6.3.3.** ROCm 6.3.3 has no Ubuntu 24.04 repo, and users report regressions
  for gfx1010 / `HSA_OVERRIDE_GFX_VERSION` on ROCm 6.4.x+ (see the
  [rccl-gfx1010](https://github.com/Marissccal/rccl-gfx1010) repo and
  [ROCm/rccl#2165](https://github.com/ROCm/rccl/issues/2165)). The Dockerfile has build args to fall back to
  ROCm 6.3.0.
- **gfx1010 is not officially supported by ROCm.** The stock `rocBLAS` from apt has no gfx1010
  kernels, so rocBLAS is **rebuilt from source** with `-DAMDGPU_TARGETS=gfx1010`. At runtime
  `HSA_OVERRIDE_GFX_VERSION=10.1.0` is required.
- **No AVX on the CPU.** Everything that is compiled for the host (rocBLAS and llama.cpp) must be built
  with `-DGGML_NATIVE=OFF` and `-march=x86-64 -mno-avx -mno-avx2 -mno-fma -mno-bmi2`, otherwise the
  binary dies with `Illegal instruction`.
- **A patch for gfx1010** (see [`patches/`](patches/)). `tqp-v0.4.0` adds a HIP-only flash-attention
  kernel for `head_dim = 512` with `K = q8_0` and `V = turbo2/3/4`. On gfx1010 it needs 65792 bytes of
  local memory and the limit is 65536, so the build fails:

  ```
  fattn-vec.cuh:23:24: error: local memory (65792) exceeds limit (65536)
  in 'void flash_attn_ext_vec<512, 1, q8_0, turbo3_0 ...>'  (compiling for gfx1010)
  ```

  The Dockerfile removes only that path. Models with `head_dim` 128/256 (Qwen3.x etc.) don't use it.
  Trade-off: models with `head_dim` 512 that use `K=q8_0` + turbo `V` lose that fast path.

## Build once, run elsewhere

The ROCm image takes roughly **1.5–2 hours** to build cold (rocBLAS + llama.cpp HIP kernels), so the
workflow is: build on a powerful machine, export, load on the target.

**1. Build and export (build machine)**

```bash
docker buildx build --platform linux/amd64 --progress=plain \
  -t llama-tqp040:rocm633 -f Dockerfile \
  --output type=docker,dest=- . 2> build.log | gzip > llama-tqp040-rocm633.tar.gz
grep -nE "FAILED|error:|ERROR:" build.log | head
ls -lh llama-tqp040-rocm633.tar.gz        # ~3.2 GB expected, NOT 20 bytes
```

Why export like this: a normal `docker build` also unpacks the finished image into Docker's local store
(`/var/lib/containerd` or `/var/lib/docker`, usually on the root filesystem). That step is where we hit
`no space left on device` several times. `--output type=docker,dest=-` streams a tarball to stdout
instead, so the final file lands wherever you point it. If your buildx rejects `type=docker`, use
`docker build -t ... . && docker save ... | gzip > file.tar.gz`.

**2. Load (target machine)**

```bash
gunzip -c llama-tqp040-rocm633.tar.gz | docker load
docker images | grep tqp040
```

**3. Verify the image (don't skip this)**

```bash
# binaries present, no missing libs (should print only the 4 names and no "not found")
docker run --rm --entrypoint /bin/bash llama-tqp040:rocm633 -c \
  "ls /app/build/bin | grep -E '^llama-(server|cli|bench)$|libggml-hip'; \
   ldd /app/build/bin/llama-server | grep 'not found'"

# GPUs visible
docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
  --entrypoint /app/build/bin/llama-server llama-tqp040:rocm633 --list-devices
```

Expected:

```
Available devices:
  ROCm0: AMD Radeon RX 5600 XT (6128 MiB, 6108 MiB free)
  ROCm1: AMD Radeon RX 5600 XT (6128 MiB, 6108 MiB free)
  ROCm2: AMD Radeon RX 5700 XT (8176 MiB, 8156 MiB free)
```

## Run

```bash
# benchmark, all GPUs (layer split is the default)
docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
  -v /path/to/models:/models --entrypoint /app/build/bin/llama-bench \
  llama-tqp040:rocm633 -m /models/model.gguf -ngl 99

# server (keep full GPU offload: -ngl 99, and set an explicit modest context)
docker run -d --name tqp040 --device /dev/kfd --device /dev/dri --group-add video \
  -p 8080:8080 -v /path/to/models:/models llama-tqp040:rocm633 \
  -m /models/model.gguf -ngl 99 -c 4096 --host 0.0.0.0 --port 8080
docker logs -f tqp040
```

Use `HIP_VISIBLE_DEVICES=2` (`-e`) to pin a single GPU (here, index 2 = the 5700 XT).

## Test results

All runs used `-ngl 99` unless noted (`-ngl -1` default in the 0.8B bench).

| Test | Image | Result |
|---|---|---|
| Qwen3.5-0.8B Q8_0, RX 5700 XT only | earlier ROCm 6.3 image (llama.cpp `781dbc5ac`) | pp512 **2033.9 ± 3.3** t/s, tg128 **88.5 ± 0.2** t/s |
| Qwen3.5-0.8B Q8_0, RX 5700 XT only | this repo (`tqp-v0.4.0`, `bcb85fc`) | pp512 **1715.9 ± 88.8** t/s, tg128 **99.9 ± 0.8** t/s |
| 14B Q4_K_M (8.52 GiB), 3 GPUs, layer split | earlier ROCm 6.3.0 image (llama.cpp `781dbc5ac`) | pp512 **59.0** t/s, tg128 **17.6** t/s |
| 14B Q4_K_M (8.52 GiB), 3 GPUs, layer split | this repo | pp512 **53.7 ± 6.7** t/s, tg128 **18.5 ± 0.7** t/s — completed cleanly, no hang/crash |
| 27B UD quant (~16 GB), 3 GPUs | Vulkan image (RADV) | prefill ~9.2 t/s, generation ~7 t/s; model load ~17 min |

Notes: on the same 14B / 3-GPU test the new image is about −9% in prefill and +5% in generation vs. the
earlier image. The 0.8B numbers are noisy and not representative of larger models (generation +13%,
prefill −16% vs. the older image). The 14B run is the important one: older ROCm 6.3.x builds on this rig
loaded larger models but failed after some use, so a clean multi-GPU run is a good sign — but one bench
is not a stress test.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Illegal instruction` at start | Something was compiled for the host CPU. Check `-DGGML_NATIVE=OFF` and `-mno-avx -mno-avx2 -mno-fma -mno-bmi2` on every cmake call. |
| `invalid device function` | Missing gfx1010 kernels → rocBLAS wasn't built from source for `gfx1010`, or `HSA_OVERRIDE_GFX_VERSION=10.1.0` is not set. |
| `cannot open shared object file` (e.g. `libllama-cli-impl.so`) | Copy the **whole** `build/bin/` to the runtime stage, not only the executables (the Dockerfile does this). |
| `exec: "/app/build/bin/llama-server": no such file` but the build "succeeded" | A failed `ninja` was swallowed by `ninja && strip \|\| true`. The Dockerfile now runs `test -x bin/llama-server` so the build fails loudly. If you see only `libggml-base/cpu` in `bin/`, this is what happened. |
| `local memory (65792) exceeds limit (65536)` | The D=512 turbo flash-attn kernel on gfx1010. Already patched in the Dockerfile. |
| `no space left on device` during build/export | BuildKit cache + image store live on the root filesystem and every failed attempt leaves multi-GB layers. `docker builder prune --keep-storage 25GB -f` keeps recent layers; `docker builder prune -af` wipes all. Keep **25 GB+ free**; 40 GB+ for a cold build. |
| `Could not find a version that satisfies the requirement PyYAML (from amdsmi)` | Harmless in this build (amdsmi's post-install pip can't reach PyPI). `python3-yaml` is installed up front in the Dockerfile so pip sees it as already satisfied. |
| Very slow model load (minutes) | Usually disk throughput, not swap. Test: `echo 3 > /proc/sys/vm/drop_caches; dd if=model.gguf of=/dev/null bs=16M status=progress`. ~15–30 MB/s means the disk is the bottleneck → use SSD/NVMe. Also close monitors like `radeontop` while loading on a 2-thread CPU. |
| Server UI missing | The build downloads the web UI assets from Hugging Face; the log should say `UI: archive verified and extracted`. |

## Vulkan fallback

[`Dockerfile.vulkan`](Dockerfile.vulkan) builds upstream llama.cpp with Vulkan (Mesa RADV). No ROCm, no
HSA override, same no-AVX flags. It works on all three GPUs and is the safer choice if ROCm
misbehaves with big models, at a lower speed. Use `-dev Vulkan1,Vulkan2,Vulkan3` to exclude an Intel iGPU
(`Vulkan0`). With `-ngl` set manually, `--fit` aborts with an informational message (not fatal).

## Not tested yet

- **27B on ROCm: known problem on the earlier image.** With the earlier ROCm 6.3.0 image (llama.cpp
  `781dbc5ac`), 27B models (UD-IQ4_XS and UD-Q4_K_M) loaded into VRAM (67–78% per GPU) but `llama-bench`
  hung with no output for 18+ min, GPU graphics pipe idle, no OOM in `dmesg`. **Not re-tested with this image.**
- Sustained use of `llama-server` (many requests over time) with this image.
- 27B / 30B-A3B models on **ROCm** with `tqp-v0.4.0`.
- The turbo KV cache types (`-ctk` / `-ctv turbo2|3|4`, `--kv-stream-arena-mib`) on gfx1010.
- Whether the `tqp-v0.4.0` HIP changes are a net win on gfx1010 (the 0.8B bench says "mixed").
- The ROCm 6.3.0 variant of the Dockerfile (build args exist, it was not rebuilt with them here).

## Credits

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) (`tqp-v0.4.0`)
- ROCm / rocBLAS by AMD; gfx1010 community research: [Marissccal/rccl-gfx1010](https://github.com/Marissccal/rccl-gfx1010), [Efenstor/PyTorch-ROCm-gfx1010](https://github.com/Efenstor/PyTorch-ROCm-gfx1010).

> No license file is included yet — add the one you prefer.
