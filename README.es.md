# llama.cpp (turboquant) en AMD RDNA1 (gfx1010) + CPUs sin AVX — Docker

🇬🇧 [Read in English](README.md)

Recetas de Docker para correr [llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)
`tqp-v0.4.0` en **GPUs AMD RDNA1 (gfx1010: RX 5700 XT / RX 5600 XT)** con **ROCm 6.3.3**, en una
máquina cuyo **CPU no tiene AVX** (probado en un Celeron 3855U). Incluye una imagen de respaldo con Vulkan.

> **Estado:** la imagen ROCm compila, detecta las 3 GPUs y corre benchmarks (0.8B y 14B).
> El uso sostenido del server, los modelos de 27B y los tipos de KV cache turbo en esta imagen
> **todavía no están probados** (ver [Sin probar todavía](#sin-probar-todavía)). Tomalo como una base
> que funciona, no como un producto terminado.

## Hardware probado

| Ítem | Valor |
|---|---|
| GPUs | 1× RX 5700 XT (8 GB) + 2× RX 5600 XT (6 GB) — todas gfx1010, ~20 GB de VRAM en total |
| CPU | Celeron 3855U (2 hilos, **sin AVX/AVX2/FMA/BMI2**) |
| RAM | 7.7 GB |
| Host | Proxmox VE, Docker |
| Base de la imagen | Ubuntu 22.04 (jammy) + ROCm 6.3.3 |

## Por qué estas decisiones

- **Ubuntu 22.04 + ROCm 6.3.3.** ROCm 6.3.3 no tiene repo para Ubuntu 24.04, y hay usuarios que reportan
  regresiones con gfx1010 / `HSA_OVERRIDE_GFX_VERSION` en ROCm 6.4.x o superior (ver el repo
  [rccl-gfx1010](https://github.com/Marissccal/rccl-gfx1010) y
  [ROCm/rccl#2165](https://github.com/ROCm/rccl/issues/2165)). El Dockerfile tiene build args para volver a
  ROCm 6.3.0.
- **gfx1010 no tiene soporte oficial en ROCm.** El `rocBLAS` de apt no trae kernels para gfx1010, así que
  rocBLAS se **recompila desde el código fuente** con `-DAMDGPU_TARGETS=gfx1010`. En ejecución hace falta
  `HSA_OVERRIDE_GFX_VERSION=10.1.0`.
- **CPU sin AVX.** Todo lo que se compila para el host (rocBLAS y llama.cpp) tiene que usar
  `-DGGML_NATIVE=OFF` y `-march=x86-64 -mno-avx -mno-avx2 -mno-fma -mno-bmi2`; si no, el binario muere con
  `Illegal instruction`.
- **Un parche para gfx1010** (ver [`patches/`](patches/)). `tqp-v0.4.0` agrega un kernel de flash-attention
  solo para HIP con `head_dim = 512`, `K = q8_0` y `V = turbo2/3/4`. En gfx1010 necesita 65792 bytes de
  memoria local y el límite es 65536, así que la compilación falla:

  ```
  fattn-vec.cuh:23:24: error: local memory (65792) exceeds limit (65536)
  in 'void flash_attn_ext_vec<512, 1, q8_0, turbo3_0 ...>'  (compiling for gfx1010)
  ```

  El Dockerfile saca solo ese camino. Los modelos con `head_dim` 128/256 (Qwen3.x, etc.) no lo usan.
  Costo: los modelos con `head_dim` 512 que usen `K=q8_0` + `V` turbo pierden ese camino rápido.

## Compilar una vez, correr en otro lado

La imagen ROCm tarda aproximadamente **1.5–2 horas** en compilarse desde cero (rocBLAS + kernels HIP de
llama.cpp), así que el flujo es: compilar en una máquina potente, exportar y cargar en la máquina destino.

**1. Compilar y exportar (máquina de build)**

```bash
docker buildx build --platform linux/amd64 --progress=plain \
  -t llama-tqp040:rocm633 -f Dockerfile \
  --output type=docker,dest=- . 2> build.log | gzip > llama-tqp040-rocm633.tar.gz
grep -nE "FAILED|error:|ERROR:" build.log | head
ls -lh llama-tqp040-rocm633.tar.gz        # se esperan ~3.2 GB, NO 20 bytes
```

Por qué se exporta así: un `docker build` normal además desempaqueta la imagen terminada en el store local
de Docker (`/var/lib/containerd` o `/var/lib/docker`, normalmente en el filesystem raíz). Ese paso fue
donde nos apareció `no space left on device` varias veces. `--output type=docker,dest=-` manda un tarball
por stdout, y el archivo final queda donde vos le indiques. Si tu buildx rechaza `type=docker`, usá
`docker build -t ... . && docker save ... | gzip > archivo.tar.gz`.

**2. Cargar (máquina destino)**

```bash
gunzip -c llama-tqp040-rocm633.tar.gz | docker load
docker images | grep tqp040
```

**3. Verificar la imagen (no te lo saltees)**

```bash
# binarios presentes, sin librerías faltantes (deben salir solo los 4 nombres y ningún "not found")
docker run --rm --entrypoint /bin/bash llama-tqp040:rocm633 -c \
  "ls /app/build/bin | grep -E '^llama-(server|cli|bench)$|libggml-hip'; \
   ldd /app/build/bin/llama-server | grep 'not found'"

# GPUs visibles
docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
  --entrypoint /app/build/bin/llama-server llama-tqp040:rocm633 --list-devices
```

Resultado esperado:

```
Available devices:
  ROCm0: AMD Radeon RX 5600 XT (6128 MiB, 6108 MiB free)
  ROCm1: AMD Radeon RX 5600 XT (6128 MiB, 6108 MiB free)
  ROCm2: AMD Radeon RX 5700 XT (8176 MiB, 8156 MiB free)
```

## Ejecutar

```bash
# benchmark, todas las GPUs (el reparto por capas es el modo por defecto)
docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
  -v /ruta/a/modelos:/models --entrypoint /app/build/bin/llama-bench \
  llama-tqp040:rocm633 -m /models/modelo.gguf -ngl 99

# server (offload completo a GPU: -ngl 99, y un contexto explícito y moderado)
docker run -d --name tqp040 --device /dev/kfd --device /dev/dri --group-add video \
  -p 8080:8080 -v /ruta/a/modelos:/models llama-tqp040:rocm633 \
  -m /models/modelo.gguf -ngl 99 -c 4096 --host 0.0.0.0 --port 8080
docker logs -f tqp040
```

Con `-e HIP_VISIBLE_DEVICES=2` se fija una sola GPU (acá, el índice 2 = la 5700 XT).

## Resultados de las pruebas

Todas con `-ngl 99`, salvo el bench del 0.8B (`-ngl -1`, por defecto).

| Prueba | Imagen | Resultado |
|---|---|---|
| Qwen3.5-0.8B Q8_0, solo RX 5700 XT | imagen ROCm 6.3 anterior (llama.cpp `781dbc5ac`) | pp512 **2033.9 ± 3.3** t/s, tg128 **88.5 ± 0.2** t/s |
| Qwen3.5-0.8B Q8_0, solo RX 5700 XT | este repo (`tqp-v0.4.0`, `bcb85fc`) | pp512 **1715.9 ± 88.8** t/s, tg128 **99.9 ± 0.8** t/s |
| 14B Q4_K_M (8.52 GiB), 3 GPUs, reparto por capas | imagen ROCm 6.3.0 anterior (llama.cpp `781dbc5ac`) | pp512 **59.0** t/s, tg128 **17.6** t/s |
| 14B Q4_K_M (8.52 GiB), 3 GPUs, reparto por capas | este repo | pp512 **53.7 ± 6.7** t/s, tg128 **18.5 ± 0.7** t/s — terminó limpio, sin cuelgues ni crashes |
| 14B IQ4_XS, 3 GPUs, `-c 4096 --no-mmap`, `llama-server` | este repo | el modelo cargó en **~86 s** (sin `--no-mmap` nunca terminaba de cargar) |
| 27B cuantización UD (~16 GB), 3 GPUs | imagen Vulkan (RADV) | prefill ~9.2 t/s, generación ~7 t/s; carga del modelo ~17 min |

Notas: en la misma prueba de 14B con 3 GPUs, la imagen nueva da aprox. −9% en prefill y +5% en generación
frente a la anterior. Los números del 0.8B son ruidosos y no representan a modelos más grandes (generación +13%,
prefill −16% frente a la imagen anterior). La corrida del 14B es la importante: en esta máquina, builds
anteriores de ROCm 6.3.x cargaban modelos grandes pero fallaban tras un rato de uso, así que una corrida
multi-GPU limpia es buena señal — pero un bench no es una prueba de estrés.

## Problemas frecuentes

| Síntoma | Causa / solución |
|---|---|
| `Illegal instruction` al arrancar | Algo se compiló para la CPU del host. Revisá `-DGGML_NATIVE=OFF` y `-mno-avx -mno-avx2 -mno-fma -mno-bmi2` en cada llamada a cmake. |
| `invalid device function` | Faltan kernels gfx1010 → rocBLAS no se compiló desde fuente para `gfx1010`, o falta `HSA_OVERRIDE_GFX_VERSION=10.1.0`. |
| `cannot open shared object file` (p. ej. `libllama-cli-impl.so`) | Hay que copiar **todo** `build/bin/` a la etapa runtime, no solo los ejecutables (el Dockerfile ya lo hace). |
| `exec: "/app/build/bin/llama-server": no such file` aunque el build "terminó bien" | Un `ninja` fallido quedó tapado por `ninja && strip \|\| true`. El Dockerfile ahora corre `test -x bin/llama-server` para que el build falle a la vista. Si en `bin/` solo ves `libggml-base/cpu`, fue esto. |
| `local memory (65792) exceeds limit (65536)` | El kernel flash-attn turbo D=512 en gfx1010. Ya está parchado en el Dockerfile. |
| `no space left on device` al compilar/exportar | El cache de BuildKit y el store de imágenes viven en el filesystem raíz, y cada intento fallido deja capas de varios GB. `docker builder prune --keep-storage 25GB -f` conserva las capas recientes; `docker builder prune -af` borra todo. Mantené **25 GB o más libres**; 40 GB o más para un build desde cero. |
| `Could not find a version that satisfies the requirement PyYAML (from amdsmi)` | Inofensivo en este build (el pip de la post-instalación de amdsmi no llega a PyPI). El Dockerfile instala `python3-yaml` desde el principio, así pip lo ve como ya satisfecho. |
| La carga tarda muchos minutos, o parece trabada en `load_model: loading model ...`, solo con modelos más grandes que tu RAM libre | **Thrashing del page cache por mmap.** Por defecto llama.cpp hace `mmap` del GGUF; si el archivo no entra en RAM (acá teníamos 7.7 GB) el kernel desaloja y vuelve a leer páginas de un disco lento. **Solución: `--no-mmap`** (en builds nuevos está deprecado a favor de `--load-mode`; mirá `--help`). Medido acá: Qwen3.8-14B IQ4_XS pasó de "nunca terminaba" a **~86 s** de carga en 3 GPUs. Si sigue lento, medí el disco: `echo 3 > /proc/sys/vm/drop_caches; dd if=modelo.gguf of=/dev/null bs=16M status=progress` (~15–30 MB/s → usar SSD/NVMe) y cerrá monitores como `radeontop`. Para ver el avance: `watch -n2 'for c in /sys/class/drm/card*/device/mem_info_vram_used; do echo $c $(( $(cat $c)/1048576 )) MiB; done'` (la VRAM tiene que seguir creciendo). |
| Falta la UI del server | El build descarga los assets de la UI desde Hugging Face; el log debe decir `UI: archive verified and extracted`. |

## Alternativa con Vulkan

[`Dockerfile.vulkan`](Dockerfile.vulkan) compila llama.cpp oficial con Vulkan (Mesa RADV). Sin ROCm, sin
override de HSA, con los mismos flags sin AVX. Funciona con las tres GPUs y es la opción más segura si
ROCm se porta mal con modelos grandes, a menor velocidad. Usá `-dev Vulkan1,Vulkan2,Vulkan3` para excluir
una iGPU Intel (`Vulkan0`). Con `-ngl` manual, `--fit` aborta con un mensaje informativo (no es fatal).

## Sin probar todavía

- **27B en ROCm: problema conocido en la imagen anterior.** Con la imagen ROCm 6.3.0 anterior (llama.cpp
  `781dbc5ac`), los modelos de 27B (UD-IQ4_XS y UD-Q4_K_M) cargaban en VRAM (67–78% por GPU) pero
  `llama-bench` se colgaba sin salida durante más de 18 min, con el graphics pipe de la GPU en reposo y sin OOM
  en `dmesg`. **No se volvió a probar con esta imagen.**
- Uso sostenido de `llama-server` (muchas consultas a lo largo del tiempo) con esta imagen.
- Modelos de 27B / 30B-A3B en **ROCm** con `tqp-v0.4.0`.
- Los tipos de KV cache turbo (`-ctk` / `-ctv turbo2|3|4`, `--kv-stream-arena-mib`) en gfx1010.
- Si los cambios HIP de `tqp-v0.4.0` son una mejora neta en gfx1010 (el bench del 0.8B da resultado mixto).
- La variante ROCm 6.3.0 del Dockerfile (los build args existen, pero no se recompiló con ellos).

## Créditos

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) (`tqp-v0.4.0`)
- ROCm / rocBLAS de AMD; investigación de la comunidad sobre gfx1010: [Marissccal/rccl-gfx1010](https://github.com/Marissccal/rccl-gfx1010), [Efenstor/PyTorch-ROCm-gfx1010](https://github.com/Efenstor/PyTorch-ROCm-gfx1010).

> Todavía no incluye archivo de licencia — agregá la que prefieras.
