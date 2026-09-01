# Run DeepSeek-V4-Flash with Docker

[中文教程](../zh/DeepSeek-V4-Flash.md)

You do not need to install Python, compile KTransformers, or select a different
image for each GPU generation. The image detects the GPU and selects the
appropriate kernels automatically.

- [Running DeepSeek-V4-Flash with SGLang and KT-Kernel](#running-deepseek-v4-flash-with-sglang-and-kt-kernel)
  - [Table of Contents](#table-of-contents)
  - [Hardware Requirements](#hardware-requirements)
  - [Docker Quick Start](#docker-quick-start)
    - [Docker Runtime Configuration](#docker-runtime-configuration)
  - [Prerequisites](#prerequisites)
  - [Step 1: Download Model Weights](#step-1-download-model-weights)
  - [Step 2: Launch SGLang Server](#step-2-launch-sglang-server)
    - [Launch Command (Single RTX 5090 Example)](#launch-command-single-rtx-5090-example)
    - [Optional: Enable MTP (Multi-Token Prediction) Speculative Decoding](#optional-enable-mtp-multi-token-prediction-speculative-decoding)
  - [Step 3: Send Inference Requests](#step-3-send-inference-requests)
    - [Decode](#decode)
    - [Interactive Chat (kt chat)](#interactive-chat-kt-chat)

You need:

**Validated Configuration (this tutorial):**
- **GPU**: 1× NVIDIA RTX 5090 (32GB VRAM, SM_120)
- **CPU**: x86 CPU with AVX2 and FMA; AVX512/AMX improves throughput but is not required
- **RAM**: ≥200GB system memory
- **Storage**: ~340GB for model weights

**Supported consumer GPU architectures:**

| Arch | Compute Cap | MXFP4 MoE | NSA sparse MLA | Validated |
|------|------------|-----------|----------------|-----------|
| Consumer Blackwell (RTX 5090) | SM_120 | triton_kernels | Triton fallback | ✓ |
| Ada Lovelace (RTX 4090) | SM_89 | triton_kernels | Triton fallback | ✓ |
| Ampere (RTX 3090) | SM_86 | triton_kernels | Triton fallback | ✓ |

## Docker Quick Start

Use Docker when you want to run the prebuilt environment without cloning the repository or compiling from source. Install the NVIDIA driver, Docker, and NVIDIA Container Toolkit on the host first.

Pull the image from Docker Hub:

```bash
sudo docker pull approachingai/ktransformers:DSV4-specific
```

After downloading the model, enter the model directory and start the service:

```bash
cd /path/to/DeepSeek-V4-Flash-0731

sudo docker run --gpus all \
  --ipc host \
  --cap-add SYS_NICE \
  -p 30000:30000 \
  -v "$PWD":/model:ro \
  approachingai/ktransformers:DSV4-specific
```

The server listens on `http://localhost:30000` and exposes an OpenAI-compatible API. After the startup logs report readiness, verify it with:

```bash
curl http://localhost:30000/v1/models
```

### Docker Runtime Configuration

The image exposes the following environment variables. The launch command above uses these defaults:

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | all GPUs exposed by Docker | CUDA ordinals visible to the container; use with `TP`. |
| `TP` | `1` | SGLang tensor-parallel degree; it must not exceed the visible GPU count. |
| `MEM_FRACTION` | `0.90` | Fraction of GPU memory reserved by the server. |
| `CHUNKED_PREFILL_SIZE` | `4096` | Maximum token count in a prefill chunk. |
| `CONTEXT_LENGTH` | `16384` | Maximum model context length. |
| `MAX_RUNNING_REQUESTS` | `2` | Maximum concurrent running requests. |
| `KT_GPU_PREFILL_TOKEN_THRESHOLD` | `2048` | Layerwise GPU-prefill threshold. Set `0` to disable it. |
| `SWA_FULL_TOKENS_RATIO` | `0.4` | SWA KV-cache ratio sized for the default 4,096-token prefill chunk. |

Layerwise prefill is enabled by default for prompts of 2,048 tokens or longer.
Its slots are allocated lazily by the first qualifying request, so that request
can have a one-time setup cost. Set `KT_GPU_PREFILL_TOKEN_THRESHOLD=0` only when you intentionally want to disable layerwise prefill to reduce peak VRAM use.

For tensor parallelism, select the CUDA ordinals and set a matching degree. For
example, this starts two ranks on GPUs 0 and 1:

```bash
sudo docker run --gpus all \
  --ipc host \
  --cap-add SYS_NICE \
  -p 30000:30000 \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e TP=2 \
  -v "$PWD":/model:ro \
  approachingai/ktransformers:DSV4-specific
```


## Prerequisites

The remaining sections describe the native source installation path. Docker users can skip them.

1. **KT-Kernel installed**:
   ```bash
   git clone https://github.com/kvcache-ai/ktransformers.git
   cd ktransformers
   git submodule update --init --recursive
   cd kt-kernel && ./install.sh
   ```

2. **SGLang installed** (kvcache-ai fork):
   ```bash
   ./install.sh   # from ktransformers root
   ```

3. **CUDA 12.8+** and **flashinfer ≥ 0.6.9** (`flashinfer-python` and `flashinfer-cubin` must be the same version):
   ```bash
   pip install --upgrade flashinfer-python flashinfer-cubin
   ```
   This upgrade is required (even though `sglang-kt` pins `flashinfer_python==0.6.3`) because V4-Flash's MXFP4 MoE module imports `mxfp8_quantize`, `trtllm_fp4_block_scale_routed_moe`, etc., which only exist in flashinfer ≥ 0.6.9.

4. **transformers==4.57.1** (V4-Flash is incompatible with the 5.x series):
   ```bash
   pip install "transformers==4.57.1"
   ```
   `transformers` 5.x adds default-valued fields to `PretrainedConfig` that make `DeepSeekV4Config`'s dataclass declaration raise `TypeError: non-default argument 'quantization_config' follows default argument` at import time. `sglang-kt`'s pyproject does not pin `transformers`, so a fresh `pip install` will pull the latest 5.x and break server startup; pinning explicitly to `4.57.1` is required until the upstream fix lands.

5. **tilelang** (manual install — required for the NSA sparse-MLA tilelang indexer path used on non-Hopper GPUs):
   ```bash
   pip install tilelang "apache-tvm-ffi<0.1.12"
   ```
   `sglang-kt`'s pyproject does not declare `tilelang` as a dependency, so `pip install ./python[all]` will not pull it in. Validated with `tilelang==0.1.8`.

   > **Note:** Constrain `apache-tvm-ffi<0.1.12`. The standalone `apache-tvm-ffi` 0.1.12 wheel collides with the TVM FFI runtime bundled inside `tilelang`, so importing `tilelang` aborts with `TypeAttr __ffi_repr__ is already registered for type index 130` and the SGLang scheduler dies on startup. `apache-tvm-ffi==0.1.11` does not register the conflicting attribute and starts cleanly; pin until the upstream duplicate-registration fix lands.


## Step 1: Download Model Weights

Download the model from [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731):

```bash
mkdir -p /path/to/models
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local-dir /path/to/models/DeepSeek-V4-Flash-0731
```

## Step 2: Launch SGLang Server

### Launch Command (Single RTX 5090 Example)

```bash
export FLASHINFER_CUDA_ARCH_LIST=12.0a
export TORCH_CUDA_ARCH_LIST="12.0+PTX"

python -m sglang.launch_server \
  --host 0.0.0.0 --port 30000 \
  --model /path/to/models/DeepSeek-V4-Flash-0731 \
  --kt-weight-path /path/to/models/DeepSeek-V4-Flash-0731 \
  --kt-method MXFP4 \
  --kt-num-gpu-experts 10 \
  --kt-cpuinfer 60 \
  --kt-threadpool-count 2 \
  --kt-gpu-prefill-token-threshold 4096 \
  --kt-enable-dynamic-expert-update \
  --tensor-parallel-size 1 \
  --context-length 16384 \
  --attention-backend flashinfer \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 2048 \
  --max-prefill-tokens 2048 \
  --max-running-requests 2 \
  --watchdog-timeout 1200 \
  --disable-shared-experts-fusion \
  --trust-remote-code \
  --cuda-graph-bs 1 \
  --cuda-graph-max-bs 1 \
  --disable-radix-cache \
  --skip-server-warmup
```

Docker downloads the image, loads the model, and compiles the CUDA kernels for
the current GPU. The first start may take several minutes. As long as logs are
still appearing, do not start a second container.

`SYS_NICE` lets the CPU-expert workers bind memory to the correct NUMA nodes; it
does not grant the container full administrative access to the host.

The model is mounted read-only, so the container cannot modify the host model
files. The image uses stable CPU/GPU hybrid inference and enables lazy
layerwise prefill by default for requests of at least 2,048 tokens.

## Check that the server is ready

Keep the first terminal open and run this in a second terminal:

```bash
docker ps --filter name=ktransformers-dsv4
```

## Step 3: Send Inference Requests

### Decode

```bash
curl --fail --silent http://127.0.0.1:30000/health >/dev/null \
  && echo "DeepSeek-V4-Flash is ready"
```

If the status is still `health: starting`, the model is still loading. Continue
following the logs in the first terminal.

## Send your first request

```bash
curl -s http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Explain why the sky is blue in simple terms.",
    "sampling_params": {
      "temperature": 0.0,
      "max_new_tokens": 64
    }
  }'
```

The OpenAI-compatible API is also available at:

```text
http://127.0.0.1:30000/v1
```

See [KT-Kernel Parameters](https://github.com/kvcache-ai/ktransformers/tree/main/kt-kernel#kt-kernel-parameters) for the complete parameter reference.
