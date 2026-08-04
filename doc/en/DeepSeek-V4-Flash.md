# Run DeepSeek-V4-Flash with Docker

[中文教程](../zh/DeepSeek-V4-Flash.md)

You do not need to install Python, compile KTransformers, or select a different
image for each GPU generation. The image detects the GPU and selects the
appropriate kernels automatically.

## Before you start

You need:

- x86-64 Linux
- One NVIDIA GPU; RTX 5090 is validated and at least 32 GB VRAM is recommended
- A CPU with AVX512F
- At least 256 GiB of system memory
- About 150 GB for the model; keep at least 200 GB free
- Docker and NVIDIA Container Toolkit

Confirm that containers can access the GPU:

```bash
nvidia-smi
docker run --rm --device nvidia.com/gpu=0 \
  nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

If the second command cannot see the GPU, install or repair
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
before continuing.

## First start

Enter the downloaded model directory. Replace this path with your actual path:

```bash
cd /path/to/DeepSeek-V4-Flash
```

The directory should contain `config.json` and the `.safetensors` weights:

```bash
ls config.json *.safetensors | head
```

Copy the complete command below. No other value needs to be changed:

```bash
docker run --name ktransformers-dsv4 \
  --device nvidia.com/gpu=0 --ipc host -p 30000:30000 \
  --cap-add SYS_NICE \
  -v "$PWD":/model:ro \
  ghcr.io/kvcache-ai/ktransformers:dsv4-flash
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

When the status shows `healthy`, run:

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

## Stop and restart

Press `Ctrl+C` in the terminal showing the server logs to stop the container.

For later starts, do not repeat the full `docker run` command. Run:

```bash
docker start -a ktransformers-dsv4
```

The container retains its first-run JIT cache, so later starts are normally
faster.

To remove the container completely:

```bash
docker rm ktransformers-dsv4
```

Removing the container does not delete the host model or the downloaded Docker
image.

## Download the model

If you do not have the model yet, one option is the Hugging Face CLI:

```bash
python3 -m pip install -U huggingface_hub
hf download deepseek-ai/DeepSeek-V4-Flash \
  --local-dir "$HOME/models/DeepSeek-V4-Flash"
```

After it finishes:

```bash
cd "$HOME/models/DeepSeek-V4-Flash"
```

Then return to “First start” and copy the `docker run` command.

## Troubleshooting

### The container name already exists

You created the container previously. Start it again with:

```bash
docker start -a ktransformers-dsv4
```

To create it again from scratch, remove the stopped container first:

```bash
docker rm ktransformers-dsv4
```

### `/model/config.json` is missing

The start command was run outside the model directory. Check:

```bash
pwd
ls config.json
```

Then run the Docker command again.

### The container cannot see an NVIDIA GPU

Confirm that `nvidia-smi` works on the host, then repeat the Docker GPU check
from “Before you start.” NVIDIA Container Toolkit is normally missing or
misconfigured.

If Docker reports `unknown device nvidia.com/gpu=0`, NVIDIA CDI is not enabled
in that environment. Replace `--device nvidia.com/gpu=0` with `--gpus all` in
the start command. If that also fails, configure Docker by following the NVIDIA
Container Toolkit documentation.

### Port 30000 is already in use

After removing the old container, change the port mapping in the start command
to:

```bash
-p 30001:30000
```

The left side is the host port; the right side is the container's fixed service
port. Then access the server at `http://127.0.0.1:30001`.

### CUDA runs out of memory

By default, no routed experts are kept resident on the GPU. If you increased the
GPU expert count manually, remove the old container and lower it:

```bash
docker rm ktransformers-dsv4
```

Then add this before the image name in the start command:

```bash
-e KT_GPU_EXPERTS=4
```

If necessary, use `0` or reduce `CONTEXT_LENGTH` as well.

### Warnings appear in the log

The first start loads optional dependencies and compiles JIT kernels, so some
optional model components may print warnings. Use the container's `healthy`
status and the health endpoint—not the absence of warnings—to decide whether
startup succeeded.

## Advanced configuration

Most users do not need to change these values:

| Environment variable | Default | Purpose |
| --- | ---: | --- |
| `PORT` | `30000` | HTTP server port |
| `CONTEXT_LENGTH` | `16384` | Maximum context length |
| `MEM_FRACTION` | `0.90` | Static GPU-memory fraction |
| `KT_GPU_EXPERTS` | `0` | Number of GPU-resident experts; `0` keeps all routed experts on the CPU |
| `KT_CPUINFER_THREADS` | Auto | CPU inference threads |
| `KT_THREADPOOL_COUNT` | Auto | NUMA worker pools |
| `KT_GPU_PREFILL_TOKEN_THRESHOLD` | `2048` | Requests at or above this token count use layerwise prefill; set `0` to disable it |
| `CHUNKED_PREFILL_SIZE` | `4096` | Maximum number of tokens in one prefill scheduling round; must be a multiple of 256 |
| `MAX_PREFILL_TOKENS` | `4096` | Maximum number of prefill tokens admitted by the scheduler |
| `MAX_TOTAL_TOKENS` | automatic | Aggregate KV-token budget. Leave unset to use the profiled budget after layerwise-slot capacity is held out. |
| `SWA_FULL_TOKENS_RATIO` | `0.4` | Ratio of SWA KV tokens to full-attention KV tokens; sized for the default 4096-token prefill chunk |
| `ENABLE_MTP` | `0` | `1` experimentally enables MTP |

Add settings as `-e NAME=VALUE` before the image name. For example:

```bash
docker run --name ktransformers-dsv4 \
  --device nvidia.com/gpu=0 --ipc host -p 30000:30000 \
  --cap-add SYS_NICE \
  -v "$PWD":/model:ro \
  -e CONTEXT_LENGTH=8192 \
  ghcr.io/kvcache-ai/ktransformers:dsv4-flash
```

Layerwise prefill is enabled by default with
`KT_GPU_PREFILL_TOKEN_THRESHOLD=2048`. The image uses
`SWA_FULL_TOKENS_RATIO=0.4` so the SWA KV pool can serve the default
4096-token prefill chunk. Set `KT_GPU_PREFILL_TOKEN_THRESHOLD=0` to disable the
path. The KV-cache profiler holds out the capacity required for the two raw and
two prepared layerwise slots, but does not allocate their tensors at startup.
The first request that reaches the configured threshold allocates the slots and
therefore has a one-time initialization cost. If that allocation does not fit
in the currently available GPU memory, layerwise prefill is disabled for the
running server and the request continues with hybrid CPU/GPU inference.
