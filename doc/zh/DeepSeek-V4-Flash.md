# 用 Docker 运行 DeepSeek-V4-Flash

[English](../en/DeepSeek-V4-Flash.md)

你不需要安装 Python、编译 KTransformers，也不需要为不同显卡选择不同镜像。
Docker 镜像会自动识别显卡并选择合适的内核。

本教程使用已发布的镜像 `approachingai/ktransformers:DSV4-specific`。
如需可复现部署，请优先使用发布说明中提供的不可变版本标签。

## 开始之前

你需要准备：

- x86-64 Linux
- 一张 NVIDIA GPU；已验证 RTX 5090，推荐至少 32 GB 显存
- 支持 AVX2 与 FMA 的 x86-64 CPU；如具备 AVX512/AMX 会自动使用更快内核，但并非必需
- 建议 256 GiB 系统内存；较小内存机器可降低上下文和并发后尝试，但可能发生主机内存不足
- 约 150 GB 模型空间，建议预留 200 GB
- Docker 和 NVIDIA Container Toolkit

先确认宿主机可以使用 GPU：

```bash
nvidia-smi
docker run --rm --device nvidia.com/gpu=0 \
  nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

如果第二条命令看不到显卡，请先安装或修复
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。

## 第一次启动

首先进入模型目录。请把下面的路径替换成你的实际路径：

```bash
cd /你的/DeepSeek-V4-Flash
```

目录中应当能看到 `config.json` 和 `.safetensors` 权重文件：

```bash
ls config.json *.safetensors | head
```

然后完整复制下面的命令，不需要修改其他参数：

```bash
docker run --name ktransformers-dsv4 \
  --device nvidia.com/gpu=0 --ipc host -p 30000:30000 \
  --cap-add SYS_NICE \
  -v "$PWD":/model:ro \
  approachingai/ktransformers:DSV4-specific
```

Docker 会自动下载镜像、加载模型并编译当前显卡需要的 CUDA 内核。第一次启动可能
需要几分钟；只要终端仍在输出日志，就不要重复启动容器。

`SYS_NICE` 只用于让 CPU expert 线程正确绑定 NUMA 内存；它不会让容器获得完整的
宿主机管理权限。

模型以只读方式挂载，容器不会修改宿主机上的模型文件。镜像默认使用稳定的
CPU/GPU 混合推理，并对输入长度不低于 2,048 tokens 的请求启用懒分配的
layerwise prefill。

## 使用 tensor parallel 多卡运行

上面的单卡命令使用 `TP=1`。如需使用两张卡，显式指定容器内可见的 CUDA
序号，并设置 tensor parallel 度：

```bash
docker run --name ktransformers-dsv4-tp2 \
  --device nvidia.com/gpu=all --ipc host -p 30000:30000 \
  --cap-add SYS_NICE \
  -e CUDA_VISIBLE_DEVICES=0,1 -e TP=2 \
  -v "$PWD":/model:ro \
  approachingai/ktransformers:DSV4-specific
```

`CUDA_VISIBLE_DEVICES` 决定容器内可见的显卡；`TP` 会传给 SGLang 的
`--tensor-parallel-size`。通常按“一张卡一个 rank”使用时，应让可见卡数量与
`TP` 相等。镜像会在启动前检查 PyTorch 至少能看到 `TP` 张卡，因此配置不匹配
会直接报错，而不会悄悄退回单卡。

使用 Docker Compose 时，等价的便捷变量为：

```bash
GPU_DEVICE=0,1 TP=2 \
  IMAGE_REPOSITORY=approachingai/ktransformers IMAGE_TAG=DSV4-specific \
  docker compose -f docker/compose.dsv4.yml up -d --no-build
```

`GPU_DEVICE` 是 Compose 的输入变量，容器内会映射为 `CUDA_VISIBLE_DEVICES`。
启动后可检查日志确认 TP 语义：

```bash
docker logs ktransformers-dsv4-tp2 2>&1 | \
  grep -E 'visible_gpus=|tensor-parallel-size'
```

日志应显示选中的 GPU 数、`TP=2` 和 `--tensor-parallel-size 2`。

## 默认启用的 layerwise prefill

layerwise prefill 默认已经开启，不需要额外增加 feature flag。默认配置为：

- `KT_GPU_PREFILL_TOKEN_THRESHOLD=2048`：输入达到该 token 阈值的请求使用
  layerwise prefill。
- `CHUNKED_PREFILL_SIZE=4096` 与 `MAX_PREFILL_TOKENS=4096`：每轮 prefill
  调度最多处理 4,096 tokens。
- `SWA_FULL_TOKENS_RATIO=0.4`：为默认 chunk 大小预留足够的 SWA KV-cache 页面。

layerwise slot 会在第一个满足阈值的请求到来时才分配，因此该请求可能有一次性
初始化开销。`KT_GPU_EXPERTS=0` **不会**关闭 layerwise prefill；它只表示 routed
experts 不常驻 GPU。只有在希望降低峰值显存、刻意关闭该路径时，才设置
`KT_GPU_PREFILL_TOKEN_THRESHOLD=0`。

## 检查是否启动成功

保持第一个终端运行，打开第二个终端执行：

```bash
docker ps --filter name=ktransformers-dsv4
```

当状态显示 `healthy` 后，再执行：

```bash
curl --fail --silent http://127.0.0.1:30000/health_generate >/dev/null \
  && echo "DeepSeek-V4-Flash 已就绪"
```

如果仍显示 `health: starting`，模型还在加载。回到第一个终端继续观察日志即可。

## 发送第一条请求

```bash
curl -s http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "请用简单的语言解释天空为什么是蓝色的。",
    "sampling_params": {
      "temperature": 0.0,
      "max_new_tokens": 64
    }
  }'
```

服务还提供 OpenAI 兼容接口：

```text
http://127.0.0.1:30000/v1
```

## 停止和重新启动

在显示服务器日志的终端按 `Ctrl+C` 即可停止。

以后不需要重新执行完整的 `docker run`，直接运行：

```bash
docker start -a ktransformers-dsv4
```

容器会保留第一次运行产生的 JIT 缓存，所以重新启动通常更快。

彻底删除容器：

```bash
docker rm ktransformers-dsv4
```

删除容器不会删除宿主机上的模型，也不会删除已经下载的 Docker 镜像。

## 下载模型

如果还没有模型，可以使用 Hugging Face CLI：

```bash
python3 -m pip install -U huggingface_hub
hf download deepseek-ai/DeepSeek-V4-Flash \
  --local-dir "$HOME/models/DeepSeek-V4-Flash"
```

下载完成后：

```bash
cd "$HOME/models/DeepSeek-V4-Flash"
```

然后返回“第一次启动”，复制 `docker run` 命令。

## 常见问题

### 容器名称已经存在

说明你以前创建过容器。直接重新启动：

```bash
docker start -a ktransformers-dsv4
```

如果希望重新创建，先删除旧容器：

```bash
docker rm ktransformers-dsv4
```

### 提示找不到 `/model/config.json`

执行 `docker run` 前没有进入正确的模型目录。确认：

```bash
pwd
ls config.json
```

然后重新执行启动命令。

### 容器看不到 NVIDIA GPU

先确认宿主机上的 `nvidia-smi` 正常，再重复“开始之前”的 Docker GPU 检查。通常是
NVIDIA Container Toolkit 未安装或配置不正确。

如果 Docker 提示 `unknown device nvidia.com/gpu=0`，说明当前环境没有启用 NVIDIA
CDI。可以先把启动命令中的 `--device nvidia.com/gpu=0` 替换为 `--gpus all`；如果仍然
失败，请按照 NVIDIA Container Toolkit 文档配置 Docker。

### 30000 端口被占用

删除旧容器后，把启动命令中的端口映射改为：

```bash
-p 30001:30000
```

左边是宿主机端口，右边是容器内固定的服务端口。之后使用
`http://127.0.0.1:30001` 访问服务。

### CUDA 显存不足

默认不在 GPU 上常驻 routed experts。若你手动提高过 GPU experts，可删除旧容器后降低该值：

```bash
docker rm ktransformers-dsv4
```

在启动命令的镜像名称前增加：

```bash
-e KT_GPU_EXPERTS=4
```

如果仍然不足，可以改为 `0`，或进一步降低 `CONTEXT_LENGTH`。

### CPU 没有 AVX512

AVX512 不是硬性要求。镜像内已包含 AVX2 MXFP4 后端，在支持 AVX2 与 FMA 的
x86-64 CPU 上会自动选择它。该路径吞吐会低于 AVX512/AMX，但不需要换镜像或额外设置启动参数。

### 主机内存少于 256 GiB

256 GiB 是默认配置的建议容量，而不是启动门槛。较小内存机器仍可启动；如果主机内存不足，
请降低 `CONTEXT_LENGTH`、`MAX_RUNNING_REQUESTS`，并在显存允许范围内调整 `KT_GPU_EXPERTS`。

### 日志中出现 warning

第一次启动会加载依赖并编译 JIT 内核，部分可选模型组件可能输出 warning。判断服务
是否成功应以容器显示 `healthy` 和健康检查通过为准。

## 高级配置

普通用户不需要修改这些参数：

| 环境变量 | 默认值 | 作用 |
| --- | ---: | --- |
| `GPU_DEVICE`（Compose） | `0` | 一个 CUDA GPU 序号，或如 `0,1` 的逗号分隔列表 |
| `CUDA_VISIBLE_DEVICES`（`docker run`） | 所有已暴露 GPU | 选择容器内可见的 CUDA 序号；与 `TP` 配合使用 |
| `TP` | `1` | tensor parallel 度；不得超过选中的可见 GPU 数量 |
| `PORT` | `30000` | HTTP 服务端口 |
| `CONTEXT_LENGTH` | `16384` | 最大上下文长度 |
| `MEM_FRACTION` | `0.90` | 静态显存使用比例 |
| `KT_GPU_EXPERTS` | `0` | 常驻 GPU 的 experts 数量；`0` 表示全部由 CPU experts 处理 |
| `KT_CPUINFER_THREADS` | 自动 | CPU inference 线程数 |
| `KT_THREADPOOL_COUNT` | 自动 | NUMA 线程池数量 |
| `KT_GPU_PREFILL_TOKEN_THRESHOLD` | `2048` | 默认启用 layerwise prefill；输入达到该 token 数时使用，设为 `0` 可关闭 |
| `CHUNKED_PREFILL_SIZE` | `4096` | 每轮 prefill 的最大 token 数；必须是 256 的倍数 |
| `MAX_PREFILL_TOKENS` | `4096` | 调度器允许的最大 prefill token 数 |
| `MAX_RUNNING_REQUESTS` | `2` | 默认 layerwise 安全配置下的最大并发运行请求数 |
| `MAX_TOTAL_TOKENS` | 自动 | 聚合 KV token 预算。留空时，SGLang 会在扣除 layerwise slot 容量后自动计算。 |
| `SWA_FULL_TOKENS_RATIO` | `0.4` | SWA KV token 与 full-attention KV token 的比例，适配默认 4096-token prefill chunk。 |
| `ENABLE_MTP` | `0` | `1` 表示实验性启用 MTP |

使用 `-e 名称=值` 将参数加在镜像名称之前。例如：

```bash
docker run --name ktransformers-dsv4 \
  --device nvidia.com/gpu=0 --ipc host -p 30000:30000 \
  --cap-add SYS_NICE \
  -v "$PWD":/model:ro \
  -e CONTEXT_LENGTH=8192 \
  approachingai/ktransformers:DSV4-specific
```

使用 Compose 开启 tensor parallel 时，GPU 列表和 TP 必须匹配：

```bash
GPU_DEVICE=0,1 TP=2 \
  IMAGE_REPOSITORY=approachingai/ktransformers IMAGE_TAG=DSV4-specific \
  docker compose -f docker/compose.dsv4.yml up -d --no-build
```

启动前容器会检查 PyTorch 至少能看到 `TP` 张 GPU。
直接使用 `docker run` 时，等价配置如下：

```bash
docker run --device nvidia.com/gpu=all \
  -e CUDA_VISIBLE_DEVICES=0,1 -e TP=2 \
  --ipc host -p 30000:30000 --cap-add SYS_NICE \
  -v "$PWD":/model:ro \
  approachingai/ktransformers:DSV4-specific
```

镜像默认使用 `KT_GPU_PREFILL_TOKEN_THRESHOLD=2048`，并将
`SWA_FULL_TOKENS_RATIO` 设为 `0.4`，使 SWA KV pool 能承载默认的
4096-token prefill chunk。设置 `KT_GPU_PREFILL_TOKEN_THRESHOLD=0` 可关闭该路径。KV-cache
profiling 会预留两份 raw 和两份 prepared layerwise slot 所需的容量，但服务启动时不会创建这些
tensor。首个达到所设阈值的请求才会分配 slot，因此会承担一次初始化开销；如果当时可用显存
不足，服务会关闭 layerwise prefill，并让该请求继续使用 CPU/GPU 混合推理。
