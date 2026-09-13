# Kimi K2.5：从 PyPI 安装到 LoRA 训练、续训和推理

本教程使用 `0.7.0.post4` 发布组合，完成纯文本 Neko 风格微调。冻结原始
RAWINT4 routed experts，训练 attention LoRA 和 KT expert LoRA；不开 packing，
不修改原模型、安装目录或模型缓存。K2.6 沿用相同模型架构路径；本教程的固定
快照和实机验收使用 K2.5，未另行验收 K2.6、视觉训练或其他硬件组合。

先完成 4 步 smoke → 真续训 → 独立推理，再运行风格训练。4 步 smoke 只验证
功能闭环，不代表模型已经学会目标风格。最新安装复验结果见
[Release](https://github.com/kvcache-ai/ktransformers/releases/tag/v0.7.0.post4)。

## 1. 准备机器和两个新环境

实测训练机器为双 EPYC 9355、约 1.5 TiB RAM、8 张 RTX 5090；推理使用其中
4 张卡。这是实测配置，不是最低硬件需求。软件为 Linux x86_64、glibc 2.35+、
Python 3.12、NVIDIA driver 580.173.02、CUDA 12.8 工具链、Torch 2.9.1。

先确认 `nvidia-smi`、`nvcc --version` 正常，GPU 空闲。SGLang 首次启动可能
编译 kernel，需要 CUDA 工具链、C++ 编译器及可写缓存；首次加载不代表稳态速度。
模型、环境和缓存之外，为下面的 smoke、续训和 adapter 转换预留至少
140 GiB **持久化磁盘**；正式 Neko 输出另算。不要将示例输出放到 `/dev/shm`，
它占用真实 RAM，且重启会丢失。

工作目录也应放在容量充足的持久化磁盘上。下面将下载、编译和 LoRA 临时缓存
放在工作目录，避免沿用旧缓存或写满系统 `/tmp`。

在新目录下载固定版本的用户资料包，不需要克隆开发仓库：

```bash
mkdir kimi-post4-work
cd kimi-post4-work
curl --fail --location --retry 5 --remote-name \
  https://github.com/kvcache-ai/ktransformers/releases/download/v0.7.0.post4/kimi-k25-post4-user-kit.tar.gz
printf '%s  %s\n' \
  f9b6ca3dcde3e9349e8bddb92d3a48cdf4c4cfbddb274b17685053abba4aaa26 \
  kimi-k25-post4-user-kit.tar.gz | sha256sum --check
tar -xzf kimi-k25-post4-user-kit.tar.gz
cd kimi-k25-post4
```

后续命令均在解压后的 `kimi-k25-post4/` 下执行。资料包包含三个 YAML、数据准备
和 adapter 转换工具、带 SHA256 的依赖锁文件，以及配套训练工具 wheel。
在同一个 Bash 终端依次执行，只有向推理服务发送请求时需要另开终端。
命令显式指定新环境的 Python，无需激活环境，也不要激活旧训练环境。

| 来源 | 固定版本 |
| --- | --- |
| PyPI：`ktransformers`、`kt-kernel`、`sglang-kt` | `0.7.0.post4` |
| PyPI：`transformers-kt` | `5.6.0.post5` |
| PyPI：`accelerate-kt` | `1.14.0.post3` |
| Release `training-tools/`：LF KT-profile | `0.9.6.dev0+kt.20260912` |
| Release `training-tools/`：PEFT、TRL | `0.18.1+kt.20260912`、`0.24.0+kt.20260912` |

**三个训练工具 wheel 不在 PyPI。** LF 使用
[固定的 KT 安装分支](https://github.com/yyj6666667/LlamaFactory/tree/564574eb0214840ea4b9464b2f9178a52a075d73)，
PEFT/TRL 只调整依赖元数据，不修改运行时代码。因此不要用一次无版本的
`pip install` 或最新上游 LF 替代下面的锁定安装。

```bash
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export HF_HOME="$PWD/cache/huggingface"
export XDG_CACHE_HOME="$PWD/cache"
export TRITON_CACHE_DIR="$PWD/cache/triton"
export TMPDIR="$PWD/tmp"
mkdir -p "$TMPDIR"
export KIMI_PYPI_INDEX=https://pypi.org/simple
# 国内网络可将上行替换为：https://pypi.tuna.tsinghua.edu.cn/simple
python3.12 -m venv train-env
train-env/bin/python -m pip install --index-url "$KIMI_PYPI_INDEX" pip==25.2
train-env/bin/python -m pip install \
  --index-url "$KIMI_PYPI_INDEX" --timeout 120 --retries 10 --resume-retries 10 \
  --only-binary=:all: --no-binary=antlr4-python3-runtime \
  --require-hashes --find-links training-tools -r locks/train.lock
train-env/bin/python -m pip check

python3.12 -m venv serve-env
serve-env/bin/python -m pip install --index-url "$KIMI_PYPI_INDEX" pip==25.2
serve-env/bin/python -m pip install \
  --index-url "$KIMI_PYPI_INDEX" --timeout 120 --retries 10 --resume-retries 10 \
  --only-binary=:all: --require-hashes -r locks/serve.lock
serve-env/bin/python -m pip check
```

两次 `pip check` 都应输出 `No broken requirements found.`。ANTLR 4.9.3
使用校验过的官方源码包正常构建，其余依赖使用 wheel。不要使用 `--no-deps`、
关闭版本检查或复制旧 `site-packages`。上游 `transformers` / `accelerate` 与
KT 发行包共用 import namespace，不能混装在同一环境。

国内下载较慢时，只需将 `KIMI_PYPI_INDEX` 改为
`https://pypi.tuna.tsinghua.edu.cn/simple`（[清华 TUNA 使用说明](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)）。
保留相同版本与 `--require-hashes`；若镜像尚未同步该版本，改回官方 PyPI，
不要跳过校验或安装其他版本。

## 2. 下载模型并一次性准备数据

下载需要能访问 Hugging Face；清华 PyPI 源不代理模型和数据。如果训练机器无法
直连，可先在联网机器执行下面两条 `hf download`，再完整复制模型目录和
`data-source/NekoQA-10K.json` 到训练机器。不要只复制权重而遗漏配置或 remote code。

只需修改下面两个绝对路径。`KIMI_OUTPUT` 使用本轮新的持久化输出目录；
已经准备好相同 revision 的模型或数据时，可以跳过对应下载命令，仍需执行数据准备。

```bash
export KIMI_MODEL=/absolute/path/to/Kimi-K2.5
export KIMI_OUTPUT=/absolute/path/to/new-kimi-output
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$KIMI_OUTPUT"

train-env/bin/hf download moonshotai/Kimi-K2.5 \
  --revision 54383e83fa343a1331754112fb9e3410c55efa2f --local-dir "$KIMI_MODEL"
train-env/bin/hf download liumindmind/NekoQA-10K NekoQA-10K.json \
  --repo-type dataset --revision 1b2110c996a8237823b86c1a3d3e8a6762b38430 \
  --local-dir data-source
train-env/bin/python tools/split_nekoqa.py \
  --input data-source/NekoQA-10K.json --output-dir neko-splits
train-env/bin/python tools/prepare_kimi_data.py \
  --model "$KIMI_MODEL" --trust-remote-code \
  --input neko-splits/neko_train.json --eval-input neko-splits/neko_eval.json \
  --output-dir prepared-neko
```

保留完整原始权重、索引、tokenizer 和 remote code，不转全量 BF16、不修改模型
文件。`--trust-remote-code` 会执行固定快照中的模型代码，请先确认来源可信。

数据脚本校验原始 JSON 的 SHA256，去重并按 seed 42 固定划分为
9,477 条训练、468 条验证、32 条独立问答；保留 `split_manifest.json` 和
`preprocessing_manifest.json`。独立问答不参与训练或选 checkpoint。

预处理使用 Kimi 原生 non-thinking 模板，只有回答参与 loss。因此 YAML 必须
保持 `template: empty`、`train_on_prompt: false`，不要再次套聊天模板，也不要
直接将未经预处理的 Neko JSON 交给该配置。

## 3. 训练 4 步并保存完整 checkpoint

| 仓库配置 | 资料包内文件 | 用途 |
| --- | --- | --- |
| [train.yaml](train.yaml) | `examples/train-smoke.yaml` | 8 GPU、B1/GAS1、S512 上限、4 步 |
| [train-neko.yaml](train-neko.yaml) | `examples/train-neko.yaml` | 8 GPU、B1/GAS8、S4096 上限、一轮 Neko |
| [fsdp2_8gpu.yaml](fsdp2_8gpu.yaml) | `examples/fsdp2_8gpu.yaml` | 两种训练共用的 FSDP2 启动配置 |

仓库与资料包的 YAML 配置值一致。下面通过 LF 的 CLI 覆盖路径，不必逐个编辑
YAML；其他参数先保持不变。不要额外设置 `FORCE_TORCHRUN` 或再嵌套 `torchrun`。

```bash
PATH="$PWD/train-env/bin:$PATH" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
train-env/bin/accelerate launch --config_file examples/fsdp2_8gpu.yaml \
  --no_python train-env/bin/llamafactory-cli train examples/train-smoke.yaml \
  model_name_or_path="$KIMI_MODEL" kt_weight_path="$KIMI_MODEL" \
  dataset_dir="$PWD/prepared-neko" output_dir="$KIMI_OUTPUT/smoke-a"
```

配置使用 rank 8 / alpha 16 / dropout 0 的 attention LoRA，加上 KT fused
expert LoRA；CPU/GPU activation 均重算，`packing` / `neat_packing` 均关闭。
`use_kt: true` 由 YAML 决定，不需要额外环境开关或 `PYTHONPATH`。

成功标志：进程正常退出、loss 和 grad norm 有限，输出 `checkpoint-2` 和
`checkpoint-4`。完整 checkpoint 应包括：

- `adapter_model.safetensors`、`fused_expert_lora.safetensors` 和 adapter 配置；
- `kt_optimizer.index.json`、8 个 `optimizer_rank_*.pt`；
- `scheduler.pt`、`trainer_state.json`、8 个 `rng_state_*.pth`。

`save_only_model: false` 必须保留；只有 adapter 文件不算可真续训的 checkpoint。
本配方完整 checkpoint 约 29 GiB，转换后的 adapter 另需约 9.6 GiB。

## 4. 新进程从 step 2 真续训到 step 4

```bash
PATH="$PWD/train-env/bin:$PATH" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
train-env/bin/accelerate launch --config_file examples/fsdp2_8gpu.yaml \
  --no_python train-env/bin/llamafactory-cli train examples/train-smoke.yaml \
  model_name_or_path="$KIMI_MODEL" kt_weight_path="$KIMI_MODEL" \
  dataset_dir="$PWD/prepared-neko" output_dir="$KIMI_OUTPUT/smoke-b" \
  resume_from_checkpoint="$KIMI_OUTPUT/smoke-a/checkpoint-2"
```

保持 world size、数据、seed、scheduler 和 `max_steps: 4` 一致，检查日志确实从
step 2 恢复、最终到 step 4。`max_steps` 是总步数，不是额外训练步数；
`adapter_name_or_path` 只是加载权重，不能替代 `resume_from_checkpoint`。

连续训练与续训后的两组 adapter tensor 应完全相同，可直接检查（不依赖文件
序列化顺序）：

```bash
train-env/bin/python - "$KIMI_OUTPUT" <<'PY'
from pathlib import Path
import sys
import torch
from safetensors import safe_open

root = Path(sys.argv[1])
for name in ("adapter_model.safetensors", "fused_expert_lora.safetensors"):
    a = root / "smoke-a/checkpoint-4" / name
    b = root / "smoke-b/checkpoint-4" / name
    with safe_open(a, framework="pt", device="cpu") as left, safe_open(b, framework="pt", device="cpu") as right:
        assert set(left.keys()) == set(right.keys()), name
        for key in left.keys():
            assert torch.equal(left.get_tensor(key), right.get_tensor(key)), (name, key)
    print(name, "EXACT_RESUME_PASSED")
PY
```

两项都应输出 `EXACT_RESUME_PASSED`。发布验收还会比较各 rank optimizer/RNG、
scheduler 和恢复后的 loss；不要用删除 optimizer 状态的方式绕过续训错误。

## 5. 转换两部分 LoRA，在独立环境推理

先验证 smoke 产物，不必等正式训练完成：

```bash
export KIMI_CHECKPOINT="$KIMI_OUTPUT/smoke-b/checkpoint-4"
export KIMI_ADAPTER="$KIMI_OUTPUT/sglang-smoke"
train-env/bin/python tools/convert_kt_to_sglang_adapter.py \
  "$KIMI_CHECKPOINT" "$KIMI_ADAPTER" --base-model-name-or-path "$KIMI_MODEL"
```

转换保留原 checkpoint。该配置应导出 610 个 ordinary LoRA tensor 和 138,240 个
expert LoRA tensor；不能只复制普通 PEFT adapter 而漏掉 expert LoRA。

确认训练进程已经退出，再启动服务。日志保留在 adapter 目录旁：

```bash
set -o pipefail
PATH="$PWD/serve-env/bin:$PATH" CUDA_VISIBLE_DEVICES=0,1,2,3 \
serve-env/bin/python -m sglang.launch_server \
  --model-path "$KIMI_MODEL" --trust-remote-code \
  --served-model-name kimi --host 127.0.0.1 --port 30000 \
  --tensor-parallel-size 4 --dtype bfloat16 --context-length 2048 \
  --max-total-tokens 4096 --chunked-prefill-size 256 --max-running-requests 4 \
  --mem-fraction-static 0.75 --disable-cuda-graph --disable-radix-cache \
  --attention-backend triton --grammar-backend llguidance \
  --kt-method RAWINT4 --kt-weight-path "$KIMI_MODEL" \
  --kt-cpuinfer 64 --kt-threadpool-count 2 --kt-num-gpu-experts 0 \
  --random-seed 42 --watchdog-timeout 900 \
  --enable-lora --lora-backend triton --lora-paths "neko=$KIMI_ADAPTER" \
  2>&1 | tee "$KIMI_ADAPTER.server.log"
```

服务会持续占用当前终端。另开终端，先确认 ready，再显式请求 adapter 名称
`kimi:neko`；如果 `/health` 尚未成功，等服务加载完成后再请求：

```bash
curl --noproxy 127.0.0.1 --fail http://127.0.0.1:30000/health
curl --noproxy 127.0.0.1 --fail http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"kimi:neko","messages":[{"role":"user","content":"我今天学习有点累，请用两句话鼓励我。"}],"temperature":0,"seed":42,"max_tokens":1024,"chat_template_kwargs":{"thinking":false,"enable_thinking":false}}'
```

成功标志不只是 HTTP 200：响应有完整正文，且服务日志包含 ordinary/expert
adapter 加载记录和 `Loaded KT expert LoRA for layer ...`，覆盖 expert 层
1–60。API 中出现 adapter 名称本身不能证明 CPU expert LoRA 已加载。

本版本 CPU expert LoRA 在启动时静态加载。做 base 对照时，必须停止当前服务，
再启动一个不带 `--enable-lora`、`--lora-backend`、`--lora-paths` 的新进程，并请求
`model: kimi`；不能在同一个已加载 expert LoRA 的服务中切换模型名假装关闭 adapter。

## 6. 训练 Neko 风格，再检查未见过的问题

先在服务终端按 Ctrl-C 正常停止 smoke 服务，确认 GPU 已释放，再开始 8 卡训练：

```bash
PATH="$PWD/train-env/bin:$PATH" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
train-env/bin/accelerate launch --config_file examples/fsdp2_8gpu.yaml \
  --no_python train-env/bin/llamafactory-cli train examples/train-neko.yaml \
  model_name_or_path="$KIMI_MODEL" kt_weight_path="$KIMI_MODEL" \
  dataset_dir="$PWD/prepared-neko" output_dir="$KIMI_OUTPUT/neko"
```

本配方从 base 新训，预计一轮 149 步；每 100 步保存、验证，保留两个 checkpoint。
S4096 是长度上限：该数据准备结果最长 2,120 token，没有 packing，不是满长吞吐
benchmark。不要为追求显示的 tokens/s 擅自打开 packing 或更改 activation 策略。

已验证的风格产物为 `checkpoint-100`，验证 loss 为 1.1463。保持一轮 scheduler
并取这个 checkpoint；改为 `max_steps: 100` 会改变学习率轨迹，不是同一实验。
可以让一轮正常结束后再使用保留的 checkpoint-100，无需强杀训练进程。

训练结束后，转换风格 checkpoint 到新目录：

```bash
export KIMI_CHECKPOINT="$KIMI_OUTPUT/neko/checkpoint-100"
export KIMI_ADAPTER="$KIMI_OUTPUT/sglang-neko"
train-env/bin/python tools/convert_kt_to_sglang_adapter.py \
  "$KIMI_CHECKPOINT" "$KIMI_ADAPTER" --base-model-name-or-path "$KIMI_MODEL"
```

然后执行第 5 节的 **SGLang 服务启动命令**，使用这里刚设置的变量，不再执行
smoke 路径的赋值命令。在第二个终端发送同样的请求，检查语气是否已经变化。

用 `neko-splits/heldout.json` 中的 32 个 `prompt` 请求 `kimi:neko`，与独立 base
进程比较；不要添加“请扮演猫娘”等 system prompt。记录原始回答，分别判断风格、
内容相关性、回答是否完整；另外检查“17+25 只输出数字”“原样输出春暖花开”和 JSON
格式指令。风格、loss 和指令遵循是不同指标。

<details>
<summary>批量保存 32 条独立问答</summary>

在第二个终端进入同一个 `kimi-k25-post4/` 目录后执行。输出文件必须不存在，
避免覆盖已有结果。跑独立 base 对照时，将 `model_id` 改为 `kimi`，并更换输出文件名。

```bash
serve-env/bin/python - <<'PY'
import json
from pathlib import Path
import requests

model_id = "kimi:neko"
questions = json.loads(Path("neko-splits/heldout.json").read_text())
session = requests.Session()
session.trust_env = False
with Path("neko-heldout-responses.jsonl").open("x", encoding="utf-8") as output:
    for index, row in enumerate(questions, 1):
        body = {
            "model": model_id,
            "messages": [{"role": "user", "content": row["prompt"]}],
            "temperature": 0, "seed": 42, "max_tokens": 1024,
            "chat_template_kwargs": {"thinking": False, "enable_thinking": False},
        }
        response = session.post("http://127.0.0.1:30000/v1/chat/completions", json=body, timeout=900)
        response.raise_for_status()
        output.write(json.dumps({"prompt": row["prompt"], "response": response.json()}, ensure_ascii=False) + "\n")
        output.flush()
        print(f"{index}/{len(questions)}", flush=True)
PY
```

打开 `neko-heldout-responses.jsonl`，检查 `choices[0].message.content` 和
`finish_reason`。若为 `length`，回答被长度上限截断，不能算完整回答。

</details>

此前 checkpoint-100 的 32 条回答均呈现明显风格，但两项严格原样输出测试失败。
这是风格适配的实测证据，**不是通用能力、安全性或全部质量指标无损的承诺**。
自己的数据也应保留独立测试集；不要仅凭训练 loss 下降判断成功。

## 排错与复现记录

| 现象 | 先检查 |
| --- | --- |
| 下载超时 | 保留 pip 下载缓存，重试相同锁文件；不要换未锁版本或关闭 SHA 校验。 |
| `hf download` 超时 | 检查 Hugging Face 连通性；可在联网机器下载固定 revision 后完整拷贝，换 PyPI 源对此无效。 |
| pip 依赖冲突、找不到 LF/PEFT/TRL 版本 | 是否在新环境，是否下载了配套资料包并使用 `--find-links training-tools`。 |
| 训练 loss 异常或模板重复 | 是否先执行数据准备，且保持 `template: empty`、`packing: false`。 |
| 加载时 rank 0 消失、Gloo connection closed | 查看 rank 0 日志、主机 OOM 记录和可用 RAM；模型重载峰值高于稳定训练，先停止并发加载、清理可恢复的重复缓存。 |
| 续训变成从零开始 | 是否传入完整 checkpoint 的 `resume_from_checkpoint`，而非只加载 adapter。 |
| 推理像 base 或只有部分变化 | 转换是否包含两类 LoRA，是否以新进程加载，expert 1–60 层是否都加载，以及请求名是否为 `kimi:neko`。 |

保留最终 YAML、数据 manifest、安装锁、训练日志、完整 checkpoint 和推理响应。
官方 Release 的 `PUBLICATION.json`（资料包内）记录公开 wheel SHA256，
`main-equivalence.json` 记录构建源码与锁定 main 的等价证明；上传成功本身不能替代
新环境训练、续训和推理验收。
