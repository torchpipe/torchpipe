# 模型导出与最小验证

这份文档对应 orchid 整理后的第一步：先把模型导出到 ONNX / composite ONNX，再跑最小化 TensorRT 验证链路。

如果是第一次接手这个插件，建议先读 `docs/handoff.md` 再回来执行这里的命令。

## 文档范围

- 这里只保留当前仓库内能直接对应到现有文件的命令。
- 已经找不到实现的旧命令不再保留为正式入口。
- API 层的最小验证改为直接执行仓库里的 pytest 用例。
- 当前主要运行路径已经切到 `plugins/orchid/pyproject.toml` 管理，不再依赖外部 `llmscheduler` 源码目录。

## 0. 运行前提

- Python 依赖由 `plugins/orchid/pyproject.toml` 管理，GPU 相关依赖建议安装 `.[gpu]`。
- 路径前提：推荐用 editable install 安装当前插件。
- GPU 前提：需要可用 CUDA GPU；已有旧版 `model.plan` 时，也可能因为 TensorRT 小版本不匹配而失效。
- 如果出现 `Serialization assertion safeVersionRead == kSAFE_SERIALIZATION_VERSION failed`，说明旧 engine 需要在当前环境重建。

```bash
uv pip install -e plugins/orchid
uv pip install -e 'plugins/orchid[gpu,dev]'
```

## 1. 导出 ONNX

```bash
uv pip install optimum[onnxruntime] onnxslim

optimum-cli export onnx \
  --model Qwen/Qwen3-0.6B \
  $HOME/.cache/orchid/Qwen3-0.6B-onnx-fp16/ \
  --opset=21 \
  --dtype=fp16 \
  --slim \
  --device cpu
```

产物：`$HOME/.cache/orchid/Qwen3-0.6B-onnx-fp16/model.onnx`

## 2. 转换 composite ONNX

在仓库根目录执行：

```bash
python scripts/convert_to_composite.py \
  --model $HOME/.cache/orchid/Qwen3-0.6B-onnx-fp16/model.onnx \
  --output $HOME/.cache/orchid/Qwen3-0.6B-onnx-fp16/model.composite.onnx \
  --model_id "Qwen/Qwen3-0.6B"
```

## 3. 最小 TensorRT 验证

当前仓库内保留的脚本是 `plugins/orchid/scripts/verify_trt.py`。

```bash
export LLMSCHEDULER_TRT_INPUT_IDS_PROFILES="1,32,64;64,512,1024;1024,3072,4096"

python plugins/orchid/scripts/verify_trt.py \
  --model $HOME/.cache/orchid/Qwen3-0.6B-onnx-fp16/model.composite.onnx \
  --tokenizer Qwen/Qwen3-0.6B \
  --fp16
```

如果已有 TensorRT engine，可继续传 `--engine`。如果当前 TensorRT 版本变了，优先去掉 `--engine` 让脚本在当前环境重新构建。

## 4. API 层最小验证

当前建议用现有测试直接做最小 API 验证：

```bash
pytest plugins/orchid/tests/test_api_server_inprocess.py -q
```

这个用例会直接构造 FastAPI 应用并验证 `/health`、非 streaming `/v1/chat/completions` 与 streaming SSE 基本输出。

## 5. 高性能在线入口

- 当前推荐入口：`orchid.llmscheduler.server.api_server:app`
- 当前已经去掉 `llmscheduler_v13` 这一层，统一只保留 `orchid.llmscheduler.*` 命名。

## 6. 性能与进一步验证

- 简单吞吐对比：`python plugins/orchid/benchmarks/run_simple_suite.py ...`
- ShareGPT 对比：`python plugins/orchid/benchmarks/run_gap_sharegpt.py ...`
- EvalScope 在线回归：`python plugins/orchid/benchmarks/run_evalscope_perf.py ...`
- 已整理的结果说明：见 `docs/performance.md`

## 7. 其他模型

当 composite ONNX 上的属性推断不足时，仍可通过以下环境变量补齐：

- `LLMSCHEDULER_NUM_LAYERS`
- `LLMSCHEDULER_NUM_HEADS`
- `LLMSCHEDULER_KV_NUM_HEADS`
- `LLMSCHEDULER_HEAD_DIM`
