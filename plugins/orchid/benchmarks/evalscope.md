# EvalScope 压测入口

## 安装

```bash
source .venv/bin/activate
uv pip install 'evalscope[perf]' -U
```

## 单目标压测

对任意 OpenAI 兼容服务：

```bash
python plugins/orchid/benchmarks/run_evalscope_perf.py \
  --suite standard \
  --target orchid=http://127.0.0.1:8000/v1/chat/completions
```

如果要只复现 `openqa_stream`，可以直接查看 `summary.md` 中该场景一行，或进入对应 `outputs/.../parallel_*_number_*/benchmark_summary.json` 看详细指标。

## 直接拉起 vLLM 做基线

```bash
python plugins/orchid/benchmarks/run_evalscope_perf.py \
  --suite smoke \
  --spawn-vllm-target vllm \
  --model Qwen/Qwen3-0.6B \
  --tokenizer-path Qwen/Qwen3-0.6B
```

## 直接拉起 orchid 服务

```bash
python plugins/orchid/benchmarks/run_evalscope_perf.py \
  --suite standard \
  --spawn-orchid-target orchid \
  --orchid-max-pages 32768 \
  --orchid-kv-cache-reserved-mb 4096
```

- 对 `Qwen/Qwen3-0.6B` 来说，`random_long` 和 `openqa_stream` 这类更重的场景对 KV page 预算更敏感。
- 如果仍沿用较小的 `LLMSCHEDULER_MAX_PAGES`，高并发长输入下可能出现 `OOM: Not enough pages`。
- 当前验证里，`--orchid-max-pages 32768 --orchid-kv-cache-reserved-mb 4096` 可以稳定跑完 `standard` 和 `full`。
- 当前 streaming 路径支持 `LLMSCHEDULER_STREAM_FLUSH_TOKENS` 调整 flush 粒度；默认值是 `1`，优先保持与逐 token SSE 更接近的输出语义。

## 对比建议

- `smoke`：最小联通验证，随机数据集，小并发。
- `standard`：常规回归，包含随机短输入、随机长输入和 openqa streaming。
- `full`：更高并发的长时间压测。

## 正确性验证

建议在跑在线压测前先做两步：

```bash
pytest plugins/orchid/tests/test_api_server_inprocess.py -q

python plugins/orchid/scripts/verify_trt.py \
  --model /root/.cache/orchid/models/Qwen_Qwen3-0.6B/fp16/model.composite.onnx \
  --tokenizer Qwen/Qwen3-0.6B \
  --engine /root/.cache/orchid/models/Qwen_Qwen3-0.6B/fp16/model.rtx5070ti.trt1016.plan \
  --fp16 \
  --prompt '你好，用一句话介绍你自己。' \
  --max_tokens 12
```

- 第一个命令验证 `/health`、非 streaming `/v1/chat/completions` 和 streaming SSE 的基本输出语义。
- 第二个命令验证当前 TRT 路径仍能正常生成文本。

## 输出

- 每个目标与场景的原始 EvalScope 输出会落在 `--out-dir/<target>/<scenario>/outputs/...`
- 汇总结果写到 `--out-dir/summary.md`
- 如果同一轮同时包含 `orchid` 和 `vllm`，会额外生成 `--out-dir/compare_summary.md`
- 完整参数与结果索引写到 `--out-dir/meta.json`

## 合并已有结果生成对比表

如果两边不是同一轮跑的，也可以直接合并两份 `meta.json`：

```bash
python plugins/orchid/benchmarks/run_evalscope_perf.py \
  --merge-meta plugins/orchid/benchmarks/artifacts/evalscope_perf/orchid_standard_8002_20260330/meta.json \
  --merge-meta plugins/orchid/benchmarks/artifacts/evalscope_perf/vllm_standard_20260330/meta.json \
  --out-dir plugins/orchid/benchmarks/artifacts/evalscope_perf/compare_standard_20260330
```

## 当前已验证结果

- `standard` 对比表：`benchmarks/artifacts/evalscope_perf/compare_standard_streamopt4_20260330/compare_summary.md`
- `full` 对比表：`benchmarks/artifacts/evalscope_perf/compare_full_streamopt4_20260330/compare_summary.md`
- 截至当前，orchid 在 `random_short` 与 `random_long` 上可以接近或超过 vLLM。
- `openqa_stream` 仍落后于 vLLM，但经过 streaming 路径优化后：
  - `standard` 提升到约 `0.896x`
  - `full` 提升到约 `0.901x`
