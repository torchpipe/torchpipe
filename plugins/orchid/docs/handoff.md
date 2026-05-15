# 交接说明

这份文档面向接手 `plugins/orchid` 的同事，目标是用最短路径说明：
- 这个插件现在是什么状态
- 从哪里开始看
- 哪些命令已经验证过
- 当前有哪些已知限制

## 当前结论

- 插件目录已经从历史 `v13` 迁移到 `plugins/orchid`，并采用 `pyproject.toml + src/orchid` 的现代 Python 项目布局。
- 公开入口统一为 `orchid.llmscheduler.*`，不再保留 `llmscheduler_v13` 兼容层。
- 当前推荐的高性能在线入口是 `orchid.llmscheduler.server.api_server:app`。
- 离线最小验证入口是 `plugins/orchid/scripts/verify_trt.py`。

## 建议阅读顺序

1. `README.md`
2. `docs/model_export.md`
3. `docs/performance.md`
4. `benchmarks/README.md`
5. `benchmarks/evalscope.md`
6. `research/archive/final_report_legacy.md`

## 已验证命令

以下命令已经在当前环境实际执行过：

```bash
uv pip install -e plugins/orchid

pytest plugins/orchid/tests/test_api_server_inprocess.py \
  plugins/orchid/tests/test_schedule_step_plan.py \
  plugins/orchid/tests/test_page_manager_free.py -q

python plugins/orchid/scripts/verify_trt.py \
  --model /root/.cache/orchid/models/Qwen_Qwen3-0.6B/fp16/model.composite.onnx \
  --tokenizer Qwen/Qwen3-0.6B \
  --engine /root/.cache/orchid/models/Qwen_Qwen3-0.6B/fp16/model.rtx5070ti.trt1016.plan \
  --fp16 \
  --prompt '你好，用一句话介绍你自己。' \
  --max_tokens 12

python plugins/orchid/benchmarks/run_gap_sharegpt.py --dry-run \
  --model-path /root/.cache/orchid/models/Qwen_Qwen3-0.6B/fp16/model.composite.onnx \
  --tokenizer-path Qwen/Qwen3-0.6B \
  --engine-path /root/.cache/orchid/models/Qwen_Qwen3-0.6B/fp16/model.rtx5070ti.trt1016.plan

python plugins/orchid/benchmarks/run_evalscope_perf.py \
  --suite standard \
  --spawn-orchid-target orchid \
  --orchid-max-pages 32768 \
  --orchid-kv-cache-reserved-mb 4096

python plugins/orchid/benchmarks/run_evalscope_perf.py \
  --suite smoke \
  --spawn-vllm-target vllm \
  --vllm-gpu-mem 0.18
```

## 推荐入口

- **在线服务**
  - `orchid.llmscheduler.server.api_server:app`
- **离线 TRT 验证**
  - `python plugins/orchid/scripts/verify_trt.py ...`
- **ShareGPT 对比**
  - `python plugins/orchid/benchmarks/run_gap_sharegpt.py ...`
- **EvalScope 在线回归**
  - `python plugins/orchid/benchmarks/run_evalscope_perf.py ...`
- **vLLM bench serve**
  - `python plugins/orchid/benchmarks/run_final_vllm_bench_sharegpt.py ...`

## 当前已知问题

- `run_gap_sharegpt.py` 在 vLLM bench 的 streaming 统计上仍可能出现异常值，特别是 fresh `conc=10` 指标不能直接当正式结论。
- `run_simple_suite.py` 在当前 GPU 显存条件下，可能因为同进程同时驻留 vLLM 和 TRT 上下文而 OOM。
- `verify_trt.py` 虽然已经能跑通，但输出质量当前只能当作最小通路验证，不应把单次文本样例当作质量结论。
- EvalScope 的 `openqa_stream` 是当前最稳定的 streaming 回归入口；经过本轮 streaming 路径优化后，orchid 大约达到 vLLM 的 `0.90x`，但仍存在进一步优化空间。

## 结果口径

- 正式性能结论以 `docs/performance.md` 中表格为准。
- 当前 online streaming 回归与 orchid/vLLM 对比，以 `benchmarks/evalscope.md` 和 `compare_summary.md` 为准。
- `benchmarks/artifacts/` 默认作为本地结果目录处理，新增产物大多不会自动纳入版本控制。
- 如果后续要对外同步结果，建议先把关键数字写进文档，再决定是否保留原始产物。

## 交接建议

- 如果只是继续维护服务路径，优先从 `src/orchid/llmscheduler/server/` 和 `engine/` 看起。
- 如果要继续查性能问题，优先读 `docs/performance.md`，然后再看 `benchmarks/run_gap_sharegpt.py`。
- 如果要继续整理项目结构，优先保持 `pyproject.toml + src/orchid` 布局，不再回退到 `PYTHONPATH` 风格运行方式。
