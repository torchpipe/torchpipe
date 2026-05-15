# benchmarks

这里保留 orchid 的可复现实验脚本，以及确认要继续公开保存的 benchmark 产物。

## 目录约定

- `run_*.py`：实验入口。
- `artifacts/`：保留的公开结果与图表。
- `datasets/`：本地数据集与最小样本。

## 当前推荐入口

- `run_evalscope_perf.py`：当前推荐的在线回归与 orchid/vLLM 双边对比入口。
- `run_simple_suite.py`：离线 decode 吞吐与 token 一致性总览。
- `run_gap_sharegpt.py`：ShareGPT online 对比，默认 server-app 为 `orchid.llmscheduler.server.api_server:app`。
- `run_final_vllm_bench_sharegpt.py`：vLLM bench serve 单组实验。
- `run_additional_experiments.py`：补充实验与方差统计。

## 输出路径

- 这些脚本的默认输出目录已经统一改成通过 `orchid.paths` 计算，不再依赖当前工作目录下的 `v13/...` 相对路径。
- `benchmarks/artifacts/` 默认作为本地结果目录处理，`.gitignore` 会忽略新生成的大多数产物。
- 需要正式对外同步的结果，应先把关键数字整理进 `docs/performance.md`，再按需挑选产物单独保留。

## 当前说明

- 推荐先执行 `uv pip install -e plugins/orchid`，再运行这些脚本。
- 在线高性能路径现在来自 `plugins/orchid/src/orchid/llmscheduler/` 这套已内收模块，而不是仓库外部路径。
- 当前 fresh GPU 结果里，`run_gap_sharegpt.py` 的 vLLM bench streaming 统计仍有异常；正式结论以 `docs/performance.md` 为准。
- 对需要稳定回归 streaming 路径的场景，优先使用 `run_evalscope_perf.py`，再按需补充 ShareGPT 对比。

## 文档关系

- 模型导出与最小验证：`../docs/model_export.md`
- 当前性能结论：`../docs/performance.md`
- EvalScope 在线回归：`./evalscope.md`
- 历史材料归档：`../research/archive/`
