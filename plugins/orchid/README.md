# orchid 插件整理说明

项目组织：
- `pyproject.toml`：项目元数据、依赖和打包入口。
- `src/orchid/`：正式 Python 包根。
- `src/orchid/llmscheduler/`：已经内收到 orchid 目录中的高性能 serving 基础模块，用来消除对外部 `llmscheduler` 源码目录的依赖。

## 目录关系

- `docs/`：当前仍推荐阅读的文档。
- `benchmarks/`：可复现实验脚本与公开保留的性能产物。
- `scripts/`：最小化手工验证脚本。
- `tests/`：pytest 入口与测试产物。
- `research/archive/`：历史报告与旧资料归档，不作为当前正式入口。
- `src/orchid/`：正式包源码。
- `src/orchid/llmscheduler/`：已 vendored 到当前目录里的 serving/config/protocol/runtime/TRT 基础模块。

## 推荐阅读顺序

1. `docs/handoff.md`
2. `docs/model_export.md`
3. `docs/performance.md`
4. `benchmarks/README.md`
5. `benchmarks/evalscope.md`

## 当前命名约定

- 目录名继续保留 `orchid`，避免和已有实验记录、脚本入口完全断开。
- 文档、测试和脚本优先使用功能名，不再把 `orchid` 放进文件名。
- benchmark 与 test 的默认输出目录统一通过 `orchid.paths` 计算，不再依赖 `v13/...` 这种硬编码相对路径。

## 当前开发方式

- 在 `plugins/orchid` 目录下使用 `pyproject.toml` 管理依赖与打包。
- 推荐先执行 `uv pip install -e plugins/orchid`，再运行脚本、测试和 benchmark。
- 包导入统一使用 `orchid.*`，不再依赖 `PYTHONPATH=/workspace/torchpipe/plugins` 这类临时方式。

## 当前前提

- `plugins/orchid` 的主要 Python 运行路径已经不再依赖 `/workspace/onnxscheduler/v12` 这类外部源码目录。
- 当前推荐的高性能在线入口是 `orchid.llmscheduler.server.api_server:app`。
- 之前带 `llmscheduler_v13` 的兼容层已经移除，统一只保留 `orchid.llmscheduler.*` 这一套命名。
- 真实测速仍依赖 `flashinfer`、`tensorrt`、`vllm`、可用 CUDA GPU，以及本地模型与数据集缓存。

## 当前结论

- 建议先看 `docs/handoff.md`，里面汇总了入口、安装方式、验证命令和当前已知问题。
- 第一阶段模型导出与最小验证入口见 `docs/model_export.md`。
- 当前正式性能结论见 `docs/performance.md`。
- 当前最完整的在线回归与双边对比入口见 `benchmarks/run_evalscope_perf.py` 与 `benchmarks/evalscope.md`。
- 旧版长报告已归档到 `research/archive/final_report_legacy.md`。
