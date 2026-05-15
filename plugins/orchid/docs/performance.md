# 性能结果整理

这份说明用于对外交接和同步当前结论，优先保留这里的表格与判断。旧版长报告已经移到 `research/archive/`。

## 当前状态

- `plugins/orchid` 的主要 Python 运行路径已经完成内收，不再依赖 `/workspace/onnxscheduler/v12` 这类外部源码目录。
- 当前正式高性能入口是 `orchid.llmscheduler.server.api_server:app`。
- `llmscheduler_v13` 兼容层已经移除，当前只保留 `orchid.llmscheduler.*` 这一套实现与命名。
- 当前环境已经恢复 CUDA，可见设备为 `NVIDIA GeForce RTX 5070 Ti`。
- 本轮新增了一轮 fresh 正确性验证和一轮 fresh 在线压测；但 `run_simple_suite.py` 在同进程同时拉起 vLLM 与 TRT 路径时仍会因为显存竞争 OOM。
- `run_gap_sharegpt.py` 在当前内收后的 server 路径上可以跑通，但 vLLM bench 给出的 `conc=10` fresh 指标出现明显异常值，因此本轮不把这组 fresh gap 数据升级为新的正式基线。
- `benchmarks/artifacts/` 默认按本地结果目录管理；即使原始产物不随仓库交付，这份文档里的表格仍应视为当前正式口径。
- 本轮补充了一套 EvalScope 在线回归，用来稳定复现 streaming 路径；这套结果当前作为 streaming 对比与回归口径。

## 当前正式性能结论

来源：`benchmarks/artifacts/gap_sharegpt/final_fixed_c1_4_8_10/gap_summary.md`

| 并发 | ours tok/s | vLLM tok/s |  ratio | ours TTFT | vLLM TTFT |
| -: | ---------: | ---------: | -----: | --------: | --------: |
|  1 |     353.02 |     432.88 | 0.8155 |      7.89 |      9.39 |
|  4 |    1310.29 |    1449.95 | 0.9037 |     13.39 |     17.09 |
|  8 |    2485.59 |    2572.89 | 0.9661 |     17.48 |     27.06 |
| 10 |    2952.90 |    2822.90 | 1.0461 |     18.05 |     29.97 |

这组结果对应 vLLM 默认 CUDA Graph 路径，也是当前目录内应当对外引用的正式 ShareGPT 主对比结果。

## 当前正确性与结构验证

- `pytest plugins/orchid/tests/test_api_server_inprocess.py plugins/orchid/tests/test_schedule_step_plan.py plugins/orchid/tests/test_page_manager_free.py -q` 已通过。
- `python plugins/orchid/scripts/verify_trt.py --model ... --tokenizer ... --engine ... --prompt ...` 已重新跑通；当前仍把它视为最小 TRT 通路验证，而不是最终文本质量结论。
- `python plugins/orchid/benchmarks/run_gap_sharegpt.py --dry-run ...` 已通过，默认 server-app 已切到 `orchid.llmscheduler.server.api_server:app`。
- `LLMSCHEDULER_TEST_MODE=1` 下，`orchid.llmscheduler.server.api_server` 能构造本地 FastAPI app。
- `orchid.llmscheduler.server.api_server:app` 已收到一条真实 `/v1/chat/completions` 请求，返回中文约束性结果，说明在线 chat 路径能正常走通。
- `pytest plugins/orchid/tests/test_api_server_inprocess.py -q` 现在同时覆盖了 `/health`、非 streaming `/v1/chat/completions` 与 streaming SSE 基本输出。

## 当前 EvalScope online 结果

来源：`benchmarks/artifacts/evalscope_perf/compare_standard_streamopt4_20260330/compare_summary.md`

| 场景 | orchid req/s | vLLM req/s | req ratio | orchid tok/s | vLLM tok/s | tok ratio | orchid TTFT | vLLM TTFT |
| - | -: | -: | -: | -: | -: | -: | -: | -: |
| openqa_stream | 10.83 | 12.09 | 0.8958 | 1386.25 | 1547.98 | 0.8955 | 0.008 | 0.011 |
| random_long | 10.29 | 9.71 | 1.0597 | 1317.58 | 1243.12 | 1.0599 | 0.077 | 0.090 |
| random_short | 27.21 | 28.39 | 0.9584 | 1741.41 | 1816.64 | 0.9586 | 0.040 | 0.032 |

来源：`benchmarks/artifacts/evalscope_perf/compare_full_streamopt4_20260330/compare_summary.md`

| 场景 | orchid req/s | vLLM req/s | req ratio | orchid tok/s | vLLM tok/s | tok ratio | orchid TTFT | vLLM TTFT |
| - | -: | -: | -: | -: | -: | -: | -: | -: |
| openqa_stream | 21.06 | 23.38 | 0.9008 | 2695.92 | 2992.54 | 0.9009 | 0.011 | 0.012 |
| random_long | 15.69 | 11.67 | 1.3445 | 2008.40 | 1493.17 | 1.3451 | 0.092 | 0.199 |
| random_short | 41.19 | 41.94 | 0.9821 | 2636.24 | 2684.17 | 0.9821 | 0.053 | 0.054 |

- 这组结果说明：orchid 在 `random_long` 上已经能追平或超过 vLLM，但在 `openqa_stream` 上仍落后。
- 当前 `openqa_stream` 的主要短板更像 streaming serving 路径，而不是基础 TRT 或长输入 KV 预算本身。
- 本轮已把 engine 内逐 token detokenize 改成 API 层增量解码，并把结果通道从 `janus.Queue` 改成 `asyncio.Queue + loop.call_soon_threadsafe`；优化后 `openqa_stream` 在 `standard/full` 下都提升到了约 `0.90x vLLM`。

## 本轮保留的复验结果

来源：`benchmarks/artifacts/revalidated_simple_suite.md`

| 场景           | trt tok/s | vLLM eager tok/s |  ratio | token match |
| ------------ | --------: | ---------------: | -----: | ----------: |
| bs1\_p16\_s8 |    317.61 |           281.94 | 1.1265 |      1.0000 |

来源：`benchmarks/artifacts/revalidated_simple_suite_bs4.md`

| 场景            | trt tok/s | vLLM eager tok/s |  ratio | token match |
| ------------- | --------: | ---------------: | -----: | ----------: |
| bs4\_p64\_s32 |   1232.41 |          1047.38 | 1.1767 |      1.0000 |

来源：`benchmarks/artifacts/final_vllm_bench/sharegpt/revalidated_random2/20260325_051814/summary.md`

| 指标             |     数值 |
| -------------- | -----: |
| request\_rps   |  32.05 |
| output\_tok\_s | 224.34 |
| mean\_ttft\_ms |  22.36 |
| mean\_tpot\_ms |   6.94 |

这些结果仍然是当前目录里可对应到现有产物的有效记录，但它们不是新的“结构内收后 fresh GPU 全量重跑”。

## 本轮 fresh GPU 结果

来源：`benchmarks/artifacts/final_vllm_bench/sharegpt/fresh_random2/20260326_071056/summary.md`

| 指标             |    数值 |
| -------------- | ----: |
| request\_rps   | 65.68 |
| output\_tok\_s | 525.4 |
| mean\_ttft\_ms |   8.2 |
| mean\_tpot\_ms |  3.15 |

来源：`benchmarks/artifacts/gap_sharegpt/fresh_c10_n200/gap_summary.md`

| 并发 | ours tok/s | vLLM tok/s |  ratio | ours TTFT | vLLM TTFT |
| -: | ---------: | ---------: | -----: | --------: | --------: |
| 10 |   23483.79 |    2823.69 | 8.3167 |     65.65 |     28.21 |

- 这组 `fresh_c10_n200` 数据虽然脚本运行成功，但 `Peak concurrent requests` 超过配置上限、`ours tok/s` 明显失真，因此当前只把它当作兼容性冒烟结果，不把它视为可信主结论。
- `run_simple_suite.py` 在本轮 fresh 复跑里因为同进程同时驻留 vLLM 和 TRT 上下文而 OOM，暂不把它写成新的正式对比值。

## 历史参考结果

来源：`benchmarks/artifacts/gap_sharegpt/real_sharegpt_n200_c1_4_8_10/gap_summary.md`

| 并发 | ours tok/s | vLLM tok/s |  ratio |
| -: | ---------: | ---------: | -----: |
|  1 |     362.26 |     436.78 | 0.8294 |
|  4 |    1343.80 |    1483.89 | 0.9056 |
|  8 |    2493.29 |    2643.44 | 0.9432 |
| 10 |    2958.08 |    2947.30 | 1.0037 |

来源：`benchmarks/artifacts/gap_sharegpt/real_sharegpt_attn_bypass_c1/gap_summary.md`

| 场景                              | ours tok/s | vLLM tok/s |  ratio |
| ------------------------------- | ---------: | ---------: | -----: |
| attention bypass, concurrency=1 |     438.89 |     435.86 | 1.0070 |

## 建议如何解读

- 把 `final_fixed_c1_4_8_10` 当作当前正式结果。
- 把 `real_sharegpt_n200_c1_4_8_10` 当作历史参考结果，用来对照当前目录整理前后的性能量级。
- 把 `real_sharegpt_attn_bypass_c1` 当作低并发瓶颈归因证据。
- 把 `revalidated_*` 结果当作当前目录里的补充复验证据。
- 把 `fresh_random2` 当作当前结构内收后的 fresh 在线冒烟结果。
- 当前若要产出新的正式 ShareGPT 主对比，需要先修正 `run_gap_sharegpt.py` 在 vLLM bench 下的 streaming 统计异常，再重新跑 `1,4,8,10` 全量结果。
- 当前若要做稳定的 streaming 回归或 orchid/vLLM 在线双边对比，优先用 EvalScope，而不是只依赖 `run_gap_sharegpt.py`。

## 复现实验入口

- `python plugins/orchid/benchmarks/run_gap_sharegpt.py --server-app orchid.llmscheduler.server.api_server:app ...`
- `python plugins/orchid/benchmarks/run_final_vllm_bench_sharegpt.py --server-app orchid.llmscheduler.server.api_server:app ...`
- `python plugins/orchid/benchmarks/run_simple_suite.py ...`
- `python plugins/orchid/benchmarks/run_evalscope_perf.py --suite standard --spawn-orchid-target orchid --orchid-max-pages 32768 --orchid-kv-cache-reserved-mb 4096`
- `python plugins/orchid/benchmarks/run_evalscope_perf.py --suite smoke --spawn-vllm-target vllm --vllm-gpu-mem 0.18`
- `python plugins/orchid/benchmarks/run_evalscope_perf.py --merge-meta <orchid-meta.json> --merge-meta <vllm-meta.json> --out-dir <compare-dir>`
