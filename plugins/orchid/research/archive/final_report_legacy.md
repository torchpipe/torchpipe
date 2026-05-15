# v13 Benchmarks Final Report（Simple Suite）

## 结论（聚焦）

1. **性能**：在 `more` 预设场景集合上，TRT decode 吞吐与 vLLM **eager** 同量级；vLLM **CUDA graph** 模式通常显著更快（详见 `simple_suite_more.*` 的 eager/graph 两列与 `perf_variance.*`）。
   - 本项目 TRT 后端新增 **TRT CUDA graph**（`trt cg`）对比列：用于降低 launch/dispatch 开销，见 `simple_suite_more.*` 的 `trt cg tok/s`。
   - 说明：TRT `trt cg` 测量采用“固定 KV 长度的 cached decode（freeze lens）”以满足 CUDA graph capture 约束；更接近某一 prefill_len 下的 steady-state token/s。
2. **正确性（关键解释）**：对长 decode 场景，**自由滚动 greedy 的 token_match 不是一个稳定/公平的正确性指标**；微小数值差异会触发“序列分叉”，随后被 greedy 反馈放大。
3. **更可信的正确性指标**：`teacher-forced next-token match` 在相同历史条件下衡量 next-token 是否一致，更适合作为 correctness gate（回归监控）。

  
## Suite 总览（more preset）

产物：

- MD：`simple_suite_more.md`
- CSV：`simple_suite_more.csv`
- PNG：`simple_suite_more.png`
- PDF：`final_report.pdf`

汇总表里包含：

- `prefix match mean / min prefix`：每个 request 从第 0 个 token 开始连续匹配的前缀长度统计
- `vllm graph tok/s / ratio graph`：vLLM 非 eager（启用 CUDA graph 的执行路径）下的 cached decode tok/s 及 TRT/vLLM graph 比例
- `trt cg tok/s / TRTcg/vllm graph`：TRT 后端启用 CUDA graph 时的 cached decode tok/s 及与 vLLM graph 的比例

## 深入分析：`bs4_p64_s128` 为什么会“TRT/vLLM 很接近但 token_match 偏低”

### 现象（固定 prompt，可复现）

自由滚动 greedy 下，部分 request 发生早期分叉（`min prefix` 较小），导致后续 token 序列进入反馈回路而逐步偏离；这会显著降低 `token_match_rate`，但不直接等价于“同 history 下 next-token 预测差”。

建议关注：

- `prefix match`（分叉有多早）
- `teacher-forced next-token match`（同 history 下是否一致）

### 关键证据：teacher-forced next-token 几乎一致

对 `bs4_p64_s128` 场景，teacher-forced 指标显示：

- `tf_next_token_match_rate` 接近 1（示例见 `bs4_p64_s128_teacher_forced.csv`）

说明在相同历史条件下，两边对 next-token 的判断几乎一致；自由滚动的差异主要来自 greedy 分叉的放大效应。

产物：

- CSV：`bs4_p64_s128_teacher_forced.csv`
- PNG：`bs4_p64_s128_teacher_forced.png`

## 为了更有信服力：新增的简单级别实验（已完成）

1. **多 prompt（多 seed）分布实验：`bs4_p64_s128`**
   - 输出 `token_match_rate / prefix / tf_next_token_match_rate` 随 seed 的变化
   - 产物：`bs4_p64_s128_seed_sweep.csv`、`bs4_p64_s128_seed_sweep.png`

2. **prefill-only next-token sweep（不同 prefill_len）**
   - 只对比 “prefill 最后一个 token 的 next-token” 是否一致（链路短、解释清晰）
   - 产物：`prefill_only_sweep.csv`、`prefill_only_sweep.png`

3. **性能重复测量方差（同 prompt 重复多次）**
   - 输出 tok/s 的 mean/std，剥离计时噪声
   - 产物：`perf_variance.csv`、`perf_variance.png`（包含 vLLM eager vs vLLM graph 对比）

实验入口脚本：`../run_additional_experiments.py`

## Online 基准：ShareGPT（vllm bench serve）

这一组测试使用 vLLM 自带的 online benchmark 工具：
- client：`vllm bench serve --backend openai-chat --endpoint /v1/chat/completions --dataset-name sharegpt --dataset-path <...>`
- server：
  - 我们：`python -m uvicorn llmscheduler.server.api_server:app ...`
  - vLLM：`python -m vllm.entrypoints.openai.api_server ...`

### vLLM 是否使用 FlashInfer？

本次环境的 vLLM 版本为 0.15.1。它“可以选择 FlashInfer 作为 attention backend”，但在实际跑数里选择的是 FlashAttention：
- vLLM server log 显示：`Using FLASH_ATTN attention backend out of potential backends: ('FLASH_ATTN', 'FLASHINFER', ...)`
  - 见 [vllm_server.log](file:///data2/zhangshiyang/onnxscheduler/v12/v13/benchmarks/artifacts/gap_sharegpt/real_sharegpt_n200_c1_4_8_10/conc_1/vllm/vllm_server.log#L13-L18)
- 同时 vLLM server 初始化过程中会出现 `flashinfer.jit` 的 autotuning 日志（代表 vLLM 运行时确实加载了 flashinfer 相关组件，但不是本轮 attention 的主 backend）。
  - 见 [vllm_server.log](file:///data2/zhangshiyang/onnxscheduler/v12/v13/benchmarks/artifacts/gap_sharegpt/real_sharegpt_n200_c1_4_8_10/conc_1/vllm/vllm_server.log#L29-L35)

### 口径公平性核查（重点：chat template / prompt 构造）

1) **ShareGPT 数据集采样方式**
- `vllm bench serve --dataset-name sharegpt` 只取每条样本的第 1 个 user turn（`conversations[0]["value"]`）作为 prompt，第 2 个 turn 仅用于估算默认 output_len（当你不显式指定 `--sharegpt-output-len` 时）。
- 因此这轮 online benchmark 并不是“完整多轮对话上下文”的 ShareGPT，而是“一问一答”的 user prompt 负载。

2) **请求格式（OpenAI Chat）**
- `openai-chat` backend 的 client 请求始终是 `messages=[{role:'user', content:[{type:'text', text: prompt}]}]`，也就是把 prompt 放在单条 user message 里发送。
- 这意味着“chat template 的应用”发生在 server 侧（vLLM 会对 chat messages 做 chat template 渲染）。

3) **`--skip-chat-template` 的作用范围**
- 该参数只影响某些 dataset 在 client 侧是否用 tokenizer 的 chat template 格式化 prompt。
- ShareGPTDataset 的实现本身不会在 client 侧应用 chat template，所以 `--skip-chat-template` 对 ShareGPTDataset 基本无影响。

4) **我们这边的对齐措施**
- llmscheduler server 新增 `LLMSCHEDULER_APPLY_CHAT_TEMPLATE=1` 时，会用 tokenizer 的 `apply_chat_template(..., add_generation_prompt=True, tokenize=True)` 生成 input_ids，从而更接近 vLLM 的 chat 模式 prompt 构造。
- gap 跑数脚本默认设置了 `LLMSCHEDULER_APPLY_CHAT_TEMPLATE=1`。

相关实现见：[app_factory.py](file:///data2/zhangshiyang/onnxscheduler/v12/llmscheduler/server/app_factory.py#L240-L271)

### 实测结果（真实 ShareGPT JSON）

数据集文件来自 orchid 缓存：
`/root/.cache/orchid/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json`

**主结果（num_prompts=200，warmups=10，sharegpt_output_len=128）**
- 产物： [gap_summary.md](file:///data2/zhangshiyang/onnxscheduler/v12/v13/benchmarks/artifacts/gap_sharegpt/real_sharegpt_n200_c1_4_8_10/gap_summary.md)

|max_concurrency|ours tok/s|vllm tok/s|ratio ours/vllm|备注|
|-:|-:|-:|-:|:-|
|1|362.26|436.78|0.8294|低并发差距主要来自注意力路径（见下）|
|4|1343.80|1483.89|0.9056|差距缩小|
|8|2493.29|2643.44|0.9432|差距进一步缩小|
|10|2958.08|2947.30|1.0037|基本持平|

**vLLM CUDA Graph 对比（enforce-eager=关 CUDA Graph）**
- 同一产物里附带：`vLLM enforce-eager (CUDA Graph off)` 表
- 结论：vLLM 默认模式相比 enforce-eager 有显著收益（大约 +17%~+49% output tok/s，视并发而定）。

### 归因：注意力路径是低并发瓶颈

限定性实验：把注意力路径直接 bypass（`LLMSCHEDULER_ATTENTION_BYPASS=1`）后，c=1 基本追平 vLLM：
- [gap_summary.md](file:///data2/zhangshiyang/onnxscheduler/v12/v13/benchmarks/artifacts/gap_sharegpt/real_sharegpt_attn_bypass_c1/gap_summary.md)

这说明在当前实现下，低并发的主要差距不在 TRT 主干，而在 CompositeAttention → FlashInfer paged-kv 的注意力实现与其 host 侧开销。
