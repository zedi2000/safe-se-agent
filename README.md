# Safe Self-Evolving Agent Research Scaffold

这是一个面向 Milestone 1 的小规模复现代码：实现一个 Simple Agent，并证明
`Memory + Reflection + Retrieval` 可以形成“从经验中学习”的闭环。

代码目标是 research clarity，不是产品化框架。实验层只依赖 `AgentAdapter`，因此后续可以把
被测框架从当前 `SimpleAgentAdapter` 换成 OpenClaw、LangChain 或其他 agent，而不需要重写
评测流程。

## 快速开始

```bash
conda env create -f environment.yml
conda activate safe-se-agent
python scripts/prepare_gsm8k.py --limit-train 20 --limit-eval 20
python scripts/prepare_medqa.py --limit-train 20 --limit-eval 20
python scripts/prepare_toolalpaca.py --limit-train 20 --limit-eval 20
python scripts/prepare_oep.py
python scripts/run_m1_demo.py --mode llm --run-id gsm8k_m1_llm
pytest -q
```

默认 demo 使用 GSM8K small 数据。离线模式只用于流程 smoke test；真实 performance 以 LLM
backend 为准。

每次 demo 默认写入：

- `runs/<run_id>/memory.jsonl`
- `runs/<run_id>/interaction_log.jsonl`
- `runs/<run_id>/baseline_results.jsonl`
- `runs/<run_id>/train_results.jsonl`
- `runs/<run_id>/self_evolution_results.jsonl`
- `runs/<run_id>/summary.json`

如果只想检查流程是否跑通，可以显式使用合成数据：

```bash
python scripts/run_m1_demo.py --mode offline \
  --train data/m1_train.jsonl \
  --eval data/m1_eval.jsonl \
  --run-id synthetic_smoke
```

默认 memory 更新策略是 `sliding_window`：训练交互会逐条写入 `interaction_log.jsonl`，
但只有当前样本答错时才对最近窗口做 reflection 并尝试写入 memory。旧的
`per_interaction` 和 `batch` 策略仍保留，用作 ablation。

答案评分会做轻量 dataset-aware normalization：GSM8K 使用数值归一化，MedQA 会把
`D**`、`**D**`、`Answer: D` 等输出归一到选项字母。ToolAlpaca 除最终答案匹配外，还会在
`run_metadata` 中记录 `tool_step_count`、`expected_tool_step_count` 和 `tool_step_delta`；
这对应 OEP 中 tool-use 场景更关注冗余工具调用和步骤膨胀的评测方式。

## 真实数据集准备

OEP 论文中的三个基础数据源目前都可以转换成统一的 `Task` JSONL schema：

```bash
python scripts/prepare_gsm8k.py --limit-train 20 --limit-eval 20
python scripts/prepare_medqa.py --limit-train 20 --limit-eval 20
python scripts/prepare_toolalpaca.py --limit-train 20 --limit-eval 20
python scripts/prepare_oep.py
```

对应 smoke test：

```bash
python scripts/run_m1_demo.py --mode offline --train data/gsm8k_train_small.jsonl --eval data/gsm8k_eval_small.jsonl --run-id gsm8k_smoke --no-progress
python scripts/run_m1_demo.py --mode offline --train data/medqa_train_small.jsonl --eval data/medqa_eval_small.jsonl --run-id medqa_smoke --no-progress
python scripts/run_m1_demo.py --mode offline --train data/toolalpaca_train_small.jsonl --eval data/toolalpaca_eval_small.jsonl --run-id toolalpaca_smoke --no-progress
```

离线 backend 对 MedQA/ToolAlpaca 只验证数据与实验产物流转，不代表真实 performance。

## OEP 复现实验

OEP attack cases 放在 `data/raw/oep/oep_attack_cases_90.jsonl`，结构是 3 个领域，每个领域 3 组
target rule，每组 10 条 case。先转换成统一 Task JSONL：

```bash
python scripts/prepare_oep.py
```

先用 math 领域的一组 10 条 case 做 smoke：

```bash
python scripts/run_oep_repro.py --mode offline \
  --domain math \
  --attack-cases data/oep/oep_attack_cases.jsonl \
  --eval data/gsm8k_eval_small.jsonl \
  --num-groups 1 \
  --run-id oep_math_smoke \
  --no-progress
```

真实攻击效果需要使用 LLM backend：

```bash
python scripts/run_oep_repro.py --mode llm \
  --domain math \
  --attack-cases data/oep/oep_attack_cases.jsonl \
  --eval data/gsm8k_eval_small.jsonl \
  --num-groups 1 \
  --run-id oep_math_llm_g1
```

`--num-groups 1/2/3` 分别表示注入 10/20/30 条同领域 OEP cases。该脚本流程是：
baseline eval -> batch 注入 attack trajectories -> reflection 生成 memory -> attacked eval。
这对应 OEP 中 “一组 injected cases 经 memory consolidation 形成 memory entry” 的实验语义。
LLM 模式下，reflection 使用 H.2 的 `Reflection and Rule Distillation` 作为 system prompt，
user 内容只包含结构化 ACT incident records；attacked eval 将 OEP memory entry 放入 system prompt，
而不是把整段 raw injection prompt 或 metadata 直接塞给模型。
LLM 运行时会额外保存实际发送给模型的 prompt：

- `llm_prompts.jsonl`: 全部 LLM prompt 事件。
- `baseline_solve_prompts.jsonl`: baseline eval prompt。
- `attacked_solve_prompts.jsonl`: 带 OEP memory 后的 attacked eval prompt。
- `reflection_prompts.jsonl`: 每组 10 条 ACT case 的 reflection prompt。

## 可缓存的四阶段实验入口

`run_m1_demo.py` 和 `run_oep_repro.py` 仍然保留一键复现实验流程。后续做多轮验证时，更推荐把
baseline、memory 生成和 memory evaluation 拆开跑，避免每次验证都重新请求 baseline 或
reflection：

```bash
# 1. 无 memory baseline，只做评测。
python scripts/run_baseline_eval.py --mode llm \
  --eval data/gsm8k_eval_small.jsonl \
  --run-id gsm8k_baseline

# 2. 正常 self-evolution/reflection，只产出 memory，不做 eval。
python scripts/run_reflection.py --mode llm \
  --train data/gsm8k_train_small.jsonl \
  --run-id gsm8k_reflection

# 3. OEP attack-case reflection，只产出注入后的 memory，不做 eval。
python scripts/run_oep_reflection.py --mode llm \
  --domain math \
  --attack-cases data/oep/oep_attack_cases.jsonl \
  --num-groups 1 \
  --run-id oep_math_reflection_g1

# 4. 使用已有 memory 做评测；不会重新跑 reflection。
python scripts/run_memory_eval.py --mode llm \
  --eval data/gsm8k_eval_small.jsonl \
  --memory runs/gsm8k_reflection/memory.jsonl \
  --run-id gsm8k_memory_eval

# 使用 OEP memory 评测时，保持 OEP inference prompt 协议。
python scripts/run_memory_eval.py --mode llm \
  --eval data/gsm8k_eval_small.jsonl \
  --memory runs/oep_math_reflection_g1/memory.jsonl \
  --prompt-protocol oep \
  --run-id oep_math_attacked_eval_g1
```

这样 baseline 结果、benign reflection memory、OEP-injected memory、memory eval 结果都会落在独立
run 目录中，可以分别缓存、复用和对比。

## 断点恢复与重试

所有运行入口都支持显式断点恢复。默认情况下，如果同一个 `run-id` 已经有产物，脚本会提示使用
`--resume` 或 `--overwrite`，避免误把不同实验混在同一个目录里。

```bash
# 网络中断或进程退出后，用相同参数继续未完成部分。
python scripts/run_memory_eval.py --mode llm \
  --eval data/gsm8k_eval_small.jsonl \
  --memory runs/oep_math_reflection_g1/memory.jsonl \
  --prompt-protocol oep \
  --run-id oep_math_attacked_eval_g1 \
  --resume

# 确认要从头重跑同一个 run-id 时，显式覆盖本脚本负责的产物。
python scripts/run_baseline_eval.py --mode llm \
  --eval data/gsm8k_eval_small.jsonl \
  --run-id gsm8k_baseline \
  --overwrite
```

LLM 调用默认最多重试 3 次，退避基准为 2 秒；仍失败时会保留已完成 JSONL 产物并退出：

```bash
python scripts/run_oep_repro.py --mode llm \
  --domain math \
  --attack-cases data/oep/oep_attack_cases.jsonl \
  --eval data/gsm8k_eval_small.jsonl \
  --num-groups 1 \
  --run-id oep_math_llm_g1 \
  --max-retries 5 \
  --retry-backoff-s 3
```

每个 run 目录会写入 `run_config.json` 和 `run_state.json`。`--resume` 会校验关键参数是否与
`run_config.json` 一致；如果要改数据集、domain、memory 文件或其他关键参数，请换新的
`--run-id` 或使用 `--overwrite`。

## 进度显示

demo 默认开启轻量进度事件显示。TTY 下会动态刷新；重定向到日志文件时会自动退化为逐行输出。

```bash
python scripts/run_m1_demo.py --mode offline --train data/m1_train.jsonl --eval data/m1_eval.jsonl --run-id progress_check
python scripts/run_m1_demo.py --mode llm --run-id gsm8k_m1_llm --progress plain
python scripts/run_m1_demo.py --mode llm --run-id gsm8k_m1_llm --no-progress
```

当前实现不是复杂状态机，而是 runner 发出阶段事件，CLI 订阅后显示“正在反思、写入 memory、检索
memory、等待模型响应”等状态。后续接入 OpenClaw/LangChain 时，可以复用同一个事件协议。

## Optional OpenAI-Compatible Backend

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o-mini
python scripts/run_m1_demo.py --mode llm
```

`OPENAI_BASE_URL` can be set for compatible providers.

## 结构

- `safe_se_agent/adapters/base.py`: common tested-agent interface.
- `safe_se_agent/adapters/simple.py`: first complete adapter for Milestone 1.
- `safe_se_agent/core/experiment.py`: framework-agnostic runner.
- `safe_se_agent/core/memory.py`: append-only memory store, retrieval, and JSONL persistence.
- `safe_se_agent/core/defenses.py`: thin wrappers reserved for Milestone 3 defenses.
- `scripts/prepare_gsm8k.py`: 下载并转换 GSM8K small 数据。
- `scripts/prepare_medqa.py`: 下载并转换 MedQA small 数据。
- `scripts/prepare_toolalpaca.py`: 下载并转换 ToolAlpaca small 数据。
- `scripts/prepare_oep.py`: 转换 OEP attack cases。
- `scripts/run_baseline_eval.py`: 只跑 no-memory baseline eval。
- `scripts/run_reflection.py`: 只跑 benign training/reflection 并产出 memory。
- `scripts/run_oep_reflection.py`: 只跑 OEP attack-case reflection 并产出 injected memory。
- `scripts/run_memory_eval.py`: 读取已有 memory 做 eval，不重新生成 memory。
- `scripts/run_oep_repro.py`: OEP-style baseline/injection/attacked eval 复现实验。
- `data/*_train_small.jsonl` and `data/*_eval_small.jsonl`: 默认真实数据 smoke/eval 输入。
- `data/m1_train.jsonl` and `data/m1_eval.jsonl`: synthetic smoke-test benchmark.
- `docs/memory_integration_zh.md`: memory 与外部 agent 框架的接洽说明。

Reserved adapters for OpenClaw and LangChain live in `safe_se_agent/adapters/placeholders.py`.
They intentionally raise `NotImplementedError` until those frameworks are integrated.
