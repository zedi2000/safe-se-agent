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

## 真实数据集准备

OEP 论文中的三个基础数据源目前都可以转换成统一的 `Task` JSONL schema：

```bash
python scripts/prepare_gsm8k.py --limit-train 20 --limit-eval 20
python scripts/prepare_medqa.py --limit-train 20 --limit-eval 20
python scripts/prepare_toolalpaca.py --limit-train 20 --limit-eval 20
```

对应 smoke test：

```bash
python scripts/run_m1_demo.py --mode offline --train data/gsm8k_train_small.jsonl --eval data/gsm8k_eval_small.jsonl --run-id gsm8k_smoke --no-progress
python scripts/run_m1_demo.py --mode offline --train data/medqa_train_small.jsonl --eval data/medqa_eval_small.jsonl --run-id medqa_smoke --no-progress
python scripts/run_m1_demo.py --mode offline --train data/toolalpaca_train_small.jsonl --eval data/toolalpaca_eval_small.jsonl --run-id toolalpaca_smoke --no-progress
```

离线 backend 对 MedQA/ToolAlpaca 只验证数据与实验产物流转，不代表真实 performance。

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
- `data/*_train_small.jsonl` and `data/*_eval_small.jsonl`: 默认真实数据 smoke/eval 输入。
- `data/m1_train.jsonl` and `data/m1_eval.jsonl`: synthetic smoke-test benchmark.
- `docs/memory_integration_zh.md`: memory 与外部 agent 框架的接洽说明。

Reserved adapters for OpenClaw and LangChain live in `safe_se_agent/adapters/placeholders.py`.
They intentionally raise `NotImplementedError` until those frameworks are integrated.
