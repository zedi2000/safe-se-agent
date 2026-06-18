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
python scripts/run_m1_demo.py --mode offline
pytest -q
```

离线 demo 是确定性的。它先用一个故意简化的 baseline arithmetic solver 解题，然后从失败轨迹中
反思出可复用 memory rule，再在相似评测题上召回这些 rule，展示准确率提升。

每次 demo 默认写入：

- `runs/<run_id>/memory.jsonl`
- `runs/<run_id>/baseline_results.jsonl`
- `runs/<run_id>/self_evolution_results.jsonl`
- `runs/<run_id>/summary.json`

如果测试真实 LLM，普通算术题通常太容易，baseline 无 memory 也可能做对。可以用私有规则
数据集观察 memory 增益：

```bash
python scripts/run_m1_demo.py --mode llm --preset private_protocol --run-id m1_protocol_llm
```

## 进度显示

demo 默认开启轻量进度事件显示。TTY 下会动态刷新；重定向到日志文件时会自动退化为逐行输出。

```bash
python scripts/run_m1_demo.py --mode offline --preset arithmetic --run-id progress_check
python scripts/run_m1_demo.py --mode offline --preset arithmetic --progress plain
python scripts/run_m1_demo.py --mode offline --preset arithmetic --no-progress
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
- `data/m1_train.jsonl` and `data/m1_eval.jsonl`: tiny local benchmark.
- `docs/memory_integration_zh.md`: memory 与外部 agent 框架的接洽说明。

Reserved adapters for OpenClaw and LangChain live in `safe_se_agent/adapters/placeholders.py`.
They intentionally raise `NotImplementedError` until those frameworks are integrated.
