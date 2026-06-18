# Memory 与 Agent 框架接洽说明

本项目的实验层只依赖 `AgentAdapter`，不直接假设某个 agent 框架的内部 memory
实现。这样做的目的不是把 memory 从 agent 中“拆出去”，而是给不同框架一个统一的
实验入口。

## 两种接入模式

### native 模式

如果目标框架已经有可控、可导出的 memory 机制，例如 OpenClaw skill memory 或
LangChain memory/retriever，adapter 应该优先桥接框架原生能力：

- `retrieve(task, k)` 映射到框架原生检索接口。
- `add_memory(entries)` 映射到框架原生写入或 consolidation 接口。
- `export_memory()` 映射到框架原生导出接口，用于 ESR、debug 和防御评估。

这种模式最贴近真实被测框架。

### sidecar 模式

如果目标框架的 memory 不方便控制、导出，或当前实验只需要验证 memory 机制，可以让
adapter 使用本项目的 `MemoryStore + JsonlMemoryBackend` 作为旁路 memory：

- agent 推理仍由目标框架完成。
- 实验 memory 由 adapter 维护。
- runner 仍然只调用统一 `AgentAdapter`。

Milestone 1 的 `SimpleAgentAdapter` 使用的就是 sidecar 风格，但它仍然代表 Simple
Agent 自己的 memory 状态，而不是全局共享服务。

## 当前默认选择

Milestone 1 默认使用 JSONL 持久化，路径为 `runs/<run_id>/memory.jsonl`。JSONL 比
SQLite 更适合当前阶段，因为 memory 数量小，而且团队需要直接检查、diff 和复现实验。
如果后续 memory 规模或查询复杂度上升，可以在不改 runner 的情况下新增 SQLite backend。
