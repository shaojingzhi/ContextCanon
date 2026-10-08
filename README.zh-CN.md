# ContextCanon V2

[English](README.md)

> 在 Agent 执行有副作用的操作之前，将相互冲突的证据编译成精简、可追溯的决策。

## 为什么需要 ContextCanon

ContextCanon V2 将异构证据转换为带类型、可追溯来源的 Claim，并执行确定性的冲突治理。它位于工具调用边界：Agent 可以自由读取信息，但只有知识状态为 `CLEAR` 时才允许执行副作用操作。ContextCanon 不负责 Agent 循环、检索系统或工具后端。

当前实现是精简的 [`contextcanon_v2/`](contextcanon_v2/) 包：约 742 行物理代码，其中 `GovernedToolProxy` 约 88 行。较早的 [`contextcanon/`](contextcanon/) 包是遗留研究原型；实验与测试不属于 V2 核心。

## 架构

```text
知识源 / 运行时工具结果
          |
          v
       Evidence
          |
          v
  SemanticCompiler
          |
          v
    带类型的 Claims
          |
          v
确定性的 govern()
          |
          v
CLEAR / UNRESOLVED / INCOMPLETE
```

两个独立的 Agent 运行时使用同一个治理边界：

```text
OpenAI Agents SDK --\
                     > GovernedToolProxy -> 后端工具
LangGraph ----------/
```

Agent 运行时负责选择工具并控制模型循环。ContextCanon 只负责将成功的 READ 结果捕获为 `Evidence`，编译 SIDE_EFFECT 工具所依赖的事实，并决定分派或阻止该工具。

## 集成证据

### OpenAI Agents SDK + AgentAbstain

一次真实集成 smoke 使用了真实的 OpenAI Agents SDK 循环和真实的 AgentAbstain MCP 工具。Agent 自主选择了 6 次 READ/VERIFY 调用；6 个工具结果全部进入 V2，成为 `Evidence`。Agent 独立发现了活动日期 3 月 22 日与 3 月 23 日的冲突，并在尝试消息发送工具之前主动放弃操作。

这次运行证明了真实的 Agent/MCP 到 Evidence 路径，但它**没有**覆盖真实运行中的 dispatch 前阻断分支，因为 Agent 从未尝试副作用操作。确定性的集成测试分别证明了全部三种门控结果：

- `CLEAR` -> 分派
- `UNRESOLVED` -> 在后端分派前阻止
- `INCOMPLETE` -> 在后端分派前阻止

### LangGraph

[`experiments/v2_langgraph_demo.py`](experiments/v2_langgraph_demo.py) 是第二个独立运行时集成。它使用 LangGraph 的图/模型/工具循环，并调用同一个 `GovernedToolProxy`。该集成没有要求修改 V2 核心。

随后，[live runner](experiments/v2_langgraph_live.py) 显式启用 DeepSeek 完成了 3 轮 LangGraph 工具选择。模型依次选择 `read_policy`、`read_runtime_state`，然后尝试 `deploy_release`；两个运行时结果与静态策略共同组成 3 条 `Evidence`。第 4 次模型调用执行语义编译，之后确定性治理返回 `UNRESOLVED`，并在副作用调用到达后端之前阻止部署。整个运行没有重试或报错，`side_effect_dispatched` 为 false。

## 工程回归过程

冻结的 V2 RAMDocs 运行在 500 个样本上的完成率为 78.8%，共有 106 个语义编译失败。对运行轨迹的分析将大部分失败定位到严格类型表示：82 个日期值和 20 个数值在语义上有效，但不符合确定性校验所接受的形式。

V2.2 在现有严格 Claim 校验之前加入了少量通用规范化逻辑。在 RAMDocs 工程回归中，完成率提升到 91.6%；日期类型值失败从 82 降至 0，数值类型值失败从 20 降至 0。治理逻辑、Gold 映射和评分方式均未改变。

**这是一项基于已用于分析 V2.2 的数据集失败轨迹所做的工程回归，不是未经接触的外部验证。**

完整的真实实验清单、500 条 RAMDocs 结果、Direct 对照、失败口径和 Agent 延迟分解见
[`docs/EXPERIMENT_REPORT.zh-CN.md`](docs/EXPERIMENT_REPORT.zh-CN.md)。

## 与 Guardrail 系统的关注范围

以下是架构关注点的比较，不是基准测试，也不表示某个系统优于另一个系统。

| 系统 | 主要关注点 |
| --- | --- |
| ContextCanon | 证据语义、来源追踪、带类型 Claim，以及冲突感知治理 |
| Invariant Guardrails | 针对 Agent 轨迹、数据流和工具调用的规则策略 |
| NVIDIA NeMo Guardrails | 广泛覆盖输入、检索、对话、执行和输出阶段的护栏 |

这些能力可以互补。简明的集成与对比说明见 [`docs/V2_AGENT_INTEGRATIONS.md`](docs/V2_AGENT_INTEGRATIONS.md)。

## 仓库结构

| 路径 | 用途 |
| --- | --- |
| [`contextcanon_v2/`](contextcanon_v2/) | 当前 V2 模型、编译器、确定性治理和工具代理 |
| [`experiments/agentabstain/`](experiments/agentabstain/) | OpenAI Agents SDK + MCP 集成证明 |
| [`experiments/v2_langgraph_demo.py`](experiments/v2_langgraph_demo.py) | 使用 LangGraph 证明框架无关性 |
| [`benchmarks/`](benchmarks/) | 冻结的开发集、留出集和公共数据集 fixture/adapter |
| [`tests/`](tests/) | 确定性行为与集成覆盖 |
| [`contextcanon/`](contextcanon/) | 为对比保留的遗留研究实现 |

## 离线验证

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q .
```

LangGraph 演示使用可选依赖：

```bash
python3 -m pip install -e '.[langgraph-demo]'
python3 -m unittest tests.test_v2_langgraph_demo -v
```
