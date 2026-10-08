# ContextCanon V2 真实实验报告

更新日期：2026-10-09  
报告范围：ContextCanon V2、V2.1、V2.2，以及同条件 Direct LLM 基线的已完成真实模型实验。

## 一页结论

- 在冻结的 36 条开发集上，ContextCanon 的治理决策完全正确率为 **94.44%**，Direct LLM 为 **38.89%**。这组数据参与过设计迭代，只能作为开发集结果。
- 在 14 条未见留出集上，ContextCanon 的 relation / summary 准确率为 **85.71% / 100%**，Direct LLM 为 **57.14% / 78.57%**。ContextCanon 的 Claim 完全匹配率只有 **64.29%**，说明最终治理正确不等于中间表示完全正确。
- 在 500 条 RAMDocs 公共数据上，冻结 V2 的完成率为 **78.8%**，主要失败来自严格日期和数值表示。冻结 Direct 的完成率为 **99.0%**。
- RAMDocs 失败轨迹用于设计 V2.2 后，ContextCanon 完成率升至 **91.6%**；日期类型失败由 82 降为 0，数值类型失败由 20 降为 0。该结果是**工程回归**，不是未见数据上的泛化证据。
- 真实 Agent 集成方面：OpenAI Agents SDK + AgentAbstain 证明了真实 MCP 结果进入 V2 Evidence；LangGraph 进一步真实执行了 `Evidence -> SemanticCompiler -> govern()`，并在后端调用前阻止了冲突的部署操作。
- 单次 LangGraph smoke 中，语义编译增加约 **2.115 秒**模型时间；纯确定性 compile/govern 微基准仅约 **0.035 毫秒**。前者只有一次样本，不能当作稳定延迟结论。

## 实验边界与口径

所有线上实验均使用 DeepSeek OpenAI-compatible endpoint，且每个 case 都遵循一次模型调用、无 retry、无 repair 的约束；Agent smoke 的模型循环除外，其调用次数由 Agent 运行时决定。本文不重新运行任何模型，只整理已经保存的输出。

不同实验的 Gold 强度不同：

- 36 条开发集与 14 条留出集拥有完整 Claim、relation 和 summary Gold。
- RAMDocs 没有完整 V2 Claim Gold。它机械支持证据相关性；只有 `distinct gold_answers == 1` 的 100 条样本进入 conflict 评估。
- RAMDocs summary 是数据集派生 proxy，不是完整 V2 语义 Gold；其中 `INCOMPLETE` 只有 3 条，不能用于稳健估计。
- “completed-only”文档指标只统计成功得到预测的 case；端到端 case 指标将 provider/schema/semantic failure 计为错误。

## 实验清单

| 阶段 | 规模 / 调用 | 用途 | 结论地位 |
| --- | ---: | --- | --- |
| Semantic Compiler smoke 1 | 5 | 暴露时间 scope、modality、cardinality 的 prompt 歧义 | 开发诊断 |
| Semantic Compiler smoke 2 | 5 | 相同五例的受控前后对比 | 开发回归 |
| 初始 36 条 pilot | 36 | 发现 fixture/contract 偏差并校准 benchmark | 校准数据，不作最终结果 |
| 冻结 V2.1 36 条 | 36 | 完整 Claim 与治理评估 | 冻结开发集结果 |
| 冻结 Direct 36 条 | 36 | 同 evidence 的直接治理基线 | 冻结开发集基线 |
| 未见 holdout | 14 + 14 | ContextCanon 与 Direct 的第一次固定留出比较 | 未见但规模较小 |
| RAMDocs runtime pilot | 20 + 20 | 验证 500 条 runner 的调用、错误和计分链路 | 仅运行验证 |
| 冻结 RAMDocs | 500 + 500 | ContextCanon V2 与 Direct 公共数据评估 | 冻结主结果 |
| RAMDocs failure diagnostic | 106 | 对历史失败行重新采集 raw response | 工程诊断，不替代主结果 |
| RAMDocs V2.2 regression | 500 | 验证通用类型规范化 | RAMDocs-informed 工程回归 |
| AgentAbstain V2 smoke | 1 次 Agent run | 真实 Agents SDK + MCP + Evidence 路径 | 集成证明 |
| LangGraph V2 smoke | 1 次 Agent run，4 次 provider call | 第二运行时与真实 dispatch 前阻断 | 集成与单点延迟观测 |

早期五例 smoke、初始 36 条 calibration pilot 的临时逐条输出当前未保留，因此本文只记录其已确认的开发用途，不补造定量结果。

## 1. 冻结 36 条开发集

冻结 V2.1 与 Direct 都接收相同 Evidence 与 FactNeeds，各自每 case 一次模型调用。

| 指标 | ContextCanon V2.1 | Direct LLM |
| --- | ---: | ---: |
| Relation accuracy | 34/36 = **94.44%** | 17/36 = **47.22%** |
| Summary accuracy | 35/36 = **97.22%** | 26/36 = **72.22%** |
| Decision exact accuracy | 34/36 = **94.44%** | 14/36 = **38.89%** |
| Schema success | 36/36 | 36/36 |
| Mean model latency | 1462.9 ms | 1449.0 ms |
| P50 / P95 latency | 1508.0 / 2052.9 ms | 1240.8 / 2783.3 ms |

结论：在冻结开发集上，显式语义编译加确定性治理明显优于直接让模型输出治理决定；两者平均模型延迟接近。但这 36 条数据经历过 benchmark/prompt 校准，不应表述为外部验证。

## 2. 14 条未见留出集

| 指标 | ContextCanon | Direct LLM |
| --- | ---: | ---: |
| Relation accuracy | 12/14 = **85.71%** | 8/14 = **57.14%** |
| Summary accuracy | 14/14 = **100%** | 11/14 = **78.57%** |
| Governance decision exact¹ | 12/14 = **85.71%** | 7/14 = **50.00%** |
| Claim / full end-to-end exact² | 9/14 = **64.29%** | n/a（Direct 不输出 Claims） |
| Schema / compilation success | 14/14 | 14/14 |
| Mean model latency | 1730.6 ms | 1148.8 ms |
| P50 / P95 latency | 1717.0 / 2758.7 ms | 1071.8 / 2151.5 ms |

¹ relation 与 summary 同时完全匹配；² Claim 完全匹配，且 relation 与 summary 同时完全匹配。

ContextCanon 的 14 条分类为：9 `PASS`、2 `CLAIM_SEMANTICS_ERROR`、3 `CLAIM_WRONG_GOVERNANCE_RIGHT`、0 deterministic governance error、0 compilation error。Claim 字段中最弱的是 scope（17/24，70.83%）和 modality（18/24，75.00%）。

结论：小型未见集保持了治理层优势，但样本量只有 14；尤其应同时报告 Claim 表示错误，不能只看 summary 的 14/14。

## 3. RAMDocs 500 条冻结公共数据实验

### 3.1 数据与机械评估目标

- 500 个问题，2,766 份文档。
- Gold relevant evidence：`correct` 或 `misinfo`；Gold irrelevant evidence：`noise`。
- Conflict Gold：同时存在 correct 与 misinfo。
- 为避免多合法答案造成伪冲突，仅 100 条单一 gold answer 样本进入 conflict 评估。
- Summary proxy：无 correct 为 `INCOMPLETE`，correct + misinfo 为 `UNRESOLVED`，仅 correct 为 `CLEAR`。

### 3.2 冻结 V2 与冻结 Direct

| 指标 | ContextCanon V2 | Direct LLM |
| --- | ---: | ---: |
| Attempted / provider calls | 500 / 500 | 500 / 500 |
| Completed / failed | 394 / 106 | 495 / 5 |
| Completion rate | **78.80%** | **99.00%** |
| Relevant precision¹ | 94.77% | 95.82% |
| Relevant recall¹ | **70.66%** | 63.59% |
| Correct recall¹ | **69.51%** | 61.60% |
| Misinfo retention¹ | **77.69%** | 75.99% |
| Noise rejection¹ | 84.16% | **88.60%** |
| Eligible completed cases | 89 | 99 |
| Competing accuracy¹ | 73.03% | 78.79% |
| Competing precision¹ | 74.00% | **85.11%** |
| Competing recall¹ | **77.08%** | 74.07% |
| Competing F1¹ | 75.51% | **79.21%** |
| TP / FP / FN / TN | 37 / 13 / 11 / 28 | 40 / 7 / 14 / 38 |
| Eligible competing accuracy² | 65/100 = **65.00%** | 78/100 = **78.00%** |
| All-attempted summary proxy² | 227/500 = **45.40%** | 311/500 = **62.20%** |
| Eligible summary proxy² | 62/100 = **62.00%** | 74/100 = **74.00%** |
| Mean latency³ | 2275.9 ms | 1643.3 ms |
| P50 / P95 latency³ | 2172.2 / 3253.8 ms | 1546.3 / 2350.6 ms |

¹ completed-only；² 所有 attempted case，失败计错；³ 所有 500 次调用。

冻结 V2 的 106 个失败为：82 个日期 ISO 校验失败、20 个非数值表示失败、4 个无效 JSON。Direct 的 5 个失败均为未知 evidence reference。

核心结论：冻结 V2 在成功编译的样本上有更高的 relevant/correct recall 与 misinfo retention，但较低的 noise rejection；大量类型表示失败显著拖累了端到端 conflict 和 summary proxy。冻结 Direct 的可靠完成率更高，并在端到端 conflict、summary proxy 和延迟上领先。

### 3.3 原始文件完整性

冻结 V2 因运行会话中断被保存为两段，但没有重跑任何行：

- `/tmp/contextcanon_ramdocs_full500.jsonl`：row 0–336，共 337 条；SHA-256 `35f656c0a614680a9652f78dfe2ecabb5bc8a338285548352b5da29b6cb096f6`
- `/tmp/contextcanon_ramdocs_tail163.jsonl`：row 337–499，共 163 条；SHA-256 `d16119bced4e866bd6b19657994ea173737beb6cde2c4f2c80db57313af083bc`
- `/tmp/direct_ramdocs_full500.jsonl`：500 条 + aggregate；SHA-256 `d8b5c3c45d20bf3dd098dfd5ff72577e0dd3f2743bb90112111e28548b3f91ba`

`/tmp` 文件是本机实验产物，不随 Git 提交；本报告保存其聚合数据与校验摘要。

## 4. RAMDocs 失败诊断与 V2.2 工程回归

### 4.1 106 条历史失败诊断

为采集历史失败所缺少的 raw response，对原 106 个失败 row 各执行一次诊断请求：106 attempted，9 completed，97 failed，无 retry。剩余失败包括 74 个日期表示、18 个数值表示、4 个 JSON 解析、1 个 transport failure。

观察到的通用表示包括：

- 明确英文完整日期，如 `1 December 1737`、`2 July 1829`；
- 年份粒度日期，如 `1856`、`1902`；
- 千分位整数，如 `1,066`、`120,000`；
- 带近似语义的量，如 `about 50,000`；
- 结构未闭合或转义错误的 JSON。

V2.2 只加入通用、确定性的安全规范化；不把模糊日期强行发明成某一天，不修复任意 JSON，也没有第二次 LLM 调用。

### 4.2 V2.2 500 条回归

| 指标 | 冻结 V2 | V2.2 工程回归 | 冻结 Direct |
| --- | ---: | ---: | ---: |
| Completion | 78.80% | **91.60%** | 99.00% |
| Failed | 106 | **42** | 5 |
| Date typed-value failures | 82 | **0** | n/a |
| Number typed-value failures | 20 | **0** | n/a |
| Relevant precision¹ | 94.77% | 95.07% | 95.82% |
| Relevant recall¹ | 70.66% | **71.15%** | 63.59% |
| Correct recall¹ | 69.51% | **69.98%** | 61.60% |
| Misinfo retention¹ | 77.69% | **78.52%** | 75.99% |
| Noise rejection¹ | 84.16% | 84.82% | **88.60%** |
| Competing F1¹ | 75.51% | **80.00%** | 79.21% |
| Eligible competing accuracy² | 65.00% | **76.00%** | 78.00% |
| All-attempted summary proxy² | 45.40% | **55.00%** | 62.20% |
| Eligible summary proxy² | 62.00% | **76.00%** | 74.00% |
| Mean latency³ | 2275.9 ms | 2302.3 ms | **1643.3 ms** |
| P50 / P95 latency³ | 2172.2 / 3253.8 ms | 2139.5 / 4018.0 ms | 1546.3 / 2350.6 ms |

¹ completed-only；² 所有 attempted case；³ 所有 500 次调用。

V2.2 的 42 个失败为 39 个 semantic completion/provider failure 和 3 个无效 JSON；日期与数值 typed-value failure 均为 0。治理、RAMDocs Gold 映射、eligibility 和评分未改变。

**解释限制：V2.2 是在查看 RAMDocs failure trace 后设计的，因此 V2→V2.2 的改善证明工程恢复效果，不证明未见数据泛化。Direct 没有修改，所以没有重跑。**

## 5. 真实 Agent 运行时实验

### 5.1 OpenAI Agents SDK + AgentAbstain

- 真实 OpenAI Agents SDK 循环和真实 MCP 工具。
- Agent 依次执行 6 个 READ/VERIFY 工具；6 个结果全部成为 V2 Evidence。
- Agent 识别出 2026-03-22 与 2026-03-23 冲突并主动 abstain。
- `side_effect_attempted=false`，因此这次 live run 没有触发 ContextCanon 的 dispatch 前阻断分支。
- 模型请求数 5；input/output/total tokens 为 48,909 / 3,210 / 52,119。

该实验真实证明了 Agent SDK -> MCP -> Evidence 的路径，但不能声称 live blocking 已被覆盖；`CLEAR -> dispatch`、`UNRESOLVED -> block`、`INCOMPLETE -> block` 由确定性集成测试证明。

### 5.2 LangGraph

- LangGraph 自己控制模型循环，没有复用 OpenAI Agents SDK adapter。
- 模型选择顺序：`read_policy` -> `read_runtime_state` -> `deploy_release`。
- 两个运行时读取结果与一条静态策略合并为 3 条 Evidence。
- SemanticCompiler 生成 `stable/REQUIRED` 与 `canary/OBSERVED` Claims；`govern()` 返回 `UNRESOLVED`。
- `deploy_release` 在到达 backend 前被阻止；`side_effect_dispatched=false`。
- 3 次 Agent 模型调用 + 1 次语义编译，共 4 次 provider call；总 token 1,584；无 retry、无错误。

该 live smoke 是刻意设计的路径覆盖实验：prompt 要求模型读取两个来源后，即使发现冲突也尝试 `deploy_release`，以便由治理边界作出决定。它证明了真实模型选择 -> LangGraph runtime -> READ 结果成为 Evidence -> side effect 到达 `GovernedToolProxy` -> `SemanticCompiler` -> 确定性治理 -> backend 被阻止的完整链路；它不证明一个不受该指令约束的 Agent 在发现冲突后仍会自然尝试不安全的副作用。

#### 单次延迟分解

| 阶段 | 延迟 |
| --- | ---: |
| Agent 工具选择 1 | 1291.5 ms |
| Agent 工具选择 2 | 925.5 ms |
| Agent 工具选择 3 | 1606.8 ms |
| Agent 模型路径合计 | 3823.7 ms |
| ContextCanon SemanticCompiler | **2115.2 ms** |
| Provider 总时间 | 5939.0 ms |
| 确定性 compile/govern 微基准均值 | 0.0349 ms |

以本次单点运行计算，治理语义编译相对 Agent 模型路径增加约 **55.3%** 时间，占 provider 总时间约 **35.6%**；额外消耗 950 tokens。确定性治理本身可忽略，主要成本来自额外的语义编译模型调用。

这只是一个成功 smoke，不是延迟 benchmark。网络波动、模型缓存和输出长度都可能显著影响结果；可靠延迟结论需要固定输入的 paired A/B、多次重复和置信区间。

## 6. 综合结论

1. **确定性治理规则本身稳定。** 完整 Gold benchmark 与集成测试没有暴露治理层错误；主要误差集中在 LLM 语义编译和输出表示。
2. **结构化中间层改善了小型治理任务。** 36 条开发集和 14 条留出集都显示 ContextCanon 的 relation/summary 优于 Direct，但留出集规模太小，开发集又经历过校准。
3. **公共数据结果更复杂。** RAMDocs 上 ContextCanon 更愿意保留正确与 misinformation 证据，但 Direct 的完成率、noise rejection、冻结端到端 conflict 和 summary proxy 更好。
4. **V2.2 解决了明确的工程鲁棒性缺口。** 它消除了该回归中的日期/数值 typed-value failure，但不能被包装成新的外部验证结果。
5. **治理延迟主要是一次额外 LLM 调用。** 纯确定性部分几乎没有成本；是否值得取决于副作用风险、可缓存性以及是否能在读取阶段提前编译。
6. **框架无关性已有两条实证路径。** OpenAI Agents SDK 证明真实 MCP/Evidence 捕获，LangGraph 证明同一个 `GovernedToolProxy` 可在另一运行时完成真实阻断。

## 7. 尚未回答的问题

- V2.2 尚需一个完全未参与设计的公开数据集做外部验证。
- RAMDocs 不支持完整 scope、modality、cardinality Gold，不能回答全部 V2 Claim 质量问题。
- 14 条 holdout 太小，无法给出窄置信区间。
- LangGraph 的治理增量延迟只有一次观测；尚无 paired A/B 延迟分布。
- AgentAbstain live run 中 Agent 自主 abstain，真实 MCP side-effect 的阻断分支只在 LangGraph live smoke 中被覆盖。

## 8. 可追溯提交与产物

| 内容 | 版本 / 产物 |
| --- | --- |
| 冻结 V2 RAMDocs | `aa69e98`；两个分段 JSONL |
| Raw-response diagnostic | `48d9c08`；`/tmp/contextcanon_ramdocs_failure_diagnostic106.jsonl` |
| V2.2 normalization | `d8c395c`；`/tmp/contextcanon_ramdocs_v22_full500.jsonl` |
| Framework-neutral proxy | `373655b` |
| AgentAbstain V2 runner | `54aee8d`；`/tmp/contextcanon_v2_agent_smoke.json` |
| LangGraph offline integration | `8f2ab7a` |
| LangGraph live smoke | `80f9047`；`/tmp/contextcanon_v2_langgraph_live.json` |

报告中的百分比保留两位小数；更精确的 numerator/denominator 优先于四舍五入后的百分比。
