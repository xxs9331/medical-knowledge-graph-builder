# 宽松候选准入与独立 LLM Judge 设计问题

日期：2026-08-12

状态：第一阶段已实现，LLM Judge 尚未接入。

当前候选构建器已实现宽松准入、图内 `PARTIAL`、独立 `judge-queue.json`、四阶段
最多两次调用和发布前复验边界；尚未调用 Judge，也不能将 Judge 队列中的草稿作为图
候选或已批准医学知识。所有产物仍为 `candidate-only`、`publication_status=HOLD`。

## 问题

当前候选准入器同时承担来源回放、Schema/端点约束和部分图结构约束。若将所有
不符合预期的候选直接 `REJECTED`，可能因实体漏抽、模型表达差异、表格或规则的
复杂结构，过早丢失医学语义正确的候选。另一方面，若完全跳过本地校验并相信模型，
则无法保证候选确实来自原文、端点确实存在，也不能稳定复现审计结论。

要解决的问题是：如何提高候选阶段召回率，由大模型审查语义，同时保留原文可回放
和最终发布前的确定性约束？

## 已确认方向

采用“宽松候选准入 + 独立 LLM Judge + 发布前复验”的分层流程：

```text
LLM 抽取
  -> 宽松候选准入
  -> 独立 LLM Judge
  -> 修复、补抽或拒绝
  -> 发布前确定性复验
  -> 人工批准发布
```

1. 候选阶段的目标是保留可审计的发现，不应使用有限关键词表裁决医学语义。
2. 表达不唯一、端点暂未解析、关系方向不确定、疑似联合条件、规则输入输出不全等
   情况，优先标为 `PARTIAL` 和 `REVIEW_REQUIRED`，而不是直接 `REJECTED`。
3. 独立 LLM Judge 负责判断实体分类、关系类型、RuleDefinition 语义，以及给出的
   原文证据是否支持候选。Judge 必须只使用提供的原文和候选上下文，不得用外部医学
   知识补造端点或证据。
4. Judge 的修复意见必须生成新的候选版本，不能静默覆盖抽取结果。
5. 发布前仍必须由程序复验：原文位置、chunk hash、端点存在性、Schema domain/range、
   以及规则图结构。任何未通过复验的记录不得发布。
6. LLM Judge 只能路由和辅助审核，不能将候选自动标记为正式医学知识；最终发布仍需
   人工批准。

## 候选阶段的状态路由

| 情况 | 候选阶段处理 | Judge 后的处理 |
| --- | --- | --- |
| JSON 无法解析或模型调用失败 | 重试；失败后记录拒绝原因 | 不进入语义审核 |
| 原文引语重复、位置不唯一 | `PARTIAL`，请求 Judge 定位 | 给出位置后程序必须能回放，否则拒绝 |
| 端点不在冻结目录、端点类型不符 | `PARTIAL`，请求补抽或重映射 | 修复后的端点必须通过 Schema 复验 |
| 关系方向与词序不一致、疑似联合条件 | `PARTIAL`，交 Judge 判定语义结构 | Judge 结论需转为可复验的关系或规则结构 |
| 规则输入、输出或边不完整 | `PARTIAL`，交 Judge 诊断或补抽 | 仍不完整则拒绝发布 |
| 引语无法在源 chunk 中定位 | `REJECTED` | 不允许 Judge 以推测方式补造原文 |

“不可定位的原文”保留为硬拒绝，因为它不再是可审计候选；其余当前带有结构假设的
拒绝条件应优先降级为 `PARTIAL`，并由实验检验误放与误拒。

## Judge 输入与输出合同

Judge 输入必须包括：

- 完整 EvidenceChunk 原文及 chunk hash；
- 候选节点、关系或 RuleDefinition；
- Schema 中允许的实体和关系类型；
- 已冻结实体目录及候选阶段的 warning；
- 该候选声明的原文锚点，或待定位的 mention/exact quote。

Judge 输出固定为一个 JSON 对象，至少包含：

```json
{
  "verdict": "SUPPORTED | UNSUPPORTED | REPAIR | ABSTAIN",
  "reason": "简短、可审计的原因",
  "evidence_spans": [
    {"source_char_start": 0, "source_char_end": 0}
  ],
  "repair_instruction": "仅在 REPAIR 时给出"
}
```

- `SUPPORTED`：语义支持候选，但仍需程序复验全部证据位置和图结构。
- `UNSUPPORTED`：原文不支持候选，记录拒绝原因。
- `REPAIR`：原文可能支持，但候选端点、类型、位置或规则结构需要重建。
- `ABSTAIN`：Judge 无法可靠判断，保留给人工审核，不自动发布。

Judge 应与抽取模型独立调用；实验中应评估同模型、不同模型和多 Judge 投票的差异。

## 待验证问题

1. 将“端点类型不符”“方向不一致”“疑似联合条件”从 `REJECTED` 降级到 `PARTIAL`
   后，候选正确召回率是否提高，人工审核负担增加多少？
2. Judge 对实体、普通关系和 RuleDefinition 的语义精确率、召回率和一致性分别是多少？
3. Judge 给出的证据位置经过程序复验后的成功率是多少？失败原因如何分布？
4. 同一模型担任抽取器与 Judge 时会不会产生确认偏差？独立模型或多 Judge 是否改善？
5. 对表格、公式、否定、并列和联合条件，哪些 `PARTIAL` 原因最常被有效修复？
6. `ABSTAIN` 的比例和人工复核耗时是否可接受？
7. 是否应允许 Judge 触发“实体补抽”闭环；如何限制回合数、防止无限扩张？

## 评价指标与实验要求

在同一批人工金标 EvidenceChunk 上，对“当前严格准入”“宽松准入但无 Judge”与
“宽松准入加 Judge”进行比较。至少报告：

- 候选保留召回率：金标正确候选在候选阶段未被丢弃的比例；
- 硬校验误拒率：金标正确候选被直接 `REJECTED` 的比例；
- Judge 的语义精确率、召回率和 F1；
- Judge 误放率、误拒率和 `ABSTAIN` 比例；
- 发布前原文回放成功率：必须为 100%；
- `PARTIAL` 原因分布、人工复核耗时、每 chunk 模型调用数和 token 成本。

实验数据、模型版本、提示词、温度、Judge 输入、原文锚点和人工标注准则必须固定并
归档。不得使用最终图规模替代正确性指标，不得将未通过人工发布门的候选用于诊断或
报告推理。

## 参考文献

1. Huang, H., Chen, C., Sheng, Z., Li, Y., & Zhang, W. (2025). *Can LLMs be
   Good Graph Judge for Knowledge Graph Construction?* In *Proceedings of
   EMNLP 2025* (pp. 10929-10948). Association for Computational Linguistics.
   https://doi.org/10.18653/v1/2025.emnlp-main.554

   该工作将图谱抽取与后置 LLM Judge 分离，用 Judge 提升生成图谱质量；支持将
   “发现候选”和“语义审查”拆开，但不证明任一具体 Judge 足以取代人工审核。

2. Zhang, B., & Soh, H. (2024). *Extract, Define, Canonicalize: An LLM-based
   Framework for Knowledge Graph Construction.* In *Proceedings of EMNLP 2024*
   (pp. 9820-9836). Association for Computational Linguistics.
   https://doi.org/10.18653/v1/2024.emnlp-main.548

   该工作采用开放抽取、Schema 定义和后置规范化的多阶段流程，说明抽取后再处理
   类型与规范化是可行的工程路径。

3. Adam, D., & Kliegr, T. (2024). *Traceable LLM-based validation of statements
   in knowledge graphs.* arXiv:2409.07507. https://arxiv.org/abs/2409.07507

   该工作要求验证依赖外部可追溯文本，而非 LLM 内部知识；在 BioRED 派生数据上报告
   precision 88%、recall 44%，作者结论仍需要人工监督。因此本项目不能以 Judge 结论
   替代证据回放或人工发布门。

4. Regino, A. G., & dos Reis, J. C. (2025). *Can LLMs be Knowledge Graph
   Curators for Validating Triple Insertions?* In *Proceedings of GenAIK 2025*
   (pp. 87-99). Association for Computational Linguistics.
   https://doi.org/10.18653/v1/2025.genaik-1.10

   该工作把类型/属性对齐、URI 标准化、语义一致性和语法正确性作为不同验证任务，
   并指出领域泛化、语义漂移和 human-in-the-loop 仍是部署问题。

5. Yang, S., et al. (2026). *AutoSchemaKG: Autonomous Knowledge Graph
   Construction through Dynamic Schema Induction from Web-Scale Corpora.*
   In *Proceedings of ACL 2026*. Association for Computational Linguistics.
   https://aclanthology.org/2026.acl-long.942/

   该工作在附加分析中使用多个 Judge 交叉验证抽取质量，可作为本项目比较单 Judge 与
   多 Judge 的设计参考。其任务和数据集不同，不能外推具体性能数值。

## 与现有文档的关系

- [分阶段与联合抽取的待决问题](llm-extraction-pipeline-vs-joint-literature-2026-08-12.md)
  讨论模型输出应分阶段还是联合生成。
- 本文讨论无论采用何种抽取编排，候选如何宽松保留、如何由 Judge 评测、以及发布前
  哪些约束不可放宽。
- 当前代码的中文说明位于
  `src/medical_kg_sourceprep/extraction/graph_builder/validation/`；在本设计实现前，
  宽松 `PARTIAL` 路由和 Judge 输入队列已经实现；Judge 合同中的模型调用、修复候选
  和复验闭环仍未实现，不得描述为已有能力。
