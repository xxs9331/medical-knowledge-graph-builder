---
title: Semantica 抽取质量评测与测试机制
aliases:
  - Semantica Extraction Evaluation
tags:
  - 知识图谱
  - 实体识别
  - 关系抽取
  - 抽取评测
  - Semantica
  - 外部项目调研
source: https://github.com/semantica-agi/semantica
reviewed: 2026-08-14
---

# Semantica 抽取质量评测与测试机制

## 1. 结论

当前 Semantica 版本**没有已经实现的、基于人工标注 gold dataset 的实体识别/关系抽取评测框架**。

项目中存在三类容易混淆的内容：

1. `ExtractionValidator`：对抽取结果做结构、置信度和简单一致性检查，不知道标准答案，因此不能计算真实的 precision、recall、F1。
2. `tests/semantic_extract/`：主要是单元测试、固定样例测试、fallback 测试、mock/provider 测试和性能测试，验证代码行为，不验证领域语义正确率。
3. `semantica.evals`：文档规划了 `ExtractionEvaluator`、NER precision/recall/F1、关系评测和回归跟踪，但当前模块仍标记为 `coming_soon`，实际没有可调用实现。

因此，运行 `pytest` 通过，或者 `ExtractionValidator` 返回高分，都不能说明实体和关系抽取准确。要评测抽取质量，必须额外准备人工标注数据，并将预测结果和 gold 标注按明确匹配规则比较。

## 2. 当前版本实际具备什么

检查版本：`c0a051903f5eb58b9fab6da0983fe3ffe909034f`（2026-08-14，`main`）。

### 2.1 `ExtractionValidator` 是内部质量检查，不是 gold 评测

实体校验目前检查：

- 是否低于 `min_confidence`，低置信度只产生 warning；
- 是否有空实体文本；
- 实体总数、唯一文本数、类型数、高/中/低置信度数量；
- 平均置信度；
- 一个由置信度和简单比例惩罚组成的内部 score。

关系校验目前检查：

- 是否低于置信度阈值；
- subject/object 是否为空；
- subject 和 object 是否是同一个文本；
- 关系数量、关系类型数、平均置信度和无效关系数量；
- 一个由置信度和无效关系比例组成的内部 score。

这些指标只使用预测结果本身。例如，模型漏掉了一个应该抽取的疾病实体，只要剩余实体不为空且置信度高，内部 score 仍可能很高。它不能发现 false negative，也不能确认某个预测是否符合原文语义。

源码：[ExtractionValidator 实现](https://github.com/semantica-agi/semantica/blob/c0a051903f5eb58b9fab6da0983fe3ffe909034f/semantica/semantic_extract/extraction_validator.py#L73-L298)。

### 2.2 `semantica.evals` 目前只是规划

官方评测文档明确将 `semantica.evals` 标记为“not yet implemented”，模块的 `__all__` 为空。规划中的接口包括：

- `ExtractionEvaluator`：NER precision/recall/F1 和关系抽取指标；
- `KGEvaluator`：完整性、一致性、schema 合规、覆盖率和孤儿节点；
- `PipelineBenchmark`：吞吐、延迟、内存和错误率；
- `RegressionTracker`：跨提交或配置的回归比较。

这些是未来 API 设计，不是当前可执行的评测能力。当前文档提供的 `OntologyEvaluator` 只能评估本体覆盖和完整性，不能替代实体/关系抽取评测。

源码：[evals 占位模块](https://github.com/semantica-agi/semantica/blob/c0a051903f5eb58b9fab6da0983fe3ffe909034f/semantica/evals/__init__.py)，[官方 evals 文档](https://github.com/semantica-agi/semantica/blob/c0a051903f5eb58b9fab6da0983fe3ffe909034f/docs/reference/evals.md)。

### 2.3 行业中知识图谱一般怎样评测

知识图谱通常不会有一个可替代所有判断的“总分”。评测必须按图谱的目标和生命周期分层：文本抽取关注“从原文抽得对不对”；图谱数据质量关注“图中的事实是否正确、完整、一致、可追溯”；本体关注“模型能否支持业务问题”；图谱补全模型关注“是否能把正确候选排在前面”；最终应用关注“是否真正改善了问答、检索或临床任务”。

知识图谱质量控制综述把常见数据质量维度归纳为 accuracy、completeness、consistency、timeliness、trustworthiness 和 availability；不同领域会再加入 provenance、relevance 或 accessibility。因此应把它们作为一个指标面板，而不是压缩为单一分数。[知识图谱质量控制综述](https://www.sciencedirect.com/science/article/pii/S2667325821001655)

| 评测层 | 回答的问题 | 常用方法与指标 | 不能说明什么 |
| --- | --- | --- | --- |
| 文本抽取 | 实体、关系、属性是否从原文抽对 | 人工 gold 标注；严格/宽松 span+type 的 P/R/F1；有向 typed-relation F1；evidence-supported rate | 图谱是否完整、可查询或对业务有用 |
| 实体链接与规范化 | 提及是否链接到正确规范实体 | link accuracy；mention+KB-ID P/R/F1；同义词合并的 pairwise/B-cubed F1 | 原文是否支持关系，或关系方向是否正确 |
| 图谱事实质量 | 每条三元组是否真实、是否缺失、是否互相冲突 | 抽样人工核验的 precision/accuracy；完整度；冲突率；约束违例率；来源覆盖率；更新时间分布 | 模型在隐藏边预测上的排名能力 |
| 本体与 schema | 类、属性、约束是否满足业务需求 | competency-question answer rate；OWL 推理一致性；SHACL conform rate；schema coverage | 已有事实一定真实或完整 |
| 图谱补全/KGE | 缺失 head/tail/relation 是否排在前面 | filtered MRR、MR、Hits@1/3/10；有时是 AUC、AP、P@K | 新预测事实已被现实世界证实 |
| 查询/问答/检索 | 图谱是否改善终端任务 | query answer EM/F1；检索 Recall@K、nDCG；任务成功率；人工可用性/安全性审核 | 上游每条边都没有错误 |
| 工程运行 | 系统是否稳定、可复现、成本可接受 | 吞吐、P50/P95 延迟、错误率、内存、成本、可复跑率、版本漂移 | 抽取或图谱语义正确 |

这也是为什么 Semantica 规划的四个 evaluator 不应被视作同一类指标：

| Semantica 规划接口 | 对应行业层 | 应产出的核心证据 |
| --- | --- | --- |
| `ExtractionEvaluator` | 文本抽取、实体链接 | gold 对齐、P/R/F1、按类型错误样例、关系方向和证据错误 |
| `KGEvaluator` | 图谱事实质量、schema/结构质量 | 抽样事实核验、完整度口径、冲突/孤儿/断边、SHACL 违例、来源覆盖 |
| `PipelineBenchmark` | 工程运行 | 端到端吞吐、P50/P95、资源、成本、失败重试和可复现性 |
| `RegressionTracker` | 持续质量治理 | 固定数据集和版本下的指标差异、显著退化告警、可追溯 run artifact |

#### 2.3.1 抽取评测：gold 标注优先，P/R/F1 只是起点

对于文本到知识图谱的构建，最常见的主评测是拿冻结的人工标注样本做 partial gold standard。关系/实体预测与 gold 比较，报告 precision、recall 和 F1；对于错误检测任务，也常加 accuracy 或 AUC。知识图谱 refinement 的综述明确区分了 partial gold、以现有图谱作 silver standard、以及对新增候选做人工 retrospective evaluation 三种路线，并指出 silver standard 在开放世界图谱中会把真实但尚未收录的事实误记为 false positive。[Cimiano 与 Paulheim 的评测方法综述](https://journals.sagepub.com/doi/10.3233/SW-160218)

实际报告至少应分开给出：

- strict entity span+type micro/macro F1；
- exact directed typed-relation F1；
- 关系证据支持率和无证据关系数；
- 关键类别/关系的 P、R、F1，而不是只给总体 micro F1；
- 误报和漏报样本，以及两名人工标注者的一致性（如 Cohen's kappa）或裁决过程；
- `VALID`、`PARTIAL`、`REJECTED`、人工复核的数量与比例。

对于医疗图谱，关系的方向、否定、时间、数值和单位往往比普通边的平均 F1 更重要。比如“指标值高于上限”必须同时验证指标、数值、单位、参考区间、比较方向和原文证据；只把两个实体连成 `related_to` 不应得分。

#### 2.3.2 图谱数据质量：准确、完整、一致、时效、来源和可用性分开测

图谱事实质量通常由数据质量维度组成，而不是由模型置信度代替：

- **准确性/正确性**：对抽样三元组回到原始权威来源或由领域专家判定；可报告 triple precision、attribute accuracy、错误率和关键事实 false-positive rate。
- **完整性**：先固定“应该有什么”的宇宙或最小必填集合，再计算已覆盖比例，例如有数值的检验项中含 unit、reference_range 和 conclusion 的比例。没有明确分母时，节点/边数量不是完整度。
- **一致性**：检查逻辑矛盾、类型/域值冲突、重复但互相矛盾的时间事实，以及跨源冲突。它证明结构和规则不冲突，不证明事实本身为真。
- **时效性**：报告事实的来源日期、有效期、更新延迟和过期比例；尤其要将检查日期、记录日期和知识库入库日期区分开。
- **可信性/溯源**：报告具有可回放来源、逐字证据、source hash、抽取模型/规则版本的节点和边比例；将“有 URL”与“可核验证据”分开。
- **可用性**：测数据/服务可访问性、查询成功率、许可合规和恢复能力。它是平台质量，不是事实正确率。

完整性综述也强调，完整性与 accuracy、timeliness、provenance、accessibility 等并列，是适用性的一部分；不同文献所用的完整性口径并不相同，必须在项目内冻结定义和分母。[知识图谱完整性系统综述](https://doi.org/10.1109/ACCESS.2021.3056622)

#### 2.3.3 本体和 schema：CQ、推理与 SHACL 是不同门

本体评测至少有三个互补门：

1. **Competency questions（CQ）**：把业务问题写成可执行 SPARQL/查询断言，报告图谱能正确回答的 CQ 比例和失败原因。它评估是否“适合用途”，不是抽取正确率。
2. **逻辑推理一致性**：利用 OWL/描述逻辑 reasoner 检查不一致类、不可满足类和不应同时成立的公理。
3. **SHACL/规则约束**：以 data graph 与 shapes graph 为输入，返回 `conforms` 和逐条 violation。SHACL 能验证必填属性、数据类型、数量、取值范围、端点类型和自定义 SPARQL 约束；W3C 明确将它定义为约束符合性验证和结构化 validation report，不是事实真伪裁判。[W3C SHACL Recommendation](https://www.w3.org/TR/shacl/)

对体检报告图谱，可将这三类门分别落为：

- CQ：能否回答“某指标是否超参考区间”“哪些异常需要复查”。
- 推理：同一项检查不能同时被严格推理为 `High` 和 `Low`。
- SHACL：`LabResult` 必须包含指标、值、单位、检查日期和 evidence；数值字段必须是数值；`HAS_VALUE` 的 subject/object 类型必须合法。

#### 2.3.4 链接预测/KGE：MRR 与 Hits@K 只适用于排名补全任务

当目标是训练 embedding 或链接预测模型来补全隐藏三元组，标准做法是将测试三元组中的 head 或 tail 替换成候选实体并排序，使用 filtered setting 排除训练/验证/测试中已经为真的其他三元组，再报告：

- **MRR**：正确实体倒数排名的平均值；
- **Hits@K**：正确实体进入前 K 名的比例；
- **MR**：平均排名，通常需要与 MRR/Hits@K 一起看。

生物医学 KGE benchmark 也将 link prediction 作为标准评测任务，并采用 filtered candidates、MR、MRR 和 Hits@K。该文同时提醒，图谱不完备使未知候选不一定是假，因此排名高但未标注的候选不能自动当作错误事实。[生物医学 KGE benchmark 与最佳实践](https://pmc.ncbi.nlm.nih.gov/articles/PMC7971091/)

MRR/Hits@K 不适用于评估“从体检报告原文抽取实体和关系是否正确”。它们只能评估候选排序能力；需要另行人工/证据核验才能把预测边写入正式医疗图谱。

#### 2.3.5 应用效用：最终还要测图谱是否改善任务

如果图谱最终用于 GraphRAG、问答、临床规则或检索，应该在冻结问题集上测终端任务，并与无图谱基线比较：

- 问答：答案 exact match/F1、引用证据正确率、不可回答问题的拒答正确率；
- 检索：Recall@K、nDCG、关键证据命中率；
- 规则：规则触发 P/R、错误告警率、漏报率；
- 人工使用：专家完成任务的正确率、时长、认知负荷和高风险错误数。

这层能证明“图谱有业务效用”，但不应掩盖上游错误。最佳实践是保留从任务答案回到 graph edge、再回到原文 `exact_quote` 的链路，使每个成功和失败都可审计。

#### 2.3.6 建议采用的最小评测面板

对于本项目，建议每次版本发布统一报告下面八项，而不是寻求单一 KG score：

| 类别 | 最小发布指标 |
| --- | --- |
| 实体抽取 | strict span+type micro F1、每类 F1、关键实体漏报数 |
| 关系抽取 | exact directed typed F1、证据支持率、关键关系 FP/FN |
| 数值语义 | value/unit/reference-range 三元组准确率、异常方向错误数 |
| 事实质量 | 专家抽样 triple precision、无 evidence 边比例 |
| 完整性 | 必填字段/必填关系覆盖率，注明分母定义 |
| 一致性 | SHACL/规则违例率、逻辑冲突率、跨源冲突未解决数 |
| 本体适用性 | CQ 正确回答率、schema coverage |
| 运行与回归 | P50/P95、成本、错误率、冻结集指标变化和版本证据 |

各指标应同时附样本量、置信区间或至少分子/分母、数据集版本、模型/prompt/schema 版本和人工裁决规则。否则两个版本之间的差异无法解释，也不应据此宣布质量提升。

## 3. 现有测试是怎样测试的

### 3.1 初始化和调用链测试

`test_extractors.py` 会验证：

- `NERExtractor`、`RelationExtractor`、`TripletExtractor` 能初始化；
- 方法分发器被调用；
- 返回值是 list；
- 批量输入返回与输入数量对应的 list；
- 组件之间的调用关系没有断裂。

其中不少测试用 `MagicMock` 替换真实方法，只验证“调用了某个方法”，不验证模型抽出了什么。

这类测试适合发现 import error、参数传递错误和 batch 结构错误，不适合估计准确率。

源码：[抽取器基础测试](https://github.com/semantica-agi/semantica/blob/c0a051903f5eb58b9fab6da0983fe3ffe909034f/tests/semantic_extract/test_extractors.py)。

### 3.2 固定样例和回归测试

关系测试会构造少量已知实体和固定文本，检查例如：

- LLM 返回多个 founder 时是否全部保留；
- NER 没找到的实体是否变成 synthetic `UNKNOWN`；
- predicate 和 confidence 是否被保留；
- 空 subject/object 是否被跳过。

这些测试验证具体 bug 修复和结果解析逻辑，但不是完整 gold corpus。测试中的“期望关系”是代码作者在测试中手写的 fixture，不代表对真实语料进行统计评测。

源码：[关系解析回归测试](https://github.com/semantica-agi/semantica/blob/c0a051903f5eb58b9fab6da0983fe3ffe909034f/tests/semantic_extract/test_relation_extractor.py)。

### 3.3 Fallback 测试

`test_robustness_fallback.py` 明确要求：

- 一个英文首字母大写词可以通过 last-resort NER fallback 被标成 `UNKNOWN`；
- 没有模板关系且实体距离较远时，关系抽取器仍会按相邻实体生成 `last_resort_adjacency` 关系；
- triplet 可以从已有 relation 转换得到。

这些测试证明 fallback 按设计工作，但从质量评测角度，它们也说明“非空结果”不是正确率的证据：相邻实体强制连边可能是无语义证据的 false positive。

源码：[fallback 测试](https://github.com/semantica-agi/semantica/blob/c0a051903f5eb58b9fab6da0983fe3ffe909034f/tests/semantic_extract/test_robustness_fallback.py)。

### 3.4 性能测试和 benchmark

项目有性能测试以及可手动触发的 GitHub Actions benchmark workflow。它们关注：

- 吞吐量；
- 批处理和 worker 配置；
- 文本长度、切块和缓存行为；
- 可选依赖和真实库的运行情况。

性能 benchmark 不会告诉我们实体边界、实体类型、关系方向和关系类型是否正确。性能和质量必须分开报告。

源码：[性能测试](https://github.com/semantica-agi/semantica/tree/c0a051903f5eb58b9fab6da0983fe3ffe909034f/tests/semantic_extract)，[benchmark workflow](https://github.com/semantica-agi/semantica/blob/c0a051903f5eb58b9fab6da0983fe3ffe909034f/.github/workflows/benchmark.yml)。

## 4. 正确评测实体识别质量的方法

### 4.1 先定义 gold 标注格式

每条文本至少保存：

```json
{
  "doc_id": "report-001",
  "text": "血红蛋白 135 g/L，参考范围 120-160 g/L。",
  "entities": [
    {
      "text": "血红蛋白",
      "label": "LAB_INDICATOR",
      "start_char": 0,
      "end_char": 4
    },
    {
      "text": "135",
      "label": "VALUE",
      "start_char": 5,
      "end_char": 8
    },
    {
      "text": "g/L",
      "label": "UNIT",
      "start_char": 9,
      "end_char": 12
    }
  ],
  "relations": []
}
```

医疗文本还应单独标注：

- 否定范围；
- 数值与单位的绑定；
- 数值与参考区间的绑定；
- 异常方向，如 high/low/positive/negative；
- 同义词或缩写归一化；
- 是否允许嵌套实体；
- 是否允许一个实体有多个合法标签。

### 4.2 实体匹配规则

建议同时报告两种实体匹配：

**严格 span+type 匹配**：预测实体的 `start_char`、`end_char` 和 `label` 全部与 gold 相同才算 TP。适合评估边界和类型都必须准确的场景。

**宽松 overlap+type 匹配**：预测和 gold 类型相同，且字符区间有交集，才视为候选匹配；对于部分边界偏差，可以继续单独统计 boundary error。宽松匹配不能替代严格指标，只能作为诊断指标。

对实体集合计算：

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 * precision * recall / (precision + recall)
```

建议至少报告：

- strict span+type micro/macro F1；
- boundary-only F1；
- type-only F1；
- 每个实体类型的 precision/recall/F1；
- 每个文档的实体覆盖率；
- 漏掉关键实体的数量。

## 5. 正确评测关系抽取质量的方法

### 5.1 关系 gold 格式

关系必须绑定到实体 ID，而不是只保存模糊的实体文本：

```json
{
  "subject_id": "e1",
  "predicate": "HAS_VALUE",
  "object_id": "e2",
  "evidence": {
    "quote": "血红蛋白 135 g/L",
    "start_char": 0,
    "end_char": 12
  }
}
```

关系评测至少需要明确以下维度：

1. **端点是否正确**：subject 和 object 是否指向正确实体。
2. **方向是否正确**：`指标 -> 数值` 与 `数值 -> 指标` 不能混淆。
3. **关系类型是否正确**：`HAS_VALUE`、`HAS_UNIT`、`ABOVE_RANGE` 等不能只按相关性算对。
4. **证据是否正确**：关系是否能在原文中找到支持它的 quote。
5. **否定/条件/时间是否正确**：如“无异常”“建议复查”“既往史”不能被当成当前事实。

### 5.2 关系匹配层级

建议分层报告，而不是只给一个关系 F1：

| 指标 | 匹配条件 | 用途 |
| --- | --- | --- |
| endpoint F1 | subject/object 实体 ID 正确，忽略 predicate | 检查端点链接 |
| typed relation F1 | subject、predicate、object 全部正确 | 核心关系准确率 |
| directed relation F1 | 额外要求方向正确 | 检查主客体反转 |
| evidence-supported F1 | 关系还必须有原文证据 | 检查幻觉关系 |
| schema-valid rate | 关系类型、端点类型和字段符合 schema | 检查图谱可入库性 |

医疗 KG 最重要的通常不是平均关系 F1，而是：

- 关键关系的 recall；
- 错误高风险关系的 false positive rate；
- 无证据关系数量；
- 方向错误数量；
- 否定和异常方向错误数量。

## 6. 建议的测试数据切分

### 6.1 单元测试集

用于稳定代码行为，覆盖：

- 空文本、空实体、无关系文本；
- 单实体、多实体、嵌套实体、重叠实体；
- 主动句、被动句、否定句、并列结构；
- 数字、单位、百分比、参考区间；
- 中文标点和 OCR 错误；
- LLM JSON 缺字段、重复关系、实体未匹配；
- 长文本切块和跨块实体；
- fallback 是否按策略产生候选，而不是直接当成 VALID。

单元测试不应使用 API key，也不应把外部 LLM 的随机输出作为固定断言。LLM 解析层应使用固定 mock response；模型真实质量另行评测。

### 6.2 小型开发集

建议人工标注 50-200 篇文本，用于：

- 选择方法和阈值；
- 发现实体边界和关系方向问题；
- 调整 prompt、规则和 schema；
- 调试中文体检报告的特殊案例。

开发集不能用于最终发布结论。

### 6.3 冻结测试集

建议至少包含：

- 常规报告；
- OCR 质量差的报告；
- 多指标、多单位和参考区间；
- 否定、既往史、建议复查和异常描述；
- 容易产生相邻实体误连的负例；
- 关系方向相反的最小对比样本；
- 新医院、新版模板或新字段。

测试集在模型、prompt、规则和阈值确定后冻结，发布或切换版本时只允许读取，不允许针对结果改标注。

## 7. 推荐的执行流程

```text
冻结代码/模型/prompt/schema 版本
  -> 读取标注集
  -> 运行 NER
  -> 运行关系抽取
  -> 保存原始预测、日志和 provenance
  -> 规范化实体文本和 relation key
  -> 严格/宽松匹配 gold
  -> 计算 micro/macro Precision、Recall、F1
  -> 输出按类型、按文档、按难例的错误清单
  -> 人工复核高风险 FP/FN
  -> 决定 VALID、PARTIAL 或 REJECTED
```

每次评测都应保存：

- Semantica commit；
- Python 和依赖版本；
- NER/relation 方法；
- 模型名称和版本；
- prompt hash；
- schema 版本；
- 输入数据 hash；
- 输出预测 JSON；
- 汇总指标和错误样例。

如果使用 LLM，还要记录 provider、model、temperature、重试次数和响应解析失败数。不能只保存最终 F1，否则无法解释回归。

## 8. 适合本项目的最小评测脚本设计

建议在本项目中单独实现一个评测 runner，而不是修改 Semantica 的 `ExtractionValidator` 来冒充 gold evaluator。输入可以采用：

```text
evaluation/
  semantica/
    dev.jsonl
    test.jsonl
    predictions/
    reports/
```

runner 的最小接口：

```python
predictions = run_extraction(
    records,
    ner_method="llm",
    relation_method="llm",
    model_version="...",
    schema_version="...",
)

report = evaluate(
    gold_records=records,
    predictions=predictions,
    entity_match="strict_span_type",
    relation_match="exact_directed_typed",
)
```

输出至少包含：

```json
{
  "entities": {
    "strict_micro_precision": 0.0,
    "strict_micro_recall": 0.0,
    "strict_micro_f1": 0.0,
    "by_label": {}
  },
  "relations": {
    "typed_directed_f1": 0.0,
    "evidence_supported_f1": 0.0,
    "by_predicate": {}
  },
  "risk": {
    "unsupported_relation_count": 0,
    "direction_error_count": 0,
    "negation_error_count": 0,
    "critical_false_positive_count": 0
  }
}
```

## 9. 对 Semantica 测试机制的评价

### 优点

- 抽取模块有较完整的单元和回归测试覆盖；
- fallback、长文本、批处理、provider 解析和 synthetic entity 等边界行为有明确测试；
- 测试与方法实现分层，便于替换抽取器；
- 项目已经意识到需要 `ExtractionEvaluator` 和回归跟踪，规划方向是合理的。

### 缺口

- 没有随仓库提供可复现的实体/关系 gold dataset；
- 没有当前可调用的 extraction precision/recall/F1 evaluator；
- 没有中文或医疗领域的标注集和结果报告；
- 测试通过不能证明实体边界、关系方向和语义类型正确；
- `ExtractionValidator` 的 score 依赖预测自身的 confidence，不是外部正确性指标；
- fallback 测试反而确认了系统会在无明确关系证据时生成候选边。

## 10. 对体检报告项目的建议

本项目不能直接把 Semantica 的 `ExtractionValidator` 分数当作医疗抽取质量。建议沿用本项目已有的 evidence-first 边界：

1. 关系必须有 `exact_quote` 和字符 span。
2. entity、relation 和规则分别建立 gold 数据集，不能只用最终图结构反推正确率。
3. 正例完成后补充高难度负例，重点测试否定、建议、既往史、参考区间和相邻实体误连。
4. 语义不确定结果保留为 candidate，标记 `PARTIAL` 或 `HUMAN_REVIEW_REQUIRED`，不要因 fallback 非空就写入正式图谱。
5. 发布前至少报告严格实体 F1、严格有向关系 F1、证据支持率和关键关系 false positive rate。
6. 评测结果必须绑定源码、模型、prompt、schema 和数据版本，形成可回放的实验记录。

## 11. 最终结论

Semantica 当前的“测试”主要回答：代码是否能运行、方法是否被调用、结果结构是否符合预期、fallback 是否生效以及性能是否可接受。

它当前没有回答：抽取了多少正确实体、漏掉了多少实体、关系方向是否正确、关系是否有原文证据、医疗语义是否成立。

因此，Semantica 可以作为抽取管线和测试边界的参考，但实体/关系质量评测必须由项目方建立独立的 gold 标注、匹配规则、指标计算和人工复核流程。

## 参考资料

- [Semantica GitHub 仓库](https://github.com/semantica-agi/semantica)
- [ExtractionValidator](https://github.com/semantica-agi/semantica/blob/c0a051903f5eb58b9fab6da0983fe3ffe909034f/semantica/semantic_extract/extraction_validator.py)
- [抽取测试](https://github.com/semantica-agi/semantica/tree/c0a051903f5eb58b9fab6da0983fe3ffe909034f/tests/semantic_extract)
- [Evals 文档](https://github.com/semantica-agi/semantica/blob/c0a051903f5eb58b9fab6da0983fe3ffe909034f/docs/reference/evals.md)
- [Evals 模块](https://github.com/semantica-agi/semantica/blob/c0a051903f5eb58b9fab6da0983fe3ffe909034f/semantica/evals/__init__.py)
- [Knowledge Graph Quality Control: A Survey](https://www.sciencedirect.com/science/article/pii/S2667325821001655)
- [Knowledge Graph Refinement: A Survey of Approaches and Evaluation Methods](https://journals.sagepub.com/doi/10.3233/SW-160218)
- [Knowledge Graph Completeness: A Systematic Literature Review](https://doi.org/10.1109/ACCESS.2021.3056622)
- [W3C SHACL Recommendation](https://www.w3.org/TR/shacl/)
- [Benchmark and Best Practices for Biomedical Knowledge Graph Embeddings](https://pmc.ncbi.nlm.nih.gov/articles/PMC7971091/)
