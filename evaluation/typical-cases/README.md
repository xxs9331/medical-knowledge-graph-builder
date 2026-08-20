# 典型案例小批量金标与 LLM Judge

## 当前状态

- `typical-cases-v0.1.json` 是从人工确认的《7. 典型案例分析》转换出的图测试集。
- 图测试集已经按冻结原文范围逐项补齐为审查草稿，状态为 `HUMAN_REVIEW_REQUIRED`；
  人工逐项确认后才能升级为正式闭集基准。
- `entity-test-set-v0.1.json` 是独立实体测试集，按 `entity_type + mention` 评分。
- `relationship-test-set-v0.1.json` 是独立关系测试集，同时包含正例与禁止抽取负例。
- `rule-test-set-v0.1.json` 是独立图规则测试集，按联合条件、结论和逻辑评分。
- Judge 只读取规范 EvidenceChunk、当前 Schema 和待判候选，不读取金标答案。
- Judge 输出是独立的 `candidate-only/HOLD` 工件，不修改 `graph.json`，不写 Neo4j，
  也不自动批准或发布候选。
- 每个案例通过 `evaluation_scopes` 冻结真实 EvidenceChunk 中的闭集范围。只评价证据完整
  落在该范围内的候选，避免同一 chunk 中不同案例互相产生假阳性。
- 主指标是金标监督式 `P/R/F1`：`P=TP/(TP+FP)`，`R=TP/(TP+FN)`；实体、普通关系、
  图规则分别计算，再对三类 TP/FP/FN 求整图 micro 指标。
- LLM Judge 的 `SUPPORTED/UNSUPPORTED/REPAIR/ABSTAIN` 只作为无监督诊断统计，不参与
  Precision、Recall 或 F1。
- 全称与缩写只有在规范原文以“全称（缩写）”明确绑定时才视为等价，不使用外部别名词典。
- 第一阶段只抽取能进入知识图谱的联合语义规则。公式、参考区间、阈值分级和单指标
  时间计算属于可执行逻辑，留给后续执行器模块单独抽取和评测。

## 已知覆盖边界

1. `TC-02`：六种贫血形态分别作为联合规则；不把当前 chunk 未单独给出的“贫血”重复
   建成运行条件，也不把任一单项状态直连到分类结果。
2. `TC-03`：二维表三行均已冻结为联合规则；单元格中的子原因保留层级普通关系，不重复
   追加为规则输出。
3. `TC-04`：鉴别、排除和监测语义不在当前关系合同内，暂列 `held_semantics`，不把它们
   强行转换为 `INDICATES`。
4. `TC-06`：人群参考区间和贫血程度分级已移出图规则金标，后续按执行器规则处理。
5. `TC-08`：PTR/INR 公式已移出图规则金标；OCR 与公式参数问题在执行器数据集中处理。
6. 原案例中的 `Claim` 已映射为当前 `RuleDefinition`；`SUPPORTED_BY/Evidence` 不进入
   当前候选图评分。

## 数据集维护

人工只编辑 `typical-cases-v0.1.json`。修改后运行以下命令，机械生成实体、关系和规则
三个独立视图：

```bash
.venv/bin/python evaluation/typical-cases/build_views.py
```

## 运行真实 Judge demo

模块入口固定读取仓库已有的 PT/PTR/INR 真实候选图及其规范 EvidenceChunk，调用真实
DeepSeek Judge，并将只读结果写入
`runtime/evaluations/judge-demo/ptr-inr-0022/judge-result.json`。运行前需要在项目根目录
`.env` 或当前环境中配置 `DEEPSEEK_API_KEY`：

```bash
.venv/bin/python -m medical_kg_sourceprep.extraction.graph_builder.judge
```

生产调用方直接使用 `judge_candidate_graph()`，传入已经完成硬校验的候选图、真实
EvidenceChunk、当前 Schema、输出路径和客户端。输出判定固定为：

- `SUPPORTED`：原文支持候选，但仍是 `HOLD`；
- `UNSUPPORTED`：原文不支持；
- `REPAIR`：只记录修改建议，不自动改图；
- `ABSTAIN`：模型无法可靠判断。

主评测编排位于
`medical_kg_sourceprep.extraction.graph_builder.runner.single_pass_evaluation`，入口
`scripts/run_typical_cases_experiment.py` 运行单轮链路：真实 EvidenceChunk 生成候选图后，
同一份 `graph.json` 分别进入无监督 LLM Judge 和有监督人工金标评分，结果写入
`runtime/evaluations/typical-cases/v0.1/evaluation-result.json`。模型不读取金标答案。

“Judge 与遗漏审查 -> 携带反馈二次抽取 -> 两轮并集”属于独立的分数提升实验，由
`run_reextraction_chunk()` 和 `scripts/run_judge_reextraction_experiment.py` 执行，不参与
上述主评测链路。

## 尚未实现

- `REPAIR` 生成新候选并重新执行确定性校验；
- 多模型或多 Judge 对照；
- 自动发布。最终发布仍必须由人工批准。
