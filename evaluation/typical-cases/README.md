# 典型案例小批量金标与 LLM Judge

## 当前状态

- `typical-cases-v0.1.json` 是从人工确认的《7. 典型案例分析》转换出的图测试集。
- 图测试集已经人工确认，状态为 `HUMAN_VALIDATED`。
- `entity-test-set-v0.1.json` 是独立实体测试集，按 `entity_type + mention` 评分。
- `relationship-test-set-v0.1.json` 是独立关系测试集，同时包含正例与禁止抽取负例。
- `rule-test-set-v0.1.json` 是独立规则测试集，按阶段、输入、输出和逻辑评分。
- Judge 只读取规范 EvidenceChunk、当前 Schema 和待判候选，不读取金标答案。
- Judge 输出是独立的 `candidate-only/HOLD` 工件，不修改 `graph.json`，不写 Neo4j，
  也不自动批准或发布候选。
- 典型案例是针对关键目标与禁止边的小批量挑战集，不是每个 chunk 的穷举标注。评分使用
  目标覆盖率与禁止项规避率；未标注候选交给独立 Judge，不直接计为假阳性。
- 全称与缩写只有在规范原文以“全称（缩写）”明确绑定时才视为等价，不使用外部别名词典。
- 公式中的运行时参数可以由 RuleDefinition 的逐字公式证据证明覆盖，不要求为了评分把参数
  强制建成业务实体或 `RULE_INPUT` 图端点。

## 已知覆盖边界

1. `TC-02`：案例文档要求“贫血”作为联合规则输入，但规范 chunk 的单条分类文字没有
   单独重述这个前提；当前金标保留案例文档确认的标题上下文语义。
2. `TC-03`：第一版只冻结了二维表前两行的联合规则；高铁低 TIBC 第三行及全部端点
   类型仍需补齐。
3. `TC-04`：鉴别、排除和监测语义不在当前关系合同内，暂列 `held_semantics`，不把它们
   强行转换为 `INDICATES`。
4. `TC-06`：性别、数值和单位作为运行时参数，不加入静态业务实体；规则金标中的输入
   列表是合同描述，不等于 `RULE_INPUT` 图端点。
5. `TC-08`：规范 Markdown 将 INR 指数 OCR 为 `S1`，而正文定义 `ISI`；金标保留 ISI
   公式语义，同时保留该 OCR 记录。
6. 原案例中的 `Claim` 已映射为当前 `RuleDefinition`；`SUPPORTED_BY/Evidence` 不进入
   当前候选图评分。

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

主评测编排位于 `medical_kg_sourceprep.extraction.graph_builder.runner.evaluation`，入口
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
