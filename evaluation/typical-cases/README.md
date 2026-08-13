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

## 运行 Judge

先准备与案例对应的现有候选 `graph.json`，再运行：

```bash
.venv/bin/medical-kg-graph-builder-judge \
  --graph runtime/candidates/<run-id>/graph.json \
  --case-id TC-03 \
  --output runtime/evaluations/typical-cases/TC-03/judge-result.json
```

也可以不依赖安装后的脚本入口：

```bash
.venv/bin/python -m medical_kg_sourceprep.extraction.graph_builder.judge \
  --graph runtime/candidates/<run-id>/graph.json \
  --case-id TC-03 \
  --output runtime/evaluations/typical-cases/TC-03/judge-result.json
```

Judge 要求环境中已有 `DEEPSEEK_API_KEY`，通过现有 `trust_env=false` 客户端直连。输出固定为：

- `SUPPORTED`：原文支持候选，但仍是 `HOLD`；
- `UNSUPPORTED`：原文不支持；
- `REPAIR`：只记录修改建议，不自动改图；
- `ABSTAIN`：模型无法可靠判断。

## 尚未实现

- 金标与 Judge 输出的 Precision、Recall、F1 汇总；
- `REPAIR` 生成新候选并重新执行确定性校验；
- 多模型或多 Judge 对照；
- 自动发布。最终发布仍必须由人工批准。
