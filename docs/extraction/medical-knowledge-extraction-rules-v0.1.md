# 医学知识抽取规则规范 v0.1

- 日期：2026-08-07
- 状态：讨论稿配套规范，已确认规则可用于实现设计；不得据此宣称当前代码已经全部实现
- 架构决策来源：[知识图谱本体设计：案例驱动讨论稿](../../知识图谱架构待讨论事项-2026-08-06.md)
- 适用范围：面向检验报告单解读的医学书籍实体、关系、规则和证据抽取

## 1. 文档职责

架构讨论稿说明“为什么这样建模”，本文说明“抽取器必须怎样执行”。两份文档通过 `decision_ref` 和案例编号 B0、B1、C1-C8 对应：架构稿中的已勾选结论才能成为本文的活动规则；架构稿中的未决项只能在本文登记为 `HOLD`，不能被抽取器自行补齐。

本文是目标抽取合同，不替代以下现有实现文档：

| 文档 | 职责 |
|---|---|
| [DeepSeek Semantic Extraction v0.3](deepseek-semantic-extraction-v0.3.md) | 当前大模型适配器、窗口、候选校验和 checkpoint 行为 |
| [Composite Rules v0.2](../runtime/composite-rules-v0.2.md) | 当前复合规则数据结构和三值求值边界 |
| 本文 | 医学语义分类、关系选择、RuleDefinition 路由、自动批准和 `HOLD` 条件 |

若本文与架构讨论稿冲突，以架构讨论稿最新的已确认决策为准，并在同一次修改中同步本文。若目标合同与当前代码不一致，必须记录为待实现差距，不能把文档目标描述成已运行能力。

当前第一版代码、旧 `TestItem` schema 和既有候选工件不作为新合同的兼容或迁移约束。本文与架构讨论稿冻结后，应先实现新的 `LabPanel`/`LabIndicator` schema、提示词和验证器，再从 canonical 来源重新抽取全部实体、关系和规则，并重建图谱；旧工件只归档审计，不直接改名迁移，也不与新抽取结果合并。

## 2. 状态与产物边界

抽取、审核、发布和执行是四个不同维度，不得共用一个状态字段：

| 维度 | 推荐值 | 含义 |
|---|---|---|
| `extraction_status` | `CANDIDATE`、`VALIDATED`、`REJECTED` | 抽取结果是否通过结构和证据校验 |
| `review_status` | `PENDING`、`APPROVED`、`REJECTED` | 医学或治理审核状态 |
| `publication_status` | `HOLD`、`ACTIVE`、`RETIRED` | 是否进入当前可用知识或规则集 |
| `execution_status` | `SUCCESS`、`SKIPPED_*`、`*_REVIEW_REQUIRED` | 某次报告上的运行结果，不写入静态图谱 |

确定性展开规则可以在通过全部校验后自动形成审核记录；大模型自身不能把候选标记为 `APPROVED`。书本初始 `PREPROCESS RuleDefinition` 是否可在 `review_status=PENDING` 时发布执行，按 C6-C8 的具体规则处理，不等于所有大模型候选都可直接执行。

抽取流水线固定为：

1. 校验来源 manifest、EvidenceChunk 和哈希。
2. 保存可逐字回放的连续 Evidence。
3. 抽取完整 Mention，不在关系阶段创造新端点。
4. 将 Mention 归入五类业务实体或 `OUT_OF_SCOPE`。
5. 在简单二元关系、`PREPROCESS RuleDefinition`、`GRAPH_COMPOSITE RuleDefinition` 三种结构中选择一种。
6. 执行 domain/range、方向、逻辑门、重复边、自环、证据回放和案例规则校验。
7. 保留候选、自动批准符合条件的确定性结果，或将不明确结果置为 `HOLD`。
8. 由已批准的 `GRAPH_COMPOSITE RuleDefinition` 确定性投影 Claim、RC-03、RC-08 和 RC-09；不得在图中独立编辑规则逻辑。

## 3. 通用抽取合同

### 3.1 Evidence 与 Mention

规则 `XR-BASE-01`，对应 B0/B1：

- 来源文本按版本完整保存，不改写、不摘要、不用模型补全文本。
- `Evidence.exact_quote` 必须是来源中的连续原文，并覆盖完整声明，不能只保存“如”“引起”“提示”等触发词。
- 表格 Evidence 至少绑定表头、条件单元格、输出单元格和可复现位置；公式 Evidence 绑定完整公式及符号定义。
- 实体 Mention 必须绑定来源中连续出现的文本。规范名称可以确定性归一化，但必须保留原 Mention。
- 同一 Evidence 可以被多个实体、简单关系或 Claim 共同引用；结构展开不得拆改原文。
- 关系抽取只能引用已冻结实体目录中的 ID。需要补造端点、跨句补全或使用医学常识扩展时，结果置为 `HOLD`。

### 3.2 五类业务实体

规则 `XR-BASE-02`，对应 B0：

| 类型 | 抽取判定 | 不得混入 |
|---|---|---|
| EC-01【检验组合】`LabPanel` | 一组有明确业务含义的检验指标组合 | 单个可测指标 |
| EC-02【检查指标】`LabIndicator` | 报告中可观察或可计算的具体指标 | 本次报告数值 |
| EC-03【指标状态】 | 单一指标的高、低、正常、阳性、阴性或趋势状态 | 多指标组合状态、疾病 |
| EC-04【临床背景】 | 影响检验解释的生理、临床、既往操作或治疗背景，以及联合规则输出的独立医学结论 | 形态或时间联合条件本身、仅用于规则选路的性别和数值年龄 |
| EC-05【病】 | 疾病或明确疾病分类 | 宽泛生理背景、泛化过程状态 |

静态【临床背景】只表示该概念与检验解释有关，不表示当前受检者已具有该背景。性别和数值年龄作为报告运行时属性；但来源明确陈述的“婴幼儿生长发育”等医学背景概念可以抽为【临床背景】，不得由此生成具体年龄阈值。

#### XR-BASE-02A `HAS_METRIC` 结构归属规则

- 只在来源通过标题、表格层级、列表归属或正文明确表达“组合包含指标”时建立 `LabPanel -[:HAS_METRIC]-> LabIndicator`。
- 每个有效 `LabPanel` 必须至少连接 1 个 `LabIndicator`；每个 `LabIndicator` 可以没有所属组合，也可以属于多个来源明确给出的组合。
- 同一 `LabPanel` 与 `LabIndicator` 节点对只保留 1 条 `HAS_METRIC`；多处重复出现时合并 `evidence_refs`，不得生成平行重复边。
- `HAS_METRIC` 只表达文献结构归属，不作为医学因果、诊断或规则推理边，也不得依据医学常识补建。

### 3.3 业务范围过滤

规则 `XR-BASE-03`，对应 ER-01：

当 Mention 不属于五类业务实体，且不参与当前检验报告解读的状态判定、疾病/病因召回、规则输入输出或结果解释时：

```text
classification = OUT_OF_SCOPE
preserve = [mention, exact_quote, evidence]
forbid = [canonical_business_entity, executable_relation, graph_reasoning, model_prompt]
```

`OUT_OF_SCOPE` 不表示来源错误，也不同于等待判断的 `HOLD`。

### 3.4 关系与规则路由

规则 `XR-BASE-04`，对应 B1：

| 原文结构 | 抽取结果 | Claim |
|---|---|---|
| 可独立成立的单起点到单终点事实 | RC-01、RC-02、RC-04 至 RC-07，关系直接保存 `evidence_refs` | 不建立 |
| 参考范围、阈值、公式、单位换算、原始时间序列计算 | 图外 `PREPROCESS RuleDefinition` | 不建立 |
| 多个已计算业务状态必须共同或按受限逻辑成立 | `GRAPH_COMPOSITE RuleDefinition` | 按同一 `rule_id/version` 投影逻辑门 |
| 多个独立宾语或多个独立原因 | 分别抽取多条简单关系 | 不因端点多而建立 |

关系选择顺序为 RC-01/RC-02 结构映射、`PREPROCESS`、`GRAPH_COMPOSITE`、RC-04 因果、RC-05 正向提示、RC-06 一般关联、RC-07 严格上位分类。不得创建案例专用关系名。

### 3.5 固定关系

| 编号 | 关系 | 允许方向 | 抽取要求 |
|---|---|---|---|
| RC-01 | `HAS_METRIC` | 检验组合 → 检查指标 | 每个组合含 `1..*` 个指标，每个指标可属于 `0..*` 个组合；只表示来源明确归属，不表示医学结论 |
| RC-02 | `HAS_STATE` | 检查指标 → 指标状态 | 每个指标状态恰好绑定一个指标 |
| RC-03 | `RULE_INPUT` | 检查指标/指标状态/临床背景 → Claim | 保存输入角色、条件组和顺序 |
| RC-04 | `CAUSES` | 临床背景/病 → 指标状态/临床背景/病 | 保存 `relation_role`、`assertion_status`、`evidence_refs` |
| RC-05 | `INDICATES` | 检查指标/指标状态/临床背景 → 临床背景/病 | 只承载正向候选提示 |
| RC-06 | `ASSOCIATED_WITH` | 已批准 domain/range 内的业务实体 | 一般关联，不传播因果 |
| RC-07 | `IS_A` | 同类业务实体 → 同类上位实体 | 仅严格上下位分类 |
| RC-08 | `RULE_OUTPUT` | Claim → 检查指标/指标状态/临床背景/病 | 只有逻辑门完整命中后可读取 |
| RC-09 | `SUPPORTED_BY` | Claim → Evidence | 只用于溯源 |

候选关系在正式映射前分别保存 `evidence_effect=SUPPORTS/ARGUES_AGAINST` 和 `clinical_use=suggest/exclude/differentiate/monitor`。`ARGUES_AGAINST` 不得映射成正向 RC-05，也不得推出目标绝对不存在。

### 3.6 并列成分共享谓词

规则 `XR-BASE-05`，对应 C5：

当多个并列项分别与同一对象构成可独立成立的二元事实时，确定性展开器可以生成多条简单关系并自动批准。必须同时满足：

1. 所有端点已经存在于冻结实体目录。
2. 并列项、共享谓词和共享对象位于同一完整 Evidence。
3. 所有展开结果使用同一个固定关系类型，且方向唯一。
4. 原文不含“共同、同时、联合”等合取语义。
5. 不涉及否定、不确定作用域、阈值、时间、人群条件或跨句补全。
6. 不创建新实体、新限定或模型补写文本。
7. 每条关系通过 domain/range、自环、重复边和 Evidence 哈希回放校验。

自动审核记录保存 `expander_rule_id`、版本和 Evidence ID。任一条件不满足时置为 `HOLD`，不得降级为大模型自由展开。

## 4. 案例对应规则

### 4.1 C1：指标状态与因果链

规则 `XR-C1-01`：

- 将“血清铁降低”拆成【检查指标】“血清铁”和【指标状态】“血清铁降低”，以 RC-02 连接。
- “导致、引起、所致”在完整句义明确因果时映射 RC-04；机制、来源和病因分别写入 `relation_role=mechanism/source/etiology`。
- 保留完整中间路径，不生成跨过“慢性失血”等中间概念的直连因果边。
- 不从“胃、十二指肠溃疡出血”等省略表达二次生成原文未逐字出现的实体。

### 4.2 C2：形态分类与疾病举例分离

规则 `XR-C2-01`：

- “贫血 AND MCV 正常 AND RDW 增大”抽为一条 `GRAPH_COMPOSITE RuleDefinition`，`subject_logic=ALL`，三个输入缺一不可。
- Claim 只输出【临床背景】“正细胞不均一性贫血”。
- “如早期缺铁性贫血、G6PD 缺乏症”另抽为从形态背景到疾病的 RC-06，不作为诊断规则输出。
- “早期缺铁性贫血”保留完整 Mention；规范实体使用【病】“缺铁性贫血”，限定 `OBJECT/DISEASE_PHASE/EARLY` 保持候选，等待全局合并。
- 不增加形态模式子类型，不生成任一输入直达输出的规则边。

### 4.3 C3：二维联合检测表

规则 `XR-C3-01`：

- 表格每一行按完整输入签名抽取一条独立 `GRAPH_COMPOSITE RuleDefinition`。
- 表头与单元格组合形成“血清铁降低”“TIBC 增高”等【指标状态】，不创建“血清铁低TIBC高”复合实体。
- 同一行的输入使用 `subject_logic=ALL`；只有同一受检者、同一有效报告上下文中的全部输入命中，Claim 才能开放输出。
- 同一行多个结果属于一个 `output_semantics=CANDIDATE_SET`，不得按输出数量拆成多条规则，也不得解释为同时确诊。
- 一个 Claim 可以同时输出【临床背景】和【病】。宽泛状态“慢性感染”归【临床背景】；“肝硬变、尿毒症、肾病综合征、恶性肿瘤”归【病】。
- 每条规则和投影 Claim 必须引用覆盖表头、该行条件及完整输出单元格的 Evidence。

规则 `XR-C3-01A`，对应第一行的“痔、消化性溃疡出血、月经过多引起的慢性失血”：

```text
entities:
  - Disease: 痔
  - ClinicalContext: 消化性溃疡出血
  - ClinicalContext: 月经过多
  - ClinicalContext: 慢性失血

simple_relations:
  - 痔 --RC-04 CAUSES--> 慢性失血
    relation_role: etiology
  - 消化性溃疡出血 --RC-04 CAUSES--> 慢性失血
    relation_role: source
  - 月经过多 --RC-04 CAUSES--> 慢性失血
    relation_role: source

common:
  assertion_status: asserted
  evidence_refs: [完整表格 Evidence]
```

三条 RC-04 复用 C1 的因果角色和 `XR-BASE-05` 的共享谓词展开。联合检测 Claim 只输出“慢性失血”候选；不得生成三个具体来源直达 Claim、血清铁降低或 TIBC 增高的边，也不得用展开后的结构替换完整原文。

规则 `XR-C3-02`，对应架构 C3 的“妊娠、婴幼儿生长发育需铁量增加”：

```text
entities:
  - ClinicalContext: 妊娠
  - ClinicalContext: 婴幼儿生长发育
  - ClinicalContext: 需铁量增加

simple_relations:
  - 妊娠 --RC-04 CAUSES--> 需铁量增加
    relation_role: mechanism
    assertion_status: asserted
  - 婴幼儿生长发育 --RC-04 CAUSES--> 需铁量增加
    relation_role: mechanism
    assertion_status: asserted

graph_composite_output:
  - 血清铁降低 AND TIBC增高 --Claim/RC-08--> 需铁量增加
    output_role: candidate
    output_semantics: CANDIDATE_SET
```

两条 RC-04 复用 `XR-BASE-05` 的共享谓词展开并共享完整表格 Evidence；满足全部确定性校验后自动批准，不建立额外 Claim。三项均不增加状态子类型。不得从“婴幼儿”自行生成数值年龄范围；只有存在另行批准的年龄映射时，运行时才能据此过滤不适用候选。检验组合只产生候选解释，不得把妊娠或婴幼儿状态写成当前受检者已确认背景。

规则 `XR-C3-03`：表格只抽取来源明确给出的行。对于来源没有列出的组合，抽取器不生成 RuleDefinition、Claim、节点、关系、覆盖占位或“来源未定义”记录；运行时未命中本表任何已抽取规则时，本表不产生报告输出。不得从缺省推导正常、排除、无病或任何其他医学结论。

### 4.4 C4：极性、排除与业务范围

规则 `XR-C4-01`：

- “D-二聚体阴性”和“D-二聚体不升高”是两个【指标状态】，分别以 RC-02 绑定 D-二聚体并共享原文 Evidence。
- “阳性见于”使用 RC-06，不当作充分诊断条件。
- 正向和反向证据候选分别保存 `evidence_effect` 与 `clinical_use`；正式关系类型尚未消歧的记录保持候选，不进入正向推理。
- “溶栓治疗监测”按 ER-01 标记 `OUT_OF_SCOPE`，只保留 Mention 和 Evidence。
- 心肌梗死、脑梗死、肺栓塞、恶性肿瘤归【病】；外科手术、炎症、感染、妊娠归【临床背景】。

### 4.5 C5：并列展开与上位概念

规则 `XR-C5-01`：

- 保留【临床背景】“造血原料缺乏”及“铁缺乏、叶酸缺乏、维生素 B12 缺乏”，以 RC-07 表达严格上位关系。
- 保留“造血原料缺乏 → 贫血”和三条具体缺乏到对应贫血疾病的 RC-04；上位路径不替代原文明示的具体关系。
- “如”只作列举定位，最终关系由完整句义决定。
- “叶酸、维生素 B12 缺乏引起巨幼细胞贫血”按 `XR-BASE-05` 自动展开，无需人工逐边审核。

### 4.6 C6：参考范围与严重度决策表

规则 `XR-C6-01`：

- 参考区间和贫血分类分别抽取为两条 `PREPROCESS RuleDefinition`。
- 性别和数值年龄写入 `applicability`，不建立业务节点；所有已声明条件按 `AND` 匹配。
- 贫血分类使用一条互斥决策表，一次返回 `anemia_status` 和 `severity`。
- 原文边界重叠、开闭不明或未覆盖时，不由模型补写；规则标记 `review_required=true`、`review_status=PENDING`，歧义输入返回 `RULE_BOUNDARY_REVIEW_REQUIRED`。
- 缺少必需上下文返回 `INSUFFICIENT_CONTEXT`，多行命中返回 `RULE_SELECTION_CONFLICT`。
- 严重度成立时同时激活“贫血”和具体等级，并以 RC-07 表达等级到“贫血”的上位关系。

### 4.7 C7：时间序列与联合趋势

规则 `XR-C7-01`：

- MPV 和血小板计数的原始序列分别交给固定 PREPROCESS Python 函数计算“持续下降”状态。
- 两个状态进入同一个 `GRAPH_COMPOSITE` ALL 逻辑门，并要求 `temporal_join=SAME_WINDOW`。
- Claim 只输出“骨髓造血功能衰竭”提示性候选；不生成任一单指标趋势直达输出的 RC-05。
- 时间窗、最小观测次数、缺测容忍度、方法兼容性和同步窗口均属于 RuleDefinition/Claim 条件；不抽取“同步持续下降”等模式【临床背景】节点，也不为【临床背景】增加 `role/subtype`。只有规则输出的独立医学结论才抽为【临床背景】。
- 单次数据返回 `INSUFFICIENT_LONGITUDINAL_DATA`；参数未审核返回 `RULE_PARAMETER_REVIEW_REQUIRED`，均不得激活趋势状态或 Claim。
- 大模型不得补写最小观测次数、时间窗、缺测容忍度或方法兼容参数。

### 4.8 C8：公式与派生指标

规则 `XR-C8-01`：

- 公式抽取为 `PREPROCESS RuleDefinition`，不建立 Claim，也不设计自定义 AST、MathML 或表达式 DSL。
- 已发布规则只调用已注册的版本化 Python 函数，并保存 `function_id/version`、`implementation_sha256`、输入输出契约、单位约束和 `evidence_refs`。
- PTR 规则要求报告同时提供受检 PT 和正常 PT；INR 规则要求 PTR 与报告中的 ISI。PTR 可以来自报告或本次前序派生。
- 缺少任一必需输入时返回 `SKIPPED_MISSING_INPUT`，不查询外部方法配置，不由模型猜测 ISI。
- 报告值与派生值分别保存为 `REPORTED_*` 和 `DERIVED_*`；派生值不得覆盖报告原值。
- 简单公式通过函数白名单、schema、数值边界和测试用例校验后可自动批准；复杂、歧义或校验失败的公式保持 `HOLD`。

## 5. 结构化记录与最小审计字段

目标合同采用“公共外壳 + 分类载荷”。公共外壳统一交换、审核和发布字段，`payload` 由 `record_type` 决定；不得为追求扁平统一而给所有对象复制无关医学字段。具体 JSON Schema 在架构冻结后单独版本化。

```text
record_id
record_type                 # entity | relation | rule_definition | claim_projection | evidence | out_of_scope
schema_version
decision_ref                # B0/B1/C1-C8
extraction_rule_id          # XR-...
evidence_refs
extraction_status
review_status
publication_status
validator_version
extractor_or_expander_version
payload
```

分类载荷的最小字段如下：

| `record_type` | `payload` 最小内容 |
|---|---|
| `entity` | `entity_id/type`、规范名称、`mention_text`、别名/定义和规范实体引用 |
| `relation` | `relation_id/type`、`source_ref`、`target_ref` 和该固定关系允许的限定属性 |
| `rule_definition` | `rule_id/version`、`rule_stage`、`rule_kind`、`inputs`、`applicability`、`evaluator`、`outputs` |
| `claim_projection` | `claim_id`、`rule_id/version`、`subject_logic`、`required_input_count`、`input_signature_hash`、`output_semantics`，以及执行所必需的 `applicability/expression_ast` |
| `evidence` | `evidence_id`、`exact_quote`、`source`、`source_location`、`source_sha256`、`quote_sha256` |
| `out_of_scope` | 原始 Mention、分类理由和 Evidence 引用 |

参考范围、阈值、公式、时间参数和方法参数根据 `rule_kind` 写入有类型的 `RuleDefinition.evaluator`，不建立独立图谱节点。Claim 只能由已批准 `GRAPH_COMPOSITE RuleDefinition` 确定性投影，不能成为第二份可独立编辑的规则；投影必须能以 `rule_id/version` 和签名重新生成并校验。建议不进入公共外壳，其归属和版本结构等待单独决策。

Evidence 自身的 `evidence_refs` 可以为空；其他记录按需引用一个或多个 Evidence。自动批准记录还必须包含确定性校验结果和 `expander_rule_id/version`；人工修订规则必须保留原版本、审核人、理由和补充依据。`execution_status`、运行时患者值、报告正文、患者属性和命中状态都不写入这些静态抽取记录。

## 6. 发布前验证

- 每个实体、关系和规则均可按哈希逐字回放 Evidence。
- 所有正式关系属于 RC-01 至 RC-09，且 domain/range 和方向合法。
- 每个【指标状态】恰好有一个对应指标的 RC-02。
- 每个 `GRAPH_COMPOSITE RuleDefinition` 的输入、逻辑、输出与 Claim、RC-03、RC-08、RC-09 投影完全一致。
- 联合规则不生成输入直达输出边；简单二元事实不被强制包装为 Claim。
- `CANDIDATE_SET` 不被渲染为确诊或当前人员已具有的背景。
- `ARGUES_AGAINST` 不进入正向提示传播。
- 未决限定、缺省组合、模糊边界和模型补全均保持 `HOLD` 或明确治理结果。
- `OUT_OF_SCOPE` 只保留 Mention 和 Evidence，不进入图谱检索、推理或模型提示。
- 发布工件由来源和规则版本可复现生成；不得手工修改 Neo4j 作为最终修复。

## 7. 当前对应状态

| 决策 | 本文规则 | 状态 |
|---|---|---|
| B0/B1 | `XR-BASE-01` 至 `XR-BASE-05` | 已确认，可进入 schema/验证器设计 |
| C1 | `XR-C1-01` | 已确认 |
| C2 | `XR-C2-01` | 主规则已确认；疾病阶段限定仍为候选 |
| C3 | `XR-C3-01`、`XR-C3-01A`、`XR-C3-02`、`XR-C3-03` | 已确认 |
| C4 | `XR-C4-01` | 已确认部分可实现；正反向关系最终映射 `HOLD` |
| C5 | `XR-C5-01` | 已确认 |
| C6 | `XR-C6-01` | 已确认；专家审核参数按规则版本治理 |
| C7 | `XR-C7-01` | 已确认；趋势参数待专家审核 |
| C8 | `XR-C8-01` | 已确认；实现任务见 GitHub Issue #7 |
