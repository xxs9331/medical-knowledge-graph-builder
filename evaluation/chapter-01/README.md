# 第一章人工抽取测试集

## 文件

- `chapter-01-graph-test-set-v0.3.json`：当前人工图标注源，按 8 个章节主题组织。
- `chapter-01-entity-test-set-v0.3.json`：实体投影测试集。
- `chapter-01-relationship-test-set-v0.3.json`：普通关系投影测试集。
- `chapter-01-rule-test-set-v0.3.json`：规则投影测试集。
- `chapter-01-evidence-audit-v0.3.json`：全部正例的可回放证据审查队列。
- `chapter-01-layered-test-set-v0.4.json`：8 个案例的自动生成金标，分开保存原文
  mention、规范实体、mention 到规范实体的映射，以及基于规范实体的关系。
- `chapter-01-scoped-gold-v0.5.json`：当前自动评测入口。由于 v0.3/v0.4 不是全章
  穷尽标注，本版明确限定 mention、canonical 和关系的评测域，范围外预测不计 FP。
- `chapter-01-entity-mentions-v0.6.json`：8 个案例、44 个 chunk 的闭世界实体 mention
  标注稿，只保存原文字符跨度和五类实体，不包含 canonical、关系、规则或非逐字派生实体。
- `chapter-01-canonical-entities-v0.8.json`：以 v0.3 统一人工图为语义主干，并接入
  v0.6 的 904 条原文 mention 作为 span 评测层。它保留 mention 到 canonical 的完整映射，
  展开全部并列结构，连接原文中已有的嵌套跨度，并保留人工图中的表格派生、规则端点和
  非逐字实体；最后按“实体类型 + 规范名称”去重。外层和内层实体分别保留，不以 alias
  方式互相合并。
- `build_v04_layered.py`：根据 v0.3 图金标和人工审核用证据映射生成 v0.4。
- `build_v05_scoped.py`：从 v0.4 生成限定评测域，并把原文中的同义表面形式绑定到
  canonical ID；表面形式只来自既有证据映射，不读取待评测模型输出。
- `build_v06_exhaustive_mentions.py`：合并旧参考跨度和两次独立抽取候选，排除非逐字
  派生项，并按 Schema 裁决类型冲突。v0.6 使用模型候选辅助发现，因此不能反评生成
  候选的同一次运行，须冻结后用新运行评测。
- `build_v08_canonical_entities.py`：从冻结的 v0.6 mention 生成 span 对齐实体，并机械合并
  v0.3 人工图实体；不调用大模型、不使用外部医学知识，也不改动原文 mention。逐字mention
  缺失不再删除已人工确认的图谱实体。
- `build_v10_relationship_gold.py`：先以 canonical ID 机械继承 v0.3 的全部普通关系，再
  叠加全章关系审计；若任一人工关系无法继承，构建直接失败。
- `build_v03.py`：在 v0.2 基础上补齐贫血分类、血型遗传和诊断排除规则。
- `build_v02.py`：从 v0.1 草稿生成 v0.2 的范围合同、已确认修复和审查队列。
- `build_views.py`：从当前 v0.3 图标注源机械生成三个投影，防止多份评分数据不一致。

v0.1、v0.2 为历史版本，不再作为当前 L2 的评分输入。v0.3 保持
`HUMAN_REVIEW_REQUIRED`，自动证据审查不等于人工语义验收。

## 状态与边界

- 当前状态是 `HUMAN_REVIEW_REQUIRED`：内容由人工阅读第一章后抽取，但尚未经过用户逐条验收。
- 44 个规范 EvidenceChunk 均被纳入某个测试单元；标题块可以只提供上下文，不强制产生图元素。
- 每个 v0.3 案例都有覆盖完整运行输入的 `evaluation_scopes`，scope 与 `chunk_ids`
  严格一致，避免证据只进入评分过滤器却没有送入抽取模型。
- `chapter-01-evidence-audit-v0.3.json` 对每条正例给出 `AUTO_SURFACE_MATCH` 或
  `NEEDS_HUMAN_REVIEW`。前者只表示字面可定位，不表示关系方向、因果强度或规则逻辑已批准。
- 当前审查已移除“深静脉血栓形成 -> D-二聚体阳性”正例，改为禁止抽取关系；并统一
  “红细胞压积增高”为 `IndicatorState`。
- 规则按运行合同分层：36 条 `GRAPH_COMPOSITE` 是当前图抽取目标，包括原有 10 条、
  4 条 MCV/MCH/MCHC 分类、10 条血型可能性推理、9 条血型排除和 3 条其他排除规则；30 条
  `PREPROCESS` 保存在 `executor_rules` 等待执行器评测，3 条原文未给出具体输入配对的
  MCV/MCH/MCHC 公式保存在 `held_rules`，不参与评分。
- 血型采用“指标 -> 状态”建模；父母血型组合及子女可能/不可能血型直接使用文本状态，
  不增加模态 JSON 字段。原文明示的排除规则不同于 `must_not_extract`。
- 当前共有 241 个实体记录、193 个正关系、1 个禁止关系；所有正式评分继续为 `HOLD`，
  直到逐项人工验收完成。
- v0.4 状态为 `GENERATED_GOLD`，可以直接用于自动评测；其来源是 v0.3 canonical
  标注和自动证据映射，`human_approved=false`，不得表述为人工批准金标。
- v0.4 中 193 个严格 mention 可以按 chunk 坐标逐字回放；其余规范实体通过并列拆分、
  表格结构或上下文派生，不强制伪造为原文逐字实体。四层分别计算 TP、FP、FN 和
  P/R/F1，不能用 mention 分数代替规范实体或关系分数。
- 当前 8 例仍是目标型挑战集，不是对 44 个 chunk 的穷尽闭世界标注。直接把范围内所有
  未收录候选计为 FP 得到的是诊断性严格分数；人工补齐全部合法实体前，不应把它解释为
  模型的最终精度。
- 因此 v0.4 的全 chunk 闭世界 Precision 已停用。v0.5 只在声明域内按
  `TP/(TP+FP)`、`TP/(TP+FN)` 计算：mention 使用已标注的精确跨度，canonical 使用
  已标注证据上下文，关系只评估已知 canonical 实体之间的候选边。
- “见于”“可见于”默认标为 `ASSOCIATED_WITH`；仅在原文明示机制或因果时使用 `CAUSES`。
- ABO 表格存在 OCR 歧义，当前只保留正文明确说明和可无歧义复核的规则，疑似错误单元不作为强制金标。

## 重新生成投影

```bash
.venv/bin/python evaluation/chapter-01/build_v03.py
.venv/bin/python evaluation/chapter-01/build_views.py
.venv/bin/python evaluation/chapter-01/build_v04_layered.py
.venv/bin/python evaluation/chapter-01/build_v05_scoped.py
.venv/bin/python evaluation/chapter-01/build_v06_exhaustive_mentions.py
.venv/bin/python evaluation/chapter-01/build_v08_canonical_entities.py
```

生成后应运行：

```bash
.venv/bin/python -m json.tool evaluation/chapter-01/chapter-01-graph-test-set-v0.3.json >/dev/null
.venv/bin/python -m unittest tests.test_chapter01_evaluation_sets -v
.venv/bin/python -m unittest tests.test_graph_builder_layered_scoring -v
.venv/bin/python -m unittest tests.test_chapter01_entity_mentions_v06 -v
.venv/bin/python -m unittest tests.test_chapter01_canonical_entities_v08 -v
```
