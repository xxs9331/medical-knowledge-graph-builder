# Semantica 实体识别与关系识别机制调研

## 1. 调研范围与结论

- 项目：`semantica-agi/semantica`
- 检查版本：`c0a051903f5eb58b9fab6da0983fe3ffe909034f`（2026-08-14，`main`）
- 重点代码：`semantica/semantic_extract/`、`semantica/kg/graph_builder.py`、相关测试和 cookbook。

核心结论：Semantica 不是单一的实体识别或关系抽取算法，而是一个可插拔的多方法抽取框架。默认配置下，实体识别调用 spaCy 模型 `en_core_web_sm`，关系识别调用英文正则关系模板；LLM、HuggingFace、spaCy 依存句法、共现和相似度方法均需显式配置。抽取结果随后交给图构建、去重、冲突处理和溯源模块。

因此，README 中的 “NER/relation/event extraction” 是能力集合，不等于默认使用 LLM，也不代表项目内置了面向中文体检报告的领域模型。

## 2. 抽取链路

总体链路可以概括为：

```text
文本
  -> NERExtractor：实体识别，输出 Entity(text, label, span, confidence)
  -> RelationExtractor：以实体列表为候选端点，输出 Relation(subject, predicate, object, confidence)
  -> GraphBuilder：构图、实体解析/去重、冲突检测、溯源与导出
```

`NERExtractor` 和 `RelationExtractor` 都支持字符串或批量输入、置信度阈值、方法列表和方法注册。方法列表按顺序尝试，默认返回第一个产生合格结果的方法；NER 还可以开启 ensemble voting。代码位置：

- [NER 方法配置与调度](https://github.com/semantica-agi/semantica/blob/c0a051903f5eb58b9fab6da0983fe3ffe909034f/semantica/semantic_extract/ner_extractor.py#L84-L155)
- [NER fallback 与阈值过滤](https://github.com/semantica-agi/semantica/blob/c0a051903f5eb58b9fab6da0983fe3ffe909034f/semantica/semantic_extract/ner_extractor.py#L306-L457)
- [方法分发器](https://github.com/semantica-agi/semantica/blob/c0a051903f5eb58b9fab6da0983fe3ffe909034f/semantica/semantic_extract/methods.py#L2619-L2672)

## 3. 实体识别是怎么做的

### 3.1 默认方法：spaCy ML NER

`NERExtractor` 默认参数是 `method="ml"`，模型默认名是 `en_core_web_sm`。初始化时如果配置了 ML 方法，会尝试加载 spaCy 模型。真正的 ML 方法在 `methods.py` 中调用 `spacy.load(model)`，然后遍历 `doc.ents`，直接把 spaCy 的实体文本、类型和字符起止位置转换为 Semantica 的 `Entity`。

实体对象主要包含：

```text
text         实体文本
label        实体类别，例如 PERSON、ORG、GPE、DATE
start_char   字符起点
end_char     字符终点
confidence   置信度
metadata     模型、lemma、抽取方法等
```

当前实现需要注意：spaCy `doc.ents` 通常没有统一的实体置信度字段，因此代码默认将置信度设为 `1.0`，只有对象存在 `confidence` 或 `score` 属性时才覆盖。这意味着默认 spaCy 路径中的 `confidence` 不能直接理解为经过校准的概率。

源码：[NERExtractor 的 spaCy 转换逻辑](https://github.com/semantica-agi/semantica/blob/c0a051903f5eb58b9fab6da0983fe3ffe909034f/semantica/semantic_extract/ner_extractor.py#L533-L567)，[独立 ML 方法](https://github.com/semantica-agi/semantica/blob/c0a051903f5eb58b9fab6da0983fe3ffe909034f/semantica/semantic_extract/methods.py#L661-L725)。

### 3.2 可选实体识别方法

分发器内置以下方法：

| 方法 | 实现方式 | 适用/限制 |
| --- | --- | --- |
| `ml` / `spacy` | spaCy NER 模型 | 默认方法；默认模型是英文模型 |
| `pattern` | 固定正则 | 只覆盖少量英文 PERSON、ORG、GPE、DATE |
| `regex` | 可传入自定义正则 | 可扩展，但仍是规则匹配 |
| `rules` | 简单语言规则 | 当前示例规则主要把句首大写词当作 PERSON |
| `huggingface` | Transformers token-classification pipeline | 可换模型；依赖模型和标签体系 |
| `llm` | LLM 结构化输出 | 依赖 provider/API；结果需要解析和实体匹配 |

固定正则的默认实体类别很窄。例如 `pattern` 识别的是英文公司后缀、连续首字母大写词、`City/State/Country/Nation` 地名和数字日期；`regex` 另外加入金额和百分比。代码见 [实体 pattern/regex/rules/ML 方法](https://github.com/semantica-agi/semantica/blob/c0a051903f5eb58b9fab6da0983fe3ffe909034f/semantica/semantic_extract/methods.py#L571-L725)。

### 3.3 降级、阈值和集成

实体抽取按配置的方法顺序执行。一个方法返回至少一个达到 `min_confidence` 的实体时，默认立即返回；所有方法失败或没有合格结果时，进入内部 fallback：

1. 用固定英文正则抽取 PERSON、ORG、GPE、DATE。
2. 如果仍为空，则把长度大于 2 的英文首字母大写词标成 `UNKNOWN`，置信度 `0.5`。
3. 若提供 `entity_types`，会对类型相似度重新加权后再过滤。
4. `ensemble_voting=True` 时，按 `(实体文本小写, 类型)` 聚合多个方法的结果，以平均置信度做投票。

这套 fallback 让系统不容易返回空列表，但也可能产生“看起来有实体、实际语义错误”的结果，尤其不适合直接作为医疗知识图谱的金标准。

## 4. 关系识别是怎么做的

### 4.1 默认方法：实体约束下的英文正则关系模板

`RelationExtractor` 默认参数是 `method="pattern"`，默认置信度阈值为 `0.6`。它不是先从全文自由生成关系，而是要求调用方先提供实体列表：源码明确规定没有实体时不能进行关系抽取。

pattern 方法先把实体文本转义并拼成候选实体正则，然后只在文本中匹配“已知实体 + 预置关系句式 + 已知实体”。内置关系类型包括：

- `founded_by`
- `located_in`
- `works_for`
- `born_in`
- `acquired_by`

同时覆盖部分主动/被动句式，如 `was founded by`、`founded`、`headquartered in`、`works for`、`acquired by` 等。匹配成功后，通过实体文本映射回原来的 `Entity` 对象，并截取匹配前后文作为 `context`。pattern 关系的固定置信度为 `0.7`。

源码：[RelationExtractor 默认配置](https://github.com/semantica-agi/semantica/blob/c0a051903f5eb58b9fab6da0983fe3ffe909034f/semantica/semantic_extract/relation_extractor.py#L81-L140)，[pattern 关系模板与结果构造](https://github.com/semantica-agi/semantica/blob/c0a051903f5eb58b9fab6da0983fe3ffe909034f/semantica/semantic_extract/methods.py#L1190-L1300)。

### 4.2 可选关系识别方法

| 方法 | 实现方式 | 结果特点 |
| --- | --- | --- |
| `pattern` | 固定英文关系模板 | 默认；关系类型有限、依赖字面句式 |
| `regex` | 用户可传关系正则 | 可定制；默认只有 `founded_by`、`located_in` 示例 |
| `cooccurrence` | 实体在 100 字符内即建立 `related_to` | 召回优先，不等于语义关系 |
| `similarity` | 实体间文本与关系类型做 spaCy 向量/字符串相似度 | 需要关系类型；无向量时退化为字符串相似度 |
| `dependency` / `ml` / `spacy` | spaCy 依存句法和词性分析 | 适合英文句法；仍依赖 spaCy 模型 |
| `huggingface` | Transformers 关系模型 | 依赖具体模型，输出标签需与项目格式适配 |
| `llm` | LLM 输出 JSON/Pydantic 结构 | 可抽取开放关系和时间字段，但依赖外部模型 |

关系方法分发见 [关系方法注册](https://github.com/semantica-agi/semantica/blob/c0a051903f5eb58b9fab6da0983fe3ffe909034f/semantica/semantic_extract/methods.py#L2646-L2672)。

### 4.3 LLM 关系抽取路径

显式选择 `method="llm"` 后，代码会：

1. 检查 provider 和 API key，并支持 OpenAI-compatible endpoint。
2. 把文本和实体列表写入提示词，实体列表默认最多向提示词提供 80 个实体。
3. 要求模型输出 `{"relations": [{"subject", "predicate", "object", "confidence"}]}`。
4. 使用 Pydantic/instructor 的 typed generation；typed 结果为空时再尝试结构化 JSON。
5. 用混合相似度把模型返回的 subject/object 对齐到已有实体；对不匹配的实体创建 `UNKNOWN` synthetic entity。
6. 长文本按递归切块并保留 10% overlap；可选抽取 `valid_from`、`valid_until` 和 `temporal_source_text`。

因此，LLM 路径本质上是“候选实体约束 + LLM 开放式关系生成 + 结果解析/实体对齐”，不是完全由规则决定。源码：[LLM 关系提示词与 typed JSON 处理](https://github.com/semantica-agi/semantica/blob/c0a051903f5eb58b9fab6da0983fe3ffe909034f/semantica/semantic_extract/methods.py#L1711-L1836)、[结果解析与实体对齐](https://github.com/semantica-agi/semantica/blob/c0a051903f5eb58b9fab6da0983fe3ffe909034f/semantica/semantic_extract/methods.py#L2010-L2180)。

### 4.4 关系 fallback 的风险

当配置的方法都没有返回合格关系时，`RelationExtractor` 会退回 pattern；如果 pattern 也没有结果且至少有两个实体，则按实体在文本中的相邻顺序强制生成关系，谓词为 `related_to`，置信度为 `0.3`。这条最后降级路径对于“避免空结果”有帮助，但它没有语义证据，不能视为关系识别成功。

此外，co-occurrence 方法会把距离小于 100 个字符的实体连成 `related_to`，默认置信度为 `0.6`，刚好可以通过默认阈值。因此在启用该方法时，实体同段出现可能直接变成图边。

源码：[关系调度、过滤和 fallback](https://github.com/semantica-agi/semantica/blob/c0a051903f5eb58b9fab6da0983fe3ffe909034f/semantica/semantic_extract/relation_extractor.py#L318-L500)、[相邻实体 fallback](https://github.com/semantica-agi/semantica/blob/c0a051903f5eb58b9fab6da0983fe3ffe909034f/semantica/semantic_extract/relation_extractor.py#L505-L537)、[co-occurrence 实现](https://github.com/semantica-agi/semantica/blob/c0a051903f5eb58b9fab6da0983fe3ffe909034f/semantica/semantic_extract/methods.py#L1347-L1376)。

## 5. 面向中文体检报告的适用性判断

### 可以借鉴的设计

1. 把实体识别、关系识别、图构建、去重、冲突检测和溯源拆成独立模块，便于替换领域模型。
2. 统一的 `Entity` / `Relation` 结果结构包含字符位置、上下文、置信度和 metadata，便于保留证据。
3. 支持方法链和结构化 LLM 输出，适合将规则、模型和人工审核组合起来。
4. LLM 关系抽取提示词要求关系只能来自原文，并支持时间字段，这个接口方向可参考。

### 不应直接照搬的部分

1. 默认模型和大部分正则/关系模板是英文导向的，不能直接覆盖中文体检报告中的指标、单位、参考区间、异常描述、疾病和药物。
2. 默认实体置信度存在“spaCy 结果直接设为 1.0”的实现问题，不能把该字段直接当成校准概率。
3. `cooccurrence` 和“相邻实体强制 related_to”会制造没有语义证据的边；医疗场景应拒绝或标记为候选，而不是直接写入正式图谱。
4. LLM 返回的实体无法匹配时会创建 synthetic `UNKNOWN` 实体，这对于需要严格 provenance 和实体规范化的医疗 KG 必须改为候选/待审核状态。
5. 项目代码没有显示出面向体检报告的实体词典、数值单位解析、否定检测、比较方向、参考区间语义和医学关系约束；这些需要在上层领域管线补齐。

## 6. 对本项目的建议

如果将 Semantica 的思路用于体检报告分析，建议只借鉴其“可插拔抽取器 + 统一结果模型 + provenance metadata”的框架，不直接采用默认抽取器。推荐改造为：

```text
OCR/文本标准化
  -> 中文医学实体识别（指标、数值、单位、参考区间、疾病、药物、部位）
  -> 确定性数值/单位/区间解析
  -> LLM 或规则生成关系候选
  -> exact_quote、字符 span、source_sha256、规则/schema 校验
  -> VALID / PARTIAL / REJECTED
  -> publication_status=HOLD，经过人工审核后再写正式图谱
```

特别是关系抽取，应要求每条候选关系同时保存：subject/object 的原文 span、关系触发证据、`exact_quote`、原文来源和模型/规则版本。语义方向和医学关系类型不能仅由距离、共现或固定置信度决定。

## 7. 最终评价

Semantica 的价值主要在于提供了一个完整的语义抽取与知识图谱基础设施接口：默认路径轻量、可离线、可解释；高级路径可接 spaCy、HuggingFace 或 LLM，并把结果统一到图构建流程中。

但从实体识别和关系识别算法本身看，它更像“多策略工程框架”，而不是一个已经针对中文医学文本训练和验证的抽取系统。默认关系识别的实际能力主要来自英文正则模板，LLM 路径的质量主要取决于外部模型和提示词，fallback 还可能引入无证据关系。因此，它适合作为架构和接口参考，不适合作为体检报告知识图谱的直接抽取引擎。

## 参考资料

- [Semantica GitHub 仓库](https://github.com/semantica-agi/semantica)
- [Semantic extraction package](https://github.com/semantica-agi/semantica/tree/c0a051903f5eb58b9fab6da0983fe3ffe909034f/semantica/semantic_extract)
- [实体抽取实现](https://github.com/semantica-agi/semantica/blob/c0a051903f5eb58b9fab6da0983fe3ffe909034f/semantica/semantic_extract/methods.py)
- [关系抽取实现](https://github.com/semantica-agi/semantica/blob/c0a051903f5eb58b9fab6da0983fe3ffe909034f/semantica/semantic_extract/relation_extractor.py)
- [README：知识管线与 semantic_extract 模块说明](https://github.com/semantica-agi/semantica#architecture)
