# LLM 知识抽取：分阶段与联合抽取的待决问题及参考文献

日期：2026-08-12
状态：待决设计问题，不构成已采纳的架构结论。

## 待决问题

当前候选图构建器采用“实体候选 -> RuleDefinition 候选 -> 普通关系 ->
规则边”的分阶段流程。应当继续以冻结实体目录为关系和规则端点的唯一来源，还是让
同一次 LLM 调用联合产出实体、普通关系、规则和规则边，再由本地拆分、校验和审核？

问题的核心不在于模型是否应该读取完整原文。无论采用哪种输出编排，模型都应读取
完整原文和原始表格。争议是：语义发现是否一次联合生成，以及何时冻结可引用端点。

## 已确认的约束

- 所有产物都只是 candidate-only，`publication_status=HOLD`；不能由抽取或评测自动发布。
- 原始证据必须保留可逐字回放的锚点和 chunk hash。
- 本地校验负责 JSON 形状、来源回放、Schema、冻结端点和图结构；不使用有限关键词表
  裁决关系或规则的医学语义。
- 语义正确性应由独立评测模块或人工审核判断；评测模块不能补造端点或改写原文证据。

## 近三年 LLM 研究证据

### 1. LLM 直接一次性抽取复杂结构并不稳定

Wei et al. 的 ChatIE 将零样本 IE 改写为两阶段、多轮问答：先判断文本中可能出现的
元素类型，再根据已发现的类型链式提取具体信息。论文报告，直接给 ChatGPT 原始任务
指令不能稳定解决复杂结构化抽取，而拆分后的框架在六个中英文数据集上取得更好结果，
部分场景超过全监督基线。

对本项目的启示：可以让 LLM 保持全局阅读，但将复杂输出拆成可引用、可核验的子输出；
不能据此证明“实体必须由独立模型调用先抽完”。

### 2. 近期 LLM KG 构建常使用实体优先和增量融合

Lairgi et al. 提出 iText2KG：Document Distiller、增量实体抽取、增量关系抽取和图集成。
作者将实体/关系重复和不一致列为 LLM 建图的主要问题，并观察到同时执行实体和关系抽取
的基线会出现孤立节点；论文将该现象归因于生成式模型的幻觉与遗忘。该论文为预印本，
其因果解释应视作设计证据，而不是定论。

Graphusion 也采用“种子实体 -> LLM 候选三元组 -> 全局图融合”，把实体合并、冲突解决
和新三元组发现放在候选抽取之后。它强调只从局部句子/文档抽取三元组会缺少全局融合。

对本项目的启示：冻结目录、实体规范化和图融合有实际价值；但应保留高召回候选路径，
避免第一阶段漏抽永久阻断规则候选。

### 3. 医疗 LLM 建图中，两步提示和专家复核优于单次提示

Xu et al. 在心力衰竭知识图谱中使用 LLM、TwoStepChat 和医学专家修订。论文报告
TwoStepChat 优于单次 Vanilla prompt 以及其比较的微调 BERT 基线，并报告相较人工标注
节省约 65% 时间。这是医学领域的同行评议证据，但任务、数据集与模型设置不同，不能
直接转换为本项目的精确性能预期。

对本项目的启示：将“发现候选”和“语义评测/专家复核”明确分开是合理的；医学图谱
不能只依赖一次模型输出直接入正式图。

### 4. 近期正式论文支持“抽取后由 LLM Judge 独立过滤”

Huang et al. 的 GraphJudge（EMNLP 2025）流程为：实体抽取与实体中心去噪、基于实体
集合关系抽取形成草图、以原文和三元组为输入的 LLM Judge 逐条判断并过滤。论文动机
明确包含领域文档噪声、直接 LLM 建图的不准确性和幻觉，并在两个通用及一个领域数据集
报告优于所比较基线。

对本项目的启示：独立评测模块应成为下一步优先项。它可以对实体分类、关系类型、
RuleDefinition 语义和证据支持度做路由，但不能替代来源回放、冻结端点和候选生命周期校验。

### 5. 联合 Text-to-KG 仍是有效对照方案，不应被排除

Perrod et al. 将 Text-to-KG 定义为从文本直接生成事实三元组，并在零样本、少样本和微调
LLM 间做比较；论文将独立 NLP 流水线和端到端联合预测都列为有效实现路径。其结论不是
选择一种通用最优方案，而是不同数据、资源和适配策略有不同取舍。

对本项目的启示：应把“联合发现 + 分阶段接纳”作为可验证的备选方案，而不是把当前
分阶段调用当成无法更改的架构前提。

### 6. 实体阶段应区分语义输出与确定性溯源字段

当前实体提示词要求模型同时给出实体类型、名称、引语、出现序号、字符坐标和临时 ID。
其中只有“这是哪类实体、实体名称是否完整、为什么应抽取”需要语义判断；字符定位、出现
次数、hash、候选键和同名出现的证据聚合都可由代码对原文精确检索后确定。让模型同时完成
语义判断与复杂结构填写会扩大格式错误面。

Li et al. 的 G&O 将信息抽取拆为先生成内容、再组织为目标结构，在零样本 NER 和关系抽取
中报告优于直接结构化生成。Lu et al. 的 SchemaBench 研究也表明，即使是近期模型，复杂 JSON
Schema 的有效性和遵从性仍是独立失败来源。两项证据支持减少实体阶段的非语义输出负担，但
都不能直接证明“字段越少，医学实体 F1 必然越高”。

医疗 NER 研究还表明，少样本案例的选择和提示内容会显著影响结果；而生成式 LLM 在部分临床
NER 基准上仍落后于专用监督模型。因此，精简输出应作为待检验设计，而不是只靠直觉采纳。

对本项目的暂定方案是：实体阶段模型只输出 `type`、`mention` 和简短 `reason`；
`canonical_name_candidate` 首轮等于 `mention`，规范化另设阶段。代码以 `mention` 回查完整
chunk，生成所有匹配的 `source_ref`、表格行/单元格归属、位置、occurrence index、hash、
candidate key 与去重结果。最终发布层按实体类型和已批准规范名聚合多份来源证据。

该方案的边界：仅凭实体名称，代码无法知道模型究竟依据哪一次出现作出判断。因此，实体层
可以收集所有精确匹配作为候选证据；关系和 RuleDefinition 仍必须由模型给出可回放的关系/规则
依据，或由独立 Judge 结合原文评测。

### 7. 提示词应保持类型边界，失败样本应进入评测集而非持续堆叠

实体提示词应只保留四部分：任务和最小输出形状、每个 Schema 类型的正向定义、互相容易
混淆类型的对比边界，以及覆盖这些边界的少量示例。原文定位、坐标、去重、JSON 修复和
Schema 复验由代码完成；每次运行暴露的新错误先进入固定评测集，再决定是否需要调整类型定义
或替换一个示例，而不是无上限增加特例规则。

Mohan et al. 发现医疗 NER 的少样本效果显著依赖示例选择，基于相似性选择的示例优于随机
示例。Golde et al. 也说明，清楚的实体类型自然语言定义本身是少样本 NER 的有效信号。两项
研究支持“类型定义加具有区分力的示例”，但没有给出适用于本项目的固定最优提示词长度或示例数。
Naguib et al. 的临床 NER 对比还提醒，提示词优化不能替代独立评测。

对本项目的暂定策略：演示实体提示词采用一个 JSON 形状示例、一个疾病与机制的对比示例、
一个表格状态示例；后续只在评测集显示某类错误持续出现时，替换同类示例或修改相应类型定义。
提示词本身不承担候选接纳和发布裁决。

## 当前可检验的架构假设

建议后续以相同的书籍 evidence chunk、Schema 和人工标注样本比较三种实现：

| 方案 | 模型输出 | 本地接纳方式 | 主要风险 |
| --- | --- | --- | --- |
| A. 现有分阶段 | 实体、规则、普通关系、规则边分别调用 | 先冻结实体，再允许后续端点引用 | 实体漏抽导致后续规则/关系不可表达 |
| B. 联合发现，分阶段接纳 | 一次输出全部候选及证据锚点 | 本地先接纳实体，再解析并校验其余候选 | 单次 JSON 更长，实体与边可能不一致 |
| C. 高召回实体补抽闭环 | 常规分阶段；规则/关系阶段可报缺失端点线索 | 缺失端点回到实体补抽，不能直接入图 | 额外调用与回合收敛策略 |
| D. 精简实体语义输出 | `type`、`mention`、`reason`；代码生成定位和来源字段 | 代码回查所有出现位置并聚合证据 | 模型的实际依据位置不再显式可见 |

评价指标至少包括：实体/状态召回率、普通关系与规则的证据支持精度、冻结端点失败率、
表格规则召回率、人工复核耗时、每 chunk 的模型调用数和 token 成本。对方案 D 还应比较：
JSON/schema 失败率、实体精确率/召回率/F1、每个实体的平均输出 token、代码回查后证据候选数，
以及人工确认最终证据的耗时。所有对比都应保存相同粒度的原文锚点；不要以最终图规模替代
正确性指标。

## 抽取质量评测：无金标信号、LLM Judge 与人工校准

### 1. 无监督指标只能发现风险，不能替代实体 F1

没有人工金标时，系统不能知道未抽出的实体总数，也不能确定语义上正确的实体是否被错误
分类。因此，无监督指标不能报告为真实的 Precision、Recall 或 F1；它们适合持续监测、发现
回归和为人工/LLM 审核排序。

建议每次候选运行至少记录下列指标：

| 指标 | 计算方式 | 能发现的问题 | 不能说明什么 |
| --- | --- | --- | --- |
| 原文直接定位率 | `mention` 可在完整 chunk 精确匹配的候选占比 | 幻觉、截断和名称改写 | 语义类型是否正确 |
| 表格语义候选比例 | 不能直接匹配、但可能来自表头加单元格的候选占比 | 表格解析依赖和高风险来源 | 这些候选一定错误 |
| Schema/类型合规率 | 非空字段和允许类型的通过率 | 格式或任务理解偏差 | 医学语义正确性 |
| 去重率 | 相同 `type + mention` 的重复候选比例 | 重复输出和提示词不稳定 | 同名实体是否应合并 |
| 重复调用稳定性 | 同一 chunk 独立运行多次，比较 `type + mention` 集合的 Jaccard 一致性 | 生成随机性、边界定义不稳定 | 哪一次结果真正正确 |
| 跨模型一致性 | 抽取模型与独立模型审核的结论一致性 | 高风险分歧候选 | 两个模型共同犯错的风险 |

对于“血清铁降低”这类由表头和箭头共同表达的候选，不能因字符串无法直接匹配就直接
拒绝；应由代码标记为表格语义候选并进入 Judge/人工复核。它也不能被计入“原文直接定位率”。

### 2. LLM Judge 应评测候选，不应直接改写候选

可用独立于抽取调用的 LLM Judge 输入完整 chunk、代码生成的内部候选编号，以及每个候选的
`type`、`mention`、`reason`。Judge 逐条输出下列维度：原文/表格是否支持、实体边界是否完整、
类型是否正确、`reason` 是否被原文支持，并据此路由为 `ACCEPT`、`REVIEW` 或 `REJECT`。

Judge 还应在“原文单元 + 当前候选集合”的层面列出可能遗漏的实体。该结果可计算“Judge
视角的遗漏率”，但不能命名为真实召回率。Judge 不得改写实体名、补造医学知识、扩展 Schema
或直接发布候选；其结果仍须经过本地来源回放、Schema/端点复验和最终人工发布门。

VerifiNER 提出对既有 NER 输出进行事后验证，利用 LLM 结合上下文识别并修正不忠实预测，
且在生物医学数据集上验证了这种模型无关的验证思路。GraphJudge 则将图审查器置于 KG
草图之后，以处理领域文档噪声与直接建图产生的不准确性。这两项工作支持“抽取”和“审查”
分离，但不证明 Judge 的单次判断可以作为事实真值。

Judge 提示词应保持短而可审计：只允许依据输入原文判断，固定枚举输出，不要求长链式
推理；抽取模型和 Judge 应尽量使用不同模型或不同提示词版本，以减少同源错误。

### 3. 用小型人工金标校准 Judge，才可报告真实质量

建议先建立 100 至 200 个分层抽样的评测单元：普通正文/编号条目、表格行、标题和边界
案例均应覆盖。人工标注只需确认实体边界与 Schema 类型；其中至少抽取一部分由第二位审核者
复核，以了解标注本身的一致性。

该金标集上同时计算实体级 Precision、Recall、F1，以及 LLM Judge 与人工判断的一致率。
之后可让 Judge 优先筛查高风险候选，人工随机复核一部分低风险候选，既控制人工成本，也
避免只审核模型自认为不确定的样本而产生选择偏差。

## 暂定结论

现有证据不足以支持“完全联合抽取必然取代分阶段抽取”，也不足以支持“实体先抽取必然
最佳”。较有证据支撑的方向是：LLM 对完整原文做语义理解，输出采用固定结构；系统保留
候选、原文锚点、冻结端点和独立 judge。是否在模型调用层面改成联合发现，应通过上述
A/B/C 对照实验决定。

## 参考文献

1. Wei, X., Cui, X., Cheng, N., et al. (2023). *ChatIE: Zero-Shot Information
   Extraction via Chatting with ChatGPT*. arXiv:2302.10205.
   https://arxiv.org/abs/2302.10205

2. Carta, S., Giuliani, A., Piano, L., et al. (2023). *Iterative Zero-Shot LLM
   Prompting for Knowledge Graph Construction*. arXiv:2307.01128.
   https://arxiv.org/abs/2307.01128

3. Xu, T., Gu, Y., Xue, M., Gu, R., Li, B., & Gu, X. (2024). Knowledge graph
   construction for heart failure using large language models with prompt
   engineering. *Frontiers in Computational Neuroscience, 18*, 1389475.
   https://doi.org/10.3389/fncom.2024.1389475

4. Lairgi, Y., Moncla, L., Cazabet, R., Benabdeslem, K., & Cleau, P. (2024).
   *iText2KG: Incremental Knowledge Graphs Construction Using Large Language
   Models*. arXiv:2409.03284. https://arxiv.org/abs/2409.03284

5. Yang, R., Yang, B., Feng, A., et al. (2024). *Graphusion: A RAG Framework
   for Knowledge Graph Construction with a Global Perspective*. arXiv:2410.17600.
   https://arxiv.org/abs/2410.17600

6. Huang, H., Chen, C., Sheng, Z., Li, Y., & Zhang, W. (2025). Can LLMs be
   Good Graph Judge for Knowledge Graph Construction? In *Proceedings of
   EMNLP 2025* (pp. 10929-10948). https://aclanthology.org/2025.emnlp-main.554/

7. Perrod, A., et al. (2025). Fine-tuning or prompting on LLMs: evaluating
   knowledge graph construction task. *Frontiers in Big Data*.
   https://doi.org/10.3389/fdata.2025.1505877

8. Li, Y., Ramprasad, R., & Zhang, C. (2024). *A Simple but Effective Approach
   to Improve Structured Language Model Output for Information Extraction*.
   arXiv:2402.13364. https://arxiv.org/abs/2402.13364

9. Mohan C., M., Punnan, S. S., & Kleenankandy, J. (2024). Improving few-shot
   prompting using cluster-based sample retrieval for medical NER in clinical
   text. In *Proceedings of ICON 2024* (pp. 37-44).
   https://aclanthology.org/2024.icon-1.4/

10. Naguib, M., Tannier, X., & Neveol, A. (2024). Few-shot clinical entity
    recognition in English, French and Spanish: masked language models
    outperform generative model prompting. In *Findings of EMNLP 2024*
    (pp. 6829-6852). https://aclanthology.org/2024.findings-emnlp.400/

11. Mullick, A., Gupta, M., & Goyal, P. (2024). Intent detection and entity
    extraction from biomedical literature. In *Proceedings of CL4Health 2024*
    (pp. 271-278). https://aclanthology.org/2024.cl4health-1.33/

12. Lu, Y., Li, H., Cong, X., et al. (2025). Learning to generate structured
    output with schema reinforcement learning. In *Proceedings of ACL 2025*
    (pp. 4905-4918). https://aclanthology.org/2025.acl-long.243/

13. Kim, S., Seo, K., Chae, H., Yeo, J., & Lee, D. (2024). VerifiNER:
    Verification-augmented NER via knowledge-grounded reasoning with large
    language models. In *Proceedings of ACL 2024* (pp. 2441-2461).
    https://aclanthology.org/2024.acl-long.134/

14. Golde, J., Hamborg, F., & Akbik, A. (2024). Large-scale label
    interpretation learning for few-shot named entity recognition. In
    *Proceedings of EACL 2024* (pp. 2915-2930).
    https://aclanthology.org/2024.eacl-long.178/

## 证据边界

- 文献 1、2、4、5、8 为 arXiv 预印本，应与同行评议结果区分。
- 各论文的数据集、领域、本体约束、模型版本和评价指标不同；不能将报告的分数直接
  外推到医学教材规则抽取。
- “LLM Judge”本身可能误判。它只能作为候选路由或审核辅助，不能覆盖原文回放和人工发布门。
