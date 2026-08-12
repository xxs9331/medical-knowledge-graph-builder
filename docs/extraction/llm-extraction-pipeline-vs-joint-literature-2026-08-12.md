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

## 当前可检验的架构假设

建议后续以相同的书籍 evidence chunk、Schema 和人工标注样本比较三种实现：

| 方案 | 模型输出 | 本地接纳方式 | 主要风险 |
| --- | --- | --- | --- |
| A. 现有分阶段 | 实体、规则、普通关系、规则边分别调用 | 先冻结实体，再允许后续端点引用 | 实体漏抽导致后续规则/关系不可表达 |
| B. 联合发现，分阶段接纳 | 一次输出全部候选及证据锚点 | 本地先接纳实体，再解析并校验其余候选 | 单次 JSON 更长，实体与边可能不一致 |
| C. 高召回实体补抽闭环 | 常规分阶段；规则/关系阶段可报缺失端点线索 | 缺失端点回到实体补抽，不能直接入图 | 额外调用与回合收敛策略 |

评价指标至少包括：实体/状态召回率、普通关系与规则的证据支持精度、冻结端点失败率、
表格规则召回率、人工复核耗时、每 chunk 的模型调用数和 token 成本。所有对比都应保存
相同粒度的原文锚点；不要以最终图规模替代正确性指标。

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

## 证据边界

- 文献 1、2、4、5 为 arXiv 预印本，应与同行评议结果区分。
- 各论文的数据集、领域、本体约束、模型版本和评价指标不同；不能将报告的分数直接
  外推到医学教材规则抽取。
- “LLM Judge”本身可能误判。它只能作为候选路由或审核辅助，不能覆盖原文回放和人工发布门。
