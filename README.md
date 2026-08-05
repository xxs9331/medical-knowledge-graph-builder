# 医学知识图谱构建器

Build a provenance-first medical knowledge graph with hybrid retrieval and controlled graph expansion.

## 项目目标

本项目用于构建以来源追溯为核心的医学知识图谱，为知识抽取、检索、报告分析和证据组织提供可验证的工程基础。运行时结果仍按来源、哈希、候选状态和人工审核边界分层保存。

已确定的架构方向包括：

- 使用 `KnowledgeRule` 统一承载可条件化的医学知识规则。
- 使用 `EvidenceChunk` 保留段落、表格、公式等证据及其来源定位。
- 融合精确匹配、全文检索和向量检索。
- 从检索种子节点按关系白名单进行受控动态扩图，并结合预算、证据覆盖和停止条件控制范围。

## 当前状态

当前版本已实现页级来源准备、页内 EvidenceChunk 切片、统一来源包校验、本地 SQLite 证据索引、词法/向量/图谱检索、v0.2-v0.4 候选语义抽取、第一章 candidate/final 图谱构建，以及结构化报告异常判定和证据约束分析。语义抽取与图谱产物仍遵循 candidate-only、HOLD 和人工审核边界，不代表医学规则已经批准发布。

## 代码结构

源码按业务域组织，公共校验和基础机制只保留一个主实现：

- `provenance/`：来源包、页码、EvidenceChunk、manifest 和 hash/replay 校验。
- `evidence/`：证据索引、词法检索、向量索引，以及明确标记的 `legacy/` 检索实现。
- `extraction/`：LLM 客户端、实体/指标抽取、语义合同、版本化候选验证和工件工具。
- `rules/`：指标规则、组合规则、确定性报告分析和证据策略。
- `graph/`：旧图谱、语义图、章节 candidate/final 图谱、SQLite 存储和图谱遍历。
- `report/`：报告模型、术语、OCR、布局、报告流水线和桌面页面。
- `api/` 与 `cli/`：HTTP 服务和命令行入口。

旧实现保留在 `evidence/legacy/` 或带有 legacy 语义的模块中用于审计和回归测试；新代码不通过旧路径转发。

## 来源准备

`prepare-source` 只将已有的 Unlimited-OCR Markdown 整理为可追溯的页级来源包。它不会运行 OCR、模型、嵌入、实体/关系抽取或图数据库写入。外部 OCR/PDF 项目始终是只读输入；不要把 PDF 或生成的全文来源包提交到 Git。

例如，对当前章节可显式提供已验证的页码映射：

```bash
prepare-source \
  --input /path/to/chapter-01-clinical-hematology.md \
  --output source-packages/canonical/source/chapter-01 \
  --document-id clinical-hematology \
  --chapter-id chapter-01 \
  --ocr-engine baidu/Unlimited-OCR \
  --source-pdf-locator /path/to/chapter-01-clinical-hematology-pages-21-44.pdf \
  --source-pdf-sha256 9e65944d70012466812feeafbb193ca157c31902875cac523cccd2935e80ac66 \
  --printed-page-start 4 \
  --source-pdf-page-start 21 \
  --page-count 24 \
  --page-map /path/to/chapter-01-page-map.json
```

输出包含逐页 raw/cleaned Markdown 与 `manifest.json`。每条页面记录都保留书内页码、原 PDF 页码、章节内索引、稳定页 ID 和内容 SHA-256。对 legacy 合并 Markdown，可提供严格的 `source-page-map/v0.1` JSON：它必须绑定输入文件的精确 SHA-256、声明完整连续且无重叠的 1-based 行范围，并为每页给出与 CLI 一致的三种页码和以下之一的状态：`verified against source PDF`，或 `accepted from upstream page markers`。后者只验证上游 Markdown 页码 marker 的结构，不表示独立 PDF 核验，也不对医学内容或 OCR 准确性作出断言。映射模式还会在 manifest 中原样记录每页选定状态，以及 page-map 的定位与精确 SHA-256；映射无效或处理期间发生字节漂移会在输出提交前失败。未提供映射时，工具仍只接受可自动保守分段的输入；无法证明边界时会失败且不留下部分输出。

章节切片、语义候选抽取、规则候选生成和知识图谱构建都是该来源包的下游消费者；它们必须消费同一套 manifest/page/chunk/hash 校验结果。

## EvidenceChunk 切片

`prepare-chunks` 只将已验证来源包的 cleaned 页面切为页内、无重叠的 EvidenceChunk。每个 chunk 都保留完整来源页记录、字符偏移和内容哈希，因此按页排序后可逐字回放原 cleaned 文本。它不会跨页、添加 overlap、改写文本，或进行 OCR、PDF 处理、模型调用、嵌入、实体/关系/规则抽取或图数据库写入。

```bash
prepare-chunks \
  --source-package /path/to/source-package \
  --output /path/to/local-evidence-chunks \
  --max-chars 1600 \
  --generation-timestamp 2026-01-01T00:00:00Z
```

切片优先在空行、再在换行处分界；HTML table、fenced code 和 `\\[...\\]`/`$$...$$` display math 始终保持完整。单个受保护块超过 `--max-chars` 时会保留该块并记录 `oversize_atomic_block` warning。

本项目不包含患者诊断，不处理患者数据，也不替代临床专业判断。

## 本地证据检索问答

`medical-kg-qa` 只消费已验收的 EvidenceChunk 包，建立 document-page-chunk 证据图与固定边：`DOCUMENT_HAS_PAGE`、`PAGE_HAS_CHUNK`、`CHUNK_NEXT`。构建会校验 manifest、页码来源、chunk ID、文件与 SHA-256，并通过临时 SQLite 文件原子提交。它不抽取医学实体/关系/规则，不调用模型、嵌入、向量库或外部网络。

```bash
medical-kg-qa build-evidence-index \
  --chunk-package /path/to/evidence-chunks \
  --output runtime/active/indexes/chapter-01/evidence.sqlite \
  --generation-timestamp 2026-01-01T00:00:00Z

medical-kg-qa serve-qa --index runtime/active/indexes/chapter-01/evidence.sqlite
```

服务仅绑定 `127.0.0.1`，提供 `GET /api/health`、`GET /api/meta`、`POST /api/search`、`POST /api/answer` 和 `POST /api/report-analysis`，网页在 `/`。默认回答是本地证据原句拼接，并以 `[n]` 绑定返回证据及书内页/PDF 页；无命中时明确说明证据不足。`--answer-mode openai-compatible` 仅在 `MEDICAL_KG_QA_BASE_URL`、`MEDICAL_KG_QA_API_KEY`、`MEDICAL_KG_QA_MODEL`（可选 timeout）均明确设置时启用，并要求返回内容引用提供的证据；缺失、无效或无引用时失败关闭，不会读取或暴露任何本机登录凭据。

## 桌面报告分析

桌面工作台复用同一个 loopback 服务。将真实整书工件保留在 Git 之外，并明确传入索引；可选 PDF 只从固定的本地只读文件以 `/source.pdf` 提供，不会回显本机路径。

```bash
medical-kg-qa serve-qa \
  --index runtime/active/indexes/full-book-v0.2/evidence.sqlite \
  --chunk-package source-packages/canonical/evidence/full-book-v0.2 \
  --knowledge-graph runtime/active/graphs/chapter-01-final-v0.1/knowledge.sqlite \
  --host 127.0.0.1 \
  --port 18852 \
  --source-pdf /absolute/path/to/read-only-book.pdf
```

`--knowledge-graph`当前挂载的是第一章最终图谱。`/api/search`、`/api/answer`和`/api/report-generation`会合并整书词法检索与最终图谱检索，返回`channels.graph`、逐词诊断及有向三元组路径；报告页面会列出缺失缩写、未收录实体和成功匹配映射。最终规则会用报告实际数值、单位和程序状态匹配规则case，并沿显式`RULE_SATISFIES_PRECONDITION`边做受控两跳搜索，例如先由HGB和性别产生贫血程度规则结果，再决定是否满足贫血形态分类的前置条件。图谱路径自己的哈希绑定证据可按chunk ID从只读证据索引注入，不受词法top-k截断；某指标存在图谱锚定证据时，报告分析优先使用这些证据，图谱无证据时才回退整书词法结果。最终证据列表只保留报告正文、关联分析、建议或规则路径实际引用的逐字哈希证据。第一章之外或未命中图节点的查询继续使用整书词法检索；运行时不加载候选图谱。

打开 `http://127.0.0.1:18852/` 后可上传 PNG/JPEG 报告单，或粘贴 `structured-report/v0.2` JSON。图片入口调用 PaddleOCR AI Studio Jobs API：`PaddleOCR-VL-1.6` 提取版面表格，`PP-OCRv6` 独立提取原始文字和非敏感元数据。格式对齐采用固定管线：读取官方 `prunedResult` 版面块、将 HTML 或管道 Markdown 表格归一为带来源单元格 ID 的矩形网格、按通用字段角色发现表头和重复列组、按视觉栏顺序读取、用受控检验术语表规范标准名和缩写、重算参考区间与异常标记，最后校验为 `structured-report/v0.2`。报告术语层会把原始写法`NEUT%`、`LYM%`拆解为规范缩写`NEUT`、`LYM`和“百分数”结果类型，并分别链接到“中性粒细胞百分数”“淋巴细胞百分数”；原始写法继续保留用于溯源，也不会据此推断绝对值项目。对明确登记默认单位的项目可在报告单位缺失时应用受控默认值，目前红细胞计数使用`10^12/L`；输出同时携带`unit_source=controlled_default`和可见提示，未登记项目不会获得猜测单位。这些属于报告实体链接，不伪装成书籍原文别名。它支持表头不在首行、左右双栏以及 `rowspan`/`colspan`，不包含医院或指标专用模板。无法确定的风险分层区间或被水印覆盖的单位保留为空，不猜测；同一来源单元格重复会去重，不同来源却同名则失败关闭。OCR 开始和失败时都会清空上一张报告的 JSON。图片最大 10 MiB，只在当前 HTTP 请求和 PaddleOCR 作业中使用，本服务不落盘。JSON 输入也只在当前请求和浏览器内存中使用：服务不写日志、不写数据库，也不使用浏览器存储。

图片识别凭证只从环境读取：

```bash
export PADDLEOCR_ACCESS_TOKEN=...
export PADDLEOCR_JOB_URL=https://paddleocr.aistudio-app.com/api/v2/ocr/jobs
```

也可单独将图片转换为 JSON：

```bash
medical-report-ocr --image /path/to/report.jpg --output report.json
```

## 结构化报告闭环

可直接对 `structured-report/v0.2` JSON 执行异常指标检索和 DeepSeek 证据约束分析；网页图片入口先生成同一结构的 JSON，再复用此闭环：

```bash
DEEPSEEK_API_KEY=... medical-report analyze \
  --report sample-report.json \
  --index runtime/active/indexes/full-book-v0.2/evidence.sqlite \
  --output analysis.md
```

客户端固定使用 `deepseek-v4-flash`、`temperature=0`、`thinking.type=disabled` 和 JSON 响应格式；代理环境不会被读取。可选 `DEEPSEEK_API_IP` 用于保留 SNI 的固定 IP 连接。

`--chunk-package` 用 manifest 的精确 SHA-256 绑定 SQLite 索引，并在启动时逐条核对 chunk 的页码、字符偏移、文本和哈希。配置成功后，检索证据会同时返回书内页、原 PDF 页、cleaned 字符区间、cleaned Markdown 页内行号和上游来源 Markdown 行范围；任何漂移都会失败关闭。省略该参数时旧检索接口仍可使用，但 `location_status` 明确为 `unavailable`，不会伪造 Markdown 行号。

## 许可证

许可证待用户另行决定。
