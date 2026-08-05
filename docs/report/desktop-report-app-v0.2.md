# 桌面循证报告分析 v0.2

该页面是 `medical-kg-qa serve-qa` 的同源桌面工作台，固定绑定 loopback。
它不做 OCR、模型调用、外部检索或患者数据持久化。真实整书 SQLite 和 PDF
均由操作者在 Git 外的本地只读路径提供。

```bash
medical-kg-qa serve-qa \
  --index runtime/active/indexes/full-book-v0.2/evidence.sqlite \
  --chunk-package source-packages/canonical/evidence/full-book-v0.2 \
  --host 127.0.0.1 \
  --port 18852 \
  --source-pdf /absolute/path/to/read-only-book.pdf
```

访问 `http://127.0.0.1:18852/`。服务拒绝非 loopback 地址、未知路由、未知
JSON 字段、超过 16 KiB 的请求和超过 200 个 observation 的报告。响应头固定
包含 CSP、`nosniff` 和 `no-referrer`；`BaseHTTPRequestHandler.log_message` 被禁用。

## 报告 JSON

```json
{
  "schema_version": "structured-report/v0.2",
  "metadata": {
    "hospital": "optional",
    "patient_name": "accepted then discarded",
    "patient_identifier": "accepted then discarded"
  },
  "observations": [
    {
      "raw_name": "synthetic_metric",
      "standard_name": "synthetic_metric",
      "value": "12",
      "unit": "U",
      "reference_interval": {"lower": "1", "upper": "10"},
      "report_flag": "high"
    }
  ]
}
```

允许的 `report_flag` 是 `low`、`normal`、`high`。每项只接受固定字段；非法
数值、区间或单位进入结构化错误并且不会成为 claim。metadata 可携带医院、日期、
科室、样本和有限人口学概览，也可携带姓名、证件和原始文本以兼容上游格式，但服务
在解析后立即丢弃 metadata，绝不把它写进响应、REPORT snapshot、SQLite、日志或文件。
页面不用 cookie、localStorage、sessionStorage 或 indexedDB；刷新和清空都不恢复输入。

`POST /api/report-analysis` 响应包括程序重算的 abnormalities、三值规则 trace、
claims、Evidence gaps 与 CitationBundles。只有已审核的 approved 规则同时具备
REPORT、COMPUTATION 和 BOOK 三链时才可能有医学 claim。当前真实规则数为 0，
所以应用如实返回零条 claim 与“暂无 approved 规则”。

## 证据与限制

整书检索保留 `POST /api/search` 和 `POST /api/answer`。每条返回证据有 chunk ID、
书内页、PDF 页、SQLite 中的 cleaned 字符区间、exact quote 和检索原因。配置
`--chunk-package` 后，服务先用 manifest 的精确 SHA-256 绑定索引，再逐条核对页码、
偏移、文本和哈希；通过后才按同页连续 chunk 计算 1-based cleaned Markdown 页内行号，
并单独返回 manifest 声明的上游来源 Markdown 行范围。两类行号不会混称。

未配置 `--chunk-package` 时保留旧检索能力，但 `location_status` 为 `unavailable`，
行号字段为空；任何索引、manifest、chunk 文件、偏移、页码、文本或哈希漂移都会失败
关闭。证据抽屉逐项展示这些字段，不显示本机路径。配置 `--source-pdf` 时，PDF 仅由
固定 `/source.pdf` 路由读取，抽屉可打开对应原 PDF 页；未配置时仍保留已索引页码。

本版本没有真实 approved 规则。它是本地调试界面，不构成诊断、治疗或临床建议。
