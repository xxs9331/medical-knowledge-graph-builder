# 工件清理记录

日期：2026-08-05

本记录对应 `docs/runtime/artifact-registry.json`。删除前已生成 source、runtime、knowledge、tmp 清单和 SHA-256 快照；快照保留在本次执行环境的临时审计目录中，不进入 Git。来源包、SQLite、模型输出和运行缓存均未加入 Git。

## 删除记录

### 完全重复的候选图谱

- 原路径：`runtime/chapter-01-knowledge-graph-v0.3`
- 保留替代：`runtime/archive/graph/chapter-01-graph-v0.2`
- 校验：`diff -qr` 无差异；节点/边为 630/488。
- 关键文件 SHA-256：`graph.json` = `03bf4c230f5bd0fed8c47af2dc00ccafc7965b8e467da3d3082a17d39516bf96`，`triples.json` = `342859bda22775c02271540efb4ccfa535e356231032242ea064ce495858bd57`，`knowledge.sqlite` = `517691c91b9c2e13c191614ef8a735c85d36a51f2676348a6821d83858a3dc30`，`run-manifest.json` = `8b0774ba16d46ce1c92710f842427a24121a23b21b7fa8e69b6f38ed2535551d`。
- 原因：字节级完全重复，不承担新的页码、chunk ID、schema 或审核责任。
- 可重建方式：使用 v0.2 图谱构建输入重新执行对应 graph pipeline。

### 可重建比较缓存

- 原路径：`tmp/kg-extractor-comparison-v0.1`
- 规模：20,020 个文件，约 485 MB。
- 清理原因：仅为比较流程生成，不被 CLI、HTTP 服务、索引或图谱引用。
- 可重建方式：重新执行 comparison workflow。
- SHA-256 追踪：删除前快照按相对路径保存每个文件的 SHA-256；本条目同时以文件数、大小和快照清单作为完整性摘要，避免把 485 MB 的生成内容或模型输出提交到 Git。

### Label Studio 无版本别名

- `runtime/candidates/chapter-01/indicator-library-v0.1/label-studio-tasks.json`
  - SHA-256：`24fefb80642b2bb95fad759ad67873bb60178ca720e7661820f34b255830a2f0`
  - 等同于：`label-studio-tasks-v0.5-indicator-ner-only.json`
- `runtime/candidates/chapter-01/indicator-library-v0.1/label-studio-config.xml`
  - SHA-256：`a1e902ab2efa312a6ef8660775cd4db191d31f67aca0f14cf431945dd5396703`
  - 等同于：`label-studio-config-v0.5-indicator-ner-only.xml`
- 原因：保留带版本名的文件即可，删除无版本别名不会改变任务或标注内容。

### 已有完整版本替代的 checkpoint

以下文件仅为重试前、网络探测前或不完整状态，完整 v0.4 checkpoint 仍保留：

| 文件 | SHA-256 |
| --- | --- |
| `rule-checkpoint.pre-hard-timeout.json` | `aced6bb86b4407e0d482020840a80935445d89a7f6317b12967f35bee070653f` |
| `rule-checkpoint.pre-ip-pin.json` | `724378e1d4ec74b8985fb2af0166418159568d591248d735229129334825ccbd` |
| `rule-checkpoint.pre-ip-switch.json` | `ef1a2f793bfb46b1e6aa66c7add153ae812f139a6a35ab3286f4f2d611aae96a` |
| `rule-checkpoint.v0.4.1-8192-partial.json` | `61aee1c6ecc35add1f4a578406ce0c7c48d8cd94e9bc0139696ff95b61ddc8e9` |

## 保留边界

不同版本的 semantic extraction、entity/rule pipeline、candidate graph v0.1/v0.2/v0.4、full-book v0.3 calibration/extraction attempts 均已移动到 `runtime/archive/`，没有删除。当前 HTTP 服务只使用 `runtime/active/` 下的 evidence index 和 final graph；candidate 图谱仍保持 candidate-only/HOLD，不作为最终运行时输入。
