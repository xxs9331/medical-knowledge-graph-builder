# 第一章人工抽取测试集

## 文件

- `chapter-01-graph-test-set-v0.1.json`：第一章人工图标注源，按 8 个章节主题组织。
- `chapter-01-entity-test-set-v0.1.json`：实体投影测试集。
- `chapter-01-relationship-test-set-v0.1.json`：普通关系投影测试集。
- `chapter-01-rule-test-set-v0.1.json`：规则投影测试集。
- `build_views.py`：从图标注源机械生成后三个投影，防止四套数据不一致。

## 状态与边界

- 当前状态是 `HUMAN_REVIEW_REQUIRED`：内容由人工阅读第一章后抽取，但尚未经过用户逐条验收。
- 44 个规范 EvidenceChunk 均被纳入某个测试单元；标题块可以只提供上下文，不强制产生图元素。
- 数据集测试候选图中的业务对象，不重复保存 `exact_quote`、字符偏移和哈希；这些证据字段继续由确定性校验器逐候选验证。
- 表格、公式、阈值、趋势及联合条件进入规则集，不降级为单输入到结论的普通关系。
- “见于”“可见于”默认标为 `ASSOCIATED_WITH`；仅在原文明示机制或因果时使用 `CAUSES`。
- ABO 表格存在 OCR 歧义，当前只保留正文明确说明和可无歧义复核的规则，疑似错误单元不作为强制金标。

## 重新生成投影

```bash
.venv/bin/python evaluation/chapter-01/build_views.py
```

生成后应运行：

```bash
.venv/bin/python -m json.tool evaluation/chapter-01/chapter-01-graph-test-set-v0.1.json >/dev/null
.venv/bin/python -m unittest discover -s tests
```
