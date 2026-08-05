"""Strict desktop report adapter and same-origin workbench assets.

The adapter deliberately accepts only structured observations.  Metadata is
validated only to bound the request and is never returned or retained.
"""

from __future__ import annotations

from typing import Any, Mapping

from .report_model import AbnormalFlag, Observation, ReferenceInterval


REPORT_SCHEMA_VERSION = "structured-report/v0.2"
MAX_OBSERVATIONS = 200
MAX_TEXT_CHARS = 200


class DesktopAppError(ValueError):
    """Raised when a desktop report request is not a bounded contract value."""


def parse_report_payload(value: object) -> dict[str, Observation]:
    """Parse one versioned report without retaining metadata or patient text."""
    if not isinstance(value, dict) or set(value) - {"schema_version", "metadata", "observations"}:
        raise DesktopAppError("report must contain only schema_version, metadata, and observations")
    if value.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise DesktopAppError("unsupported report schema_version")
    metadata = value.get("metadata", {})
    if not isinstance(metadata, dict) or set(metadata) - {
        "hospital", "report_date", "department", "sample_type", "patient_sex",
        "patient_age_years", "patient_name", "patient_identifier", "source_text",
    }:
        raise DesktopAppError("report metadata is invalid")
    observations = value.get("observations")
    if not isinstance(observations, list) or not observations or len(observations) > MAX_OBSERVATIONS:
        raise DesktopAppError("observations must be a non-empty bounded list")
    parsed: dict[str, Observation] = {}
    for item in observations:
        observation = _parse_observation(item)
        key = observation.standard_name or observation.abbreviation or observation.raw_name
        if key in parsed:
            raise DesktopAppError("observation identifiers must be unique")
        parsed[key] = observation
    return parsed


def _parse_observation(value: object) -> Observation:
    if not isinstance(value, dict):
        raise DesktopAppError("observation must be an object")
    allowed = {
        "raw_name", "standard_name", "abbreviation", "value", "unit", "reference_interval",
        "report_flag", "sample_type", "method",
    }
    if set(value) - allowed:
        raise DesktopAppError("observation contains unknown fields")
    raw_name = _text(value.get("raw_name"), "raw_name", required=True)
    standard_name = _text(value.get("standard_name"), "standard_name")
    abbreviation = _text(value.get("abbreviation"), "abbreviation")
    unit = _text(value.get("unit"), "unit")
    sample_type = _text(value.get("sample_type"), "sample_type")
    method = _text(value.get("method"), "method")
    reference = value.get("reference_interval")
    if not isinstance(reference, dict) or set(reference) - {
        "lower", "upper", "lower_inclusive", "upper_inclusive",
    }:
        raise DesktopAppError("reference_interval is invalid")
    lower_inclusive = reference.get("lower_inclusive", True)
    upper_inclusive = reference.get("upper_inclusive", True)
    if not isinstance(lower_inclusive, bool) or not isinstance(upper_inclusive, bool):
        raise DesktopAppError("reference interval inclusivity must be boolean")
    flag = value.get("report_flag")
    if flag is not None and flag not in {item.value for item in AbnormalFlag}:
        raise DesktopAppError("report_flag is invalid")
    return Observation(
        raw_name=raw_name,
        standard_name=standard_name,
        abbreviation=abbreviation,
        value=_number_text(value.get("value"), "value"),
        unit=unit,
        reference_interval=ReferenceInterval(
            lower=_number_text(reference.get("lower"), "reference_interval.lower"),
            upper=_number_text(reference.get("upper"), "reference_interval.upper"),
            lower_inclusive=lower_inclusive,
            upper_inclusive=upper_inclusive,
        ),
        report_flag=AbnormalFlag(flag) if flag else None,
        sample_type=sample_type,
        method=method,
    )


def _text(value: object, field: str, *, required: bool = False) -> str | None:
    if value is None:
        if required:
            raise DesktopAppError(f"{field} is required")
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_TEXT_CHARS:
        raise DesktopAppError(f"{field} must be a bounded string")
    return value.strip()


def _number_text(value: object, field: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise DesktopAppError(f"{field} must be a numeric string or number")
    text = str(value)
    if len(text) > MAX_TEXT_CHARS:
        raise DesktopAppError(f"{field} is too long")
    return text


def html() -> bytes:
    return """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>本地报告分析</title><link rel="stylesheet" href="/assets/app.css"></head><body><header><strong>本地报告分析</strong><span id="index-status">本地证据索引</span><span id="status" role="status">就绪</span></header><main role="main"><nav aria-label="工作台"><button id="report-tab" type="button" aria-selected="true">报告分析</button><button id="search-tab" type="button" aria-selected="false">整书检索</button></nav><div class="workbench"><section id="report-panel"><form id="ocr-form"><label for="report-image">报告单图片</label><input id="report-image" type="file" accept="image/png,image/jpeg" required><button id="ocr-button" type="submit">图片转 JSON</button></form><p id="ocr-status" aria-live="polite">等待图片</p><form id="report-form"><label for="report-json">结构化报告 JSON</label><textarea id="report-json" required spellcheck="false" maxlength="262144"></textarea><div class="commands"><button type="submit">生成分析</button><button type="reset">清空</button></div></form><p id="rule-status" aria-live="polite">等待提交</p></section><section id="result-panel" aria-live="polite"><div class="result-header"><h1>分析报告</h1><div class="view-switch" role="group" aria-label="报告视图"><button id="report-view" type="button" aria-pressed="true">阅读版</button><button id="markdown-view" type="button" aria-pressed="false">Markdown</button></div></div><article id="analysis-result" class="report-output"><p class="empty-state">等待分析</p></article></section></div><section id="search-panel" hidden><form id="query-form"><label for="query">整书检索</label><input id="query" required maxlength="400"><button type="submit">检索</button></form><p id="answer">等待检索</p><ol id="evidence"></ol><div id="provenance" hidden></div></section></main><aside id="drawer" hidden aria-live="polite"><button id="close-drawer" type="button" aria-label="关闭">关闭</button><h2 id="drawer-title">证据</h2><div id="drawer-content"></div></aside><script src="/assets/app.js"></script></body></html>""".encode("utf-8")


def css() -> str:
    return """*,*::before,*::after{box-sizing:border-box}body{margin:0;font:14px system-ui,sans-serif;color:#17212b;background:#f3f5f6;line-height:1.55}header{height:52px;display:flex;align-items:center;gap:16px;padding:0 24px;background:#176b67;color:#fff}#status{margin-left:auto}main{max-width:1440px;margin:0 auto;padding:18px 24px}nav{display:flex;gap:4px;border-bottom:1px solid #c7d0d5}button{min-height:32px;padding:6px 10px;border:1px solid #98a8af;border-radius:4px;background:#fff;color:#17212b;font:inherit;cursor:pointer}button:hover{border-color:#176b67;color:#075d59}button:disabled{cursor:wait;opacity:.6}nav button[aria-selected=true],.view-switch button[aria-pressed=true]{border-color:#176b67;background:#e9f3f2;color:#075d59}.workbench{display:grid;grid-template-columns:380px minmax(0,1fr);gap:16px;margin-top:16px}section{min-width:0}#report-panel,#result-panel,#search-panel{border:1px solid #c7d0d5;background:#fff;padding:16px}form{display:grid;gap:8px}#ocr-form{margin-bottom:4px;padding-bottom:14px;border-bottom:1px solid #dce3e6}#ocr-status{margin:8px 0 16px;color:#53636b}textarea,input{min-width:0;width:100%;max-width:100%;padding:9px;border:1px solid #98a8af;border-radius:3px;font:inherit}textarea{height:580px;resize:vertical;font-family:ui-monospace,monospace}.commands,.result-header,.view-switch,.citation-list,.evidence-row{display:flex;align-items:center;gap:8px}.result-header{justify-content:space-between;padding-bottom:12px;border-bottom:1px solid #dce3e6}h1,h2,h3{margin:0;color:#17212b;letter-spacing:0}h1{font-size:18px}h2{font-size:16px}h3{font-size:15px}.report-output{min-height:620px}.report-notice{margin:16px 0;padding:10px 12px;border-left:3px solid #176b67;background:#f3f7f7;color:#40515a}.analysis-basis{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1px;margin:10px 0 0;background:#dce3e6}.analysis-basis div{min-width:0;padding:9px 10px;background:#f7f9fa}.analysis-basis dt{font-size:12px;color:#53636b}.analysis-basis dd{margin:2px 0 0;font-weight:600;overflow-wrap:anywhere}.report-section{padding:18px 0;border-bottom:1px solid #e1e6e8}.report-section p{margin:8px 0 0;white-space:pre-wrap}.metric-heading{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}.metric-value{color:#53636b;font-size:13px}.flag-high{color:#a13d2d}.flag-low{color:#25649a}.citation-list{margin-top:10px;flex-wrap:wrap}.citation-button{min-height:28px;padding:3px 8px;border-color:#9bb9b6;color:#075d59}.report-list{margin:8px 0 0;padding-left:20px}.report-list li{margin:8px 0}.evidence-row{justify-content:space-between;padding:10px 0;border-bottom:1px solid #e1e6e8}.evidence-location{color:#53636b}.markdown-output{margin:16px 0 0;padding:14px;border:1px solid #dce3e6;background:#f7f9fa;white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.6 ui-monospace,monospace}.empty-state{color:#6b7880}li{overflow-wrap:anywhere}#drawer{position:fixed;z-index:2;top:0;right:0;width:min(620px,48vw);height:100%;overflow:auto;padding:20px;border-left:1px solid #9cabb1;background:#fff;box-shadow:-4px 0 12px #0002}#close-drawer{float:right}#drawer-content{clear:both;overflow-wrap:anywhere}.evidence-quote{margin:0 0 16px;padding:12px;border-left:3px solid #176b67;background:#f3f7f7;white-space:pre-wrap;overflow-wrap:anywhere}.evidence-fields{display:grid;grid-template-columns:160px minmax(0,1fr);margin:0}.evidence-fields dt,.evidence-fields dd{margin:0;padding:7px 0;border-bottom:1px solid #e1e6e8}.evidence-fields dt{font-weight:600;color:#40515a}.evidence-fields dd{white-space:pre-wrap;overflow-wrap:anywhere}.pdf-link{display:inline-block;margin-top:16px;color:#075d59}@media(max-width:900px){.workbench{grid-template-columns:1fr}textarea{height:320px}#drawer{width:min(620px,90vw)}}@media(max-width:620px){.analysis-basis{grid-template-columns:1fr}}@media(max-width:520px){form label,form input{grid-column:1/-1}}"""


def javascript() -> str:
    return """const $=selector=>document.querySelector(selector);
const node=(tag,value)=>{const element=document.createElement(tag);if(value!==undefined)element.textContent=value;return element};
const show=(selector,visible)=>{$(selector).hidden=!visible};
const unavailable=value=>value===null||value===undefined||value===''?'不可用':String(value);
const range=(start,end)=>Number.isInteger(start)&&Number.isInteger(end)?`${start}-${end}`:'不可用';
const tripleText=triple=>`${triple.subject_name} -[${triple.predicate}]-> ${triple.object_name}`;
let reportState=null;
let reportMode='report';
const evidenceMap=()=>new Map((reportState?.evidence||[]).map(item=>[item.evidence_id,item]));
const drawer=(title,value)=>{$('#drawer-title').textContent=title;$('#drawer-content').replaceChildren(node('pre',value));show('#drawer',true)};
const evidenceDrawer=evidence=>{
  $('#drawer-title').textContent='书内证据';
  const heading=node('h3','原文摘录');
  const quote=node('blockquote',unavailable(evidence.exact_quote??evidence.text));
  quote.className='evidence-quote';
  const fields=document.createElement('dl');
  fields.className='evidence-fields';
  const values=[
    ['证据编号',unavailable(evidence.evidence_id)],
    ['证据块 ID',unavailable(evidence.chunk_id)],
    ['书内页',unavailable(evidence.printed_page_number)],
    ['原 PDF 页',unavailable(evidence.source_pdf_page_number)],
    ['Cleaned 字符区间（左闭右开）',range(evidence.cleaned_char_start,evidence.cleaned_char_end)],
    ['Cleaned Markdown 页内行',range(evidence.markdown_line_start,evidence.markdown_line_end)],
    ['上游来源 Markdown 行范围',range(evidence.source_page_line_start,evidence.source_page_line_end)],
    ['检索原因',unavailable(evidence.retrieval_reason)],
    ['图谱状态',unavailable(evidence.graph?.status)],
    ['命中图节点',unavailable(evidence.graph?.matched_node_names?.join('、'))],
    ['图路径',unavailable(evidence.graph?.path_relations?.join(' → ')||(
      evidence.graph?'直接命中':''))],
    ['实体链接方式',unavailable(evidence.graph?.match_mode)],
    ['检索评分',unavailable(evidence.score)],
    ['内容哈希',unavailable(evidence.chunk_sha256)],
  ];
  values.filter(([,value])=>value!=='不可用').forEach(([label,value])=>fields.append(node('dt',label),node('dd',value)));
  const children=[heading,quote,fields];
  if(evidence.graph?.path_triples?.length){
    children.push(node('h3','有向三元组路径'));
    const triples=node('ol');
    evidence.graph.path_triples.forEach(item=>triples.append(node('li',tripleText(item))));
    children.push(triples);
  }
  if(Number.isInteger(evidence.source_pdf_page_number)){
    const link=node('a',`打开原 PDF 第 ${evidence.source_pdf_page_number} 页`);
    link.className='pdf-link';
    link.href=`/source.pdf#page=${encodeURIComponent(String(evidence.source_pdf_page_number))}`;
    link.target='_blank';
    link.rel='noreferrer';
    children.push(link);
  }
  $('#drawer-content').replaceChildren(...children);
  show('#drawer',true);
};
const citations=ids=>{
  const wrap=node('div');wrap.className='citation-list';
  const byId=evidenceMap();
  (ids||[]).forEach((id,index)=>{
    const evidence=byId.get(id);
    if(!evidence)return;
    const graphLabel=evidence.graph?' · 图谱辅助召回':'';
    const button=node('button',`证据 ${index+1} · 书内 ${evidence.printed_page_number} 页 · PDF ${evidence.source_pdf_page_number} 页${graphLabel}`);
    button.type='button';button.className='citation-button';
    button.addEventListener('click',()=>evidenceDrawer(evidence));
    wrap.append(button);
  });
  return wrap;
};
const reportSection=(title,text,ids)=>{
  const section=node('section');section.className='report-section';
  section.append(node('h2',title),node('p',text));
  if(ids?.length)section.append(citations(ids));
  return section;
};
const analysisBasis=channels=>{
  const section=node('section');section.className='report-section';section.append(node('h2','分析依据'));
  const list=node('dl');list.className='analysis-basis';
  const graph=channels?.graph||{};
  const values=[
    ['异常判定','程序重算'],
    ['书内检索',`全书 · ${channels?.lexical?.evidence_count||0} 条证据`],
    ['知识图谱',graph.enabled?`第一章候选图谱 · 辅助召回 ${graph.evidence_count||0} 条证据 · 候选路径 ${graph.reasoning_path_count||0} 条`:'未启用'],
    ['缺词诊断',graph.enabled?Object.entries(graph.query_diagnostic_counts||{}).map(([key,value])=>`${key} ${value}`).join(' · ')||'无查询':'未启用'],
  ];
  values.forEach(([label,value])=>{const item=node('div');item.append(node('dt',label),node('dd',value));list.append(item)});
  section.append(list);return section;
};
const renderReport=()=>{
  const output=$('#analysis-result');
  if(!reportState){output.replaceChildren(node('p','等待分析'));return}
  if(reportMode==='markdown'){
    const pre=node('pre',reportState.markdown);pre.className='markdown-output';output.replaceChildren(pre);return;
  }
  const report=reportState.report;
  const metrics=new Map(reportState.metrics.map(item=>[item.metric_id,item]));
  const children=[];
  const graphEnabled=Boolean(reportState.channels?.graph?.enabled);
  const notice=node('blockquote',`基于程序异常判定、整书检索证据${graphEnabled?'和第一章候选知识图谱辅助召回':''}生成，不构成诊断、治疗或用药建议。`);notice.className='report-notice';children.push(notice);
  children.push(analysisBasis(reportState.channels));
  children.push(reportSection('摘要',report.summary,[]));
  (report.abnormal_analyses||[]).forEach(item=>{
    const metric=metrics.get(item.metric_id)||{};
    const section=node('section');section.className='report-section';
    const heading=node('div');heading.className='metric-heading';
    const title=node('h2',metric.raw_name||item.metric_id);
    if(metric.computed_flag)title.className=`flag-${metric.computed_flag}`;
    const flagLabel=metric.computed_flag==='high'?'偏高':metric.computed_flag==='low'?'偏低':'无法判定';
    heading.append(title,node('span',`${unavailable(metric.value)} ${metric.unit||''} · ${flagLabel}`));
    heading.lastChild.className='metric-value';
    section.append(heading,node('p',item.analysis));
    if(item.evidence_ids?.length)section.append(citations(item.evidence_ids));
    children.push(section);
  });
  children.push(reportSection('关联分析',report.association_analysis.analysis,report.association_analysis.evidence_ids));
  if(reportState.reasoning_paths?.length){
    const section=node('section');section.className='report-section';section.append(node('h2','候选推理路径'));
    const warning=node('blockquote','以下路径只用于合并书内关联，未执行图谱规则，不构成诊断。');warning.className='report-notice';section.append(warning);
    reportState.reasoning_paths.forEach(path=>{
      const item=node('div');item.className='reasoning-path';
      item.append(node('h3',path.rule_name||'候选规则'));
      item.append(node('p',`${path.status} · 共同命中：${(path.matched_metric_ids||[]).join('、')}`));
      const triples=node('ol');(path.triples||[]).forEach(triple=>triples.append(node('li',tripleText(triple))));item.append(triples);
      if(path.evidence_ids?.length)item.append(citations(path.evidence_ids));
      section.append(item);
    });children.push(section);
  }
  if(report.attention_suggestions?.length){
    const section=node('section');section.className='report-section';section.append(node('h2','关注建议'));
    const list=node('ul');list.className='report-list';
    report.attention_suggestions.forEach(item=>{const li=node('li');li.append(node('span',item.text),citations(item.evidence_ids));list.append(li)});
    section.append(list);children.push(section);
  }
  if(report.insufficient_evidence?.length){
    const section=node('section');section.className='report-section';section.append(node('h2','证据不足'));
    const list=node('ul');list.className='report-list';report.insufficient_evidence.forEach(item=>list.append(node('li',item)));section.append(list);children.push(section);
  }
  const metricsToCheck=reportState.metrics.filter(item=>item.validation_issues?.length);
  if(metricsToCheck.length){
    const section=node('section');section.className='report-section';section.append(node('h2','数据待核对'));
    const list=node('ul');list.className='report-list';
    metricsToCheck.forEach(item=>list.append(node('li',`${item.raw_name}：${item.validation_issues.map(issue=>issue.label).join('；')}`)));
    section.append(list);children.push(section);
  }
  if(reportState.evidence.length){
    const section=node('section');section.className='report-section';section.append(node('h2','书内证据'));
    reportState.evidence.forEach((item,index)=>{const row=node('div');row.className='evidence-row';const graphLabel=item.graph?' · 图谱辅助召回':'';const label=node('span',`[${index+1}] 书内 ${item.printed_page_number} 页 · PDF ${item.source_pdf_page_number} 页${graphLabel}`);label.className='evidence-location';const button=node('button','查看证据');button.type='button';button.addEventListener('click',()=>evidenceDrawer(item));row.append(label,button);section.append(row)});children.push(section);
  }
  output.replaceChildren(...children);
};
const setReportMode=mode=>{reportMode=mode;$('#report-view').setAttribute('aria-pressed',String(mode==='report'));$('#markdown-view').setAttribute('aria-pressed',String(mode==='markdown'));renderReport()};
const reportError=message=>/evidence|citation/i.test(message)?'生成结果未通过证据引用校验，请重新生成。':message;
const clear=()=>{reportState=null;reportMode='report';renderReport();$('#rule-status').textContent='等待提交';show('#drawer',false)};
const fileBase64=async file=>{const bytes=new Uint8Array(await file.arrayBuffer());let binary='';for(let offset=0;offset<bytes.length;offset+=32768)binary+=String.fromCharCode(...bytes.subarray(offset,offset+32768));return btoa(binary)};
$('#report-tab').addEventListener('click',()=>{show('#report-panel',true);show('#result-panel',true);show('#search-panel',false);$('#report-tab').setAttribute('aria-selected','true');$('#search-tab').setAttribute('aria-selected','false')});
$('#search-tab').addEventListener('click',()=>{show('#report-panel',false);show('#result-panel',false);show('#search-panel',true);$('#report-tab').setAttribute('aria-selected','false');$('#search-tab').setAttribute('aria-selected','true')});
$('#report-view').addEventListener('click',()=>setReportMode('report'));
$('#markdown-view').addEventListener('click',()=>setReportMode('markdown'));
$('#close-drawer').addEventListener('click',()=>show('#drawer',false));
$('#report-form').addEventListener('reset',clear);
$('#ocr-form').addEventListener('submit',async event=>{event.preventDefault();const file=$('#report-image').files[0];if(!file)return;$('#report-json').value='';clear();if(file.size>10*1024*1024){$('#ocr-status').textContent='图片不能超过 10 MiB';return}const button=$('#ocr-button');button.disabled=true;$('#status').textContent='识别中';$('#ocr-status').textContent='正在识别报告单';try{const response=await fetch('/api/report-ocr',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filename:file.name,content_base64:await fileBase64(file)})});const data=await response.json();if(!response.ok)throw Error(data.detail||'识别失败');$('#report-json').value=JSON.stringify(data.report,null,2);const models=data.job.validation_model?`${data.job.model} + ${data.job.validation_model}`:data.job.model;$('#ocr-status').textContent=`识别完成 · ${data.report.observations.length} 项指标 · ${models}`;$('#status').textContent='完成'}catch(error){$('#report-json').value='';$('#ocr-status').textContent=error.message;$('#status').textContent='失败'}finally{button.disabled=false}});
$('#report-form').addEventListener('submit',async event=>{event.preventDefault();$('#status').textContent='分析中';$('#rule-status').textContent='正在合并书内检索与图谱证据';$('#analysis-result').replaceChildren(node('p','请稍候…'));try{const response=await fetch('/api/report-generation',{method:'POST',headers:{'Content-Type':'application/json'},body:$('#report-json').value});const data=await response.json();if(!response.ok)throw Error(data.detail||'请求失败');reportState=data;setReportMode('report');const graphCount=data.channels?.graph?.evidence_count||0;const pathCount=data.channels?.graph?.reasoning_path_count||0;$('#rule-status').textContent=data.channels?.graph?.enabled?`分析完成 · 图谱辅助召回 ${graphCount} 条证据 · 候选路径 ${pathCount} 条`:'分析完成 · 书内检索';$('#status').textContent='完成'}catch(error){reportState=null;$('#analysis-result').replaceChildren(node('p',reportError(error.message)));$('#rule-status').textContent='分析失败';$('#status').textContent='失败'}});
$('#query-form').addEventListener('submit',async event=>{event.preventDefault();$('#status').textContent='检索中';try{const response=await fetch('/api/answer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:$('#query').value})});const data=await response.json();if(!response.ok)throw Error(data.detail||'请求失败');$('#answer').textContent=data.answer;$('#evidence').replaceChildren(...data.evidence.map((item,index)=>{const row=node('li',`[${index+1}] 书内第${item.printed_page_number}页 / PDF第${item.source_pdf_page_number}页`);const button=node('button','查看证据');button.type='button';button.addEventListener('click',()=>evidenceDrawer(item));row.append(button);return row}));$('#status').textContent='完成'}catch(error){$('#answer').textContent=error.message;$('#status').textContent='失败'}});"""
