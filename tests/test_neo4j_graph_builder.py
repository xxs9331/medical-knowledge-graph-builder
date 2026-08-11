import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from neo4j_graphrag.experimental.components.types import Neo4jGraph, Neo4jRelationship

from medical_kg_sourceprep.extraction import neo4j_graph_builder as graph_builder
from medical_kg_sourceprep.extraction.llm_extraction import EvidenceChunk


class Neo4jGraphBuilderTests(unittest.TestCase):
    def test_missing_deepseek_key_fails_without_loading_local_environment(self):
        with self.assertRaisesRegex(graph_builder.GraphBuilderConfigurationError, "DEEPSEEK_API_KEY"):
            graph_builder.load_deepseek_api_key(env={})

    def test_factory_uses_official_deepseek_contract_and_no_proxy_client(self):
        async_client = object()
        llm = object()
        with (
            patch.object(graph_builder.httpx, "AsyncClient", return_value=async_client) as client_factory,
            patch.object(graph_builder, "OpenAILLM", return_value=llm) as llm_factory,
        ):
            result = graph_builder.create_deepseek_graph_builder(
                env={"DEEPSEEK_API_KEY": "test-key", "HTTPS_PROXY": "http://proxy.invalid"}
            )

        self.assertIs(result.llm, llm)
        self.assertIs(result.http_client, async_client)
        self.assertFalse(client_factory.call_args.kwargs["trust_env"])
        self.assertEqual(
            llm_factory.call_args.kwargs,
            {
                "model_name": "deepseek-v4-flash",
                "model_params": {
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "extra_body": {"thinking": {"type": "disabled"}},
                },
                "api_key": "test-key",
                "base_url": "https://api.deepseek.com",
                "http_client": async_client,
            },
        )

    def test_smoke_parses_graph_builder_json_without_writing_any_database(self):
        class FakeLLM:
            async def ainvoke(self, prompt):
                self.prompt = prompt
                return SimpleNamespace(
                    content=(
                        '{"nodes":[{"id":"indicator","label":"LabIndicator",'
                        '"properties":{"name":"血清铁"}}],"relationships":[]}'
                    )
                )

        client = graph_builder.DeepSeekGraphBuilderClient(
            llm=FakeLLM(), http_client=SimpleNamespace()
        )
        summary = asyncio.run(graph_builder.run_smoke(client))

        self.assertEqual(summary["model"], "deepseek-v4-flash")
        self.assertEqual(summary["node_count"], 1)
        self.assertEqual(summary["relationship_count"], 0)

    def test_close_delegates_to_graph_rag_llm(self):
        class FakeLLM:
            def __init__(self):
                self.closed = False

            async def aclose(self):
                self.closed = True

        llm = FakeLLM()
        client = graph_builder.DeepSeekGraphBuilderClient(llm=llm, http_client=SimpleNamespace())
        asyncio.run(client.aclose())
        self.assertTrue(llm.closed)

    def test_candidate_graph_validates_nodes_relations_and_writes_no_raw_response(self):
        text = "血清铁降低。生长发育需要量多导致缺铁性贫血。"
        chunk = EvidenceChunk(
            "synthetic:0001",
            text,
            hashlib.sha256(text.encode()).hexdigest(),
        )
        indicator_ref = graph_builder._source_ref(chunk, "血清铁", "血清铁降低。")
        context_ref = graph_builder._source_ref(chunk, "生长发育需要量多", "生长发育需要量多导致缺铁性贫血。")
        indicator_key = graph_builder._candidate_key("LabIndicator", "血清铁", indicator_ref)
        context_key = graph_builder._candidate_key("ClinicalContext", "生长发育需要量多", context_ref)
        disease_key = graph_builder._candidate_key("Disease", "缺铁性贫血", context_ref)

        class FakeLLM:
            def __init__(self):
                self.calls = 0

            async def ainvoke(self, prompt):
                self.calls += 1
                if self.calls == 1:
                    return SimpleNamespace(content=json.dumps({"nodes": [
                        {"label": "LabIndicator", "properties": {
                            "mention": "血清铁", "canonical_name_candidate": "血清铁", "exact_quote": "血清铁降低。"
                        }},
                        {"label": "IndicatorState", "properties": {
                            "mention": "血清铁降低", "canonical_name_candidate": "血清铁降低", "exact_quote": "血清铁降低。",
                            "bound_indicator_mention": "血清铁"
                        }},
                        {"label": "ClinicalContext", "properties": {
                            "mention": "生长发育需要量多", "canonical_name_candidate": "生长发育需要量多", "exact_quote": "生长发育需要量多导致缺铁性贫血。"
                        }},
                        {"label": "Disease", "properties": {
                            "mention": "缺铁性贫血", "canonical_name_candidate": "缺铁性贫血", "exact_quote": "生长发育需要量多导致缺铁性贫血。"
                        }}
                    ], "relationships": []}, ensure_ascii=False))
                return SimpleNamespace(content=json.dumps({"nodes": [], "relationships": [{
                    "type": "CAUSES", "start_node_id": context_key, "end_node_id": disease_key,
                    "exact_quote": "生长发育需要量多导致缺铁性贫血。", "relation_cue": "导致"
                }]}, ensure_ascii=False))

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "candidate"
            schema_path = Path(temporary) / "candidate-schema.json"
            schema_value = graph_builder.load_candidate_graph_schema()
            schema_value["schema_version"] = "test-custom-schema"
            schema_path.write_text(json.dumps(schema_value, ensure_ascii=False), encoding="utf-8")
            schema = graph_builder.load_candidate_graph_schema(schema_path)
            schema_sha256 = hashlib.sha256(schema_path.read_bytes()).hexdigest()
            summary = asyncio.run(graph_builder.run_candidate_graph(
                graph_builder.DeepSeekGraphBuilderClient(llm=FakeLLM(), http_client=SimpleNamespace()),
                chunk=chunk,
                schema=schema,
                schema_path=schema_path,
                output_dir=output,
                source_manifest_sha256="a" * 64,
            ))
            graph = json.loads((output / "graph.json").read_text(encoding="utf-8"))
            manifest = json.loads((output / "run-manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["node_count"], 4)
        self.assertEqual(summary["relationship_count"], 2)
        self.assertEqual(summary["hold_count"], 0)
        self.assertEqual({item["candidate_key"] for item in graph["nodes"]}, {
            indicator_key,
            context_key,
            disease_key,
            next(item["candidate_key"] for item in graph["nodes"] if item["entity_type"] == "IndicatorState"),
        })
        self.assertEqual({item["relation_type"] for item in graph["relationships"]}, {"HAS_STATE", "CAUSES"})
        self.assertEqual(manifest["configuration"]["database_write"], False)
        self.assertEqual(
            manifest["input"]["candidate_schema_sha256"],
            schema_sha256,
        )
        self.assertNotIn("raw_response", json.dumps(graph, ensure_ascii=False))

    def test_candidate_graph_routes_invalid_nodes_and_relations_to_hold(self):
        text = "HGB 参考区间。普通疾病。甲和乙导致丙。"
        chunk = EvidenceChunk("synthetic:0002", text, hashlib.sha256(text.encode()).hexdigest())
        schema = graph_builder.load_candidate_graph_schema()

        class FakeLLM:
            def __init__(self):
                self.calls = 0

            async def ainvoke(self, prompt):
                self.calls += 1
                if self.calls == 1:
                    return SimpleNamespace(content=json.dumps({"nodes": [
                        {"id": "range", "label": "LabIndicator", "properties": {
                            "mention": "参考区间", "canonical_name_candidate": "参考区间", "exact_quote": "HGB 参考区间。"
                        }},
                        {"id": "claim", "label": "Claim", "properties": {
                            "mention": "HGB", "canonical_name_candidate": "HGB", "exact_quote": "HGB 参考区间。"
                        }},
                        {"id": "not-a-rule", "label": "RuleDefinition", "properties": {
                            "mention": "普通疾病", "canonical_name_candidate": "普通疾病", "exact_quote": "普通疾病。"
                        }},
                        {"id": "state", "label": "IndicatorState", "properties": {
                            "mention": "丙", "canonical_name_candidate": "丙", "exact_quote": "甲和乙导致丙。",
                            "bound_indicator_mention": "不存在"
                        }}
                    ], "relationships": []}, ensure_ascii=False))
                return SimpleNamespace(content=json.dumps({"nodes": [], "relationships": [{
                    "type": "CAUSES", "start_node_id": "unknown", "end_node_id": "other",
                    "properties": {"exact_quote": "甲和乙导致丙。", "relation_cue": "导致"}
                }]}, ensure_ascii=False))

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "candidate"
            summary = asyncio.run(graph_builder.run_candidate_graph(
                graph_builder.DeepSeekGraphBuilderClient(llm=FakeLLM(), http_client=SimpleNamespace()),
                chunk=chunk,
                schema=schema,
                output_dir=output,
                source_manifest_sha256="b" * 64,
            ))
            queue = json.loads((output / "review-queue.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["node_count"], 0)
        self.assertEqual(summary["relationship_count"], 0)
        self.assertEqual(queue["status"], "HOLD")
        self.assertEqual({item["reason_code"] for item in queue["items"]}, {
            "rule_content_not_enabled_for_trial",
            "entity_type_not_enabled_for_trial",
            "rule_definition_not_explicit",
            "state_indicator_binding_not_unique",
            "relation_endpoint_not_from_frozen_catalog",
        })

    def test_candidate_graph_preserves_rule_definition_inputs_and_outputs(self):
        text = "血清铁降低且TIBC增高时，提示缺铁性贫血。"
        chunk = EvidenceChunk("synthetic:rule", text, hashlib.sha256(text.encode()).hexdigest())
        schema = graph_builder.load_candidate_graph_schema()
        source_ref = graph_builder._source_ref(chunk, "血清铁", text)
        state_key = graph_builder._candidate_key("IndicatorState", "血清铁降低", source_ref)
        tibc_state_key = graph_builder._candidate_key("IndicatorState", "TIBC增高", source_ref)
        disease_key = graph_builder._candidate_key("Disease", "缺铁性贫血", source_ref)
        rule_mention = "血清铁降低且TIBC增高时，提示缺铁性贫血"
        rule_key = graph_builder._candidate_key("RuleDefinition", rule_mention, source_ref)

        class FakeLLM:
            def __init__(self):
                self.calls = 0

            async def ainvoke(self, prompt):
                self.calls += 1
                if self.calls == 1:
                    nodes = [
                        ("LabIndicator", "血清铁", {}),
                        ("IndicatorState", "血清铁降低", {"bound_indicator_mention": "血清铁"}),
                        ("LabIndicator", "TIBC", {}),
                        ("IndicatorState", "TIBC增高", {"bound_indicator_mention": "TIBC"}),
                        ("Disease", "缺铁性贫血", {}),
                        ("RuleDefinition", rule_mention, {"rule_stage_candidate": "GRAPH_COMPOSITE"}),
                    ]
                    return SimpleNamespace(content=json.dumps({"nodes": [
                        {"label": label, "properties": {
                            "mention": mention, "canonical_name_candidate": mention, "exact_quote": text, **extra
                        }} for label, mention, extra in nodes
                    ], "relationships": []}, ensure_ascii=False))
                relationships = [
                    ("RULE_INPUT", state_key, rule_key),
                    ("RULE_INPUT", tibc_state_key, rule_key),
                    ("RULE_OUTPUT", rule_key, disease_key),
                ]
                return SimpleNamespace(content=json.dumps({"nodes": [], "relationships": [
                    {"type": relation_type, "start_node_id": source_key, "end_node_id": target_key,
                     "properties": {"exact_quote": text}}
                    for relation_type, source_key, target_key in relationships
                ]}, ensure_ascii=False))

        with tempfile.TemporaryDirectory() as temporary:
            summary = asyncio.run(graph_builder.run_candidate_graph(
                graph_builder.DeepSeekGraphBuilderClient(llm=FakeLLM(), http_client=SimpleNamespace()),
                chunk=chunk,
                schema=schema,
                output_dir=Path(temporary) / "candidate",
                source_manifest_sha256="c" * 64,
            ))

        self.assertEqual(summary["node_count"], 6)
        self.assertEqual(summary["relationship_count"], 5)
        self.assertEqual(summary["hold_count"], 0)
        self.assertEqual(
            {item["relation_type"] for item in summary["relationships"]},
            {"HAS_STATE", "RULE_INPUT", "RULE_OUTPUT"},
        )

    def test_joint_condition_is_not_split_into_a_simple_causal_relation(self):
        text = "甲和乙导致丙。"
        chunk = EvidenceChunk("synthetic:0003", text, hashlib.sha256(text.encode()).hexdigest())
        source_ref = graph_builder._source_ref(chunk, "甲", text)
        source = {
            "candidate_key": "candidate:context",
            "entity_type": "ClinicalContext",
            "mention": "甲",
            "canonical_name_candidate": "甲",
            "source_ref": source_ref,
        }
        target = {
            "candidate_key": "candidate:disease",
            "entity_type": "Disease",
            "mention": "丙",
            "canonical_name_candidate": "丙",
            "source_ref": graph_builder._source_ref(chunk, "丙", text),
        }
        graph = Neo4jGraph(relationships=[Neo4jRelationship(
            start_node_id=f"{chunk.chunk_id}:{source['candidate_key']}",
            end_node_id=f"{chunk.chunk_id}:{target['candidate_key']}",
            type="CAUSES",
            properties={"exact_quote": text, "relation_cue": "导致"},
        )])
        relations, holds = graph_builder.normalize_candidate_relationships(
            graph,
            chunk=chunk,
            schema=graph_builder.load_candidate_graph_schema(),
            nodes=[source, target],
        )

        self.assertEqual(relations, [])
        self.assertEqual([item["reason_code"] for item in holds], ["relation_may_be_joint_condition"])

    def test_trial_section_text_excludes_reference_range_and_next_heading(self):
        text = "前文\n【参考区间】\n(1) 血清铁降低\n仅保留\n(2) 血清铁升高\n后文"
        chunk = EvidenceChunk("synthetic:0004", text, hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(graph_builder.trial_section_text(chunk), "(1) 血清铁降低\n仅保留")

    def test_candidate_prompts_require_complete_evidence_and_explicit_causality(self):
        self.assertIn("complete sentence", graph_builder.NODE_PROMPT_TEMPLATE)
        self.assertIn("explicit \"A 导致 B\"", graph_builder.RELATION_PROMPT_TEMPLATE)
