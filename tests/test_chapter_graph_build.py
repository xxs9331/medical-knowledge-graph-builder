import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from medical_kg_sourceprep.chapter_graph_build import (
    ChapterGraphBuilder,
    build_entity_evidence,
    validate_rule_evidence,
    write_graph_package,
)


class ChapterGraphBuildTests(unittest.TestCase):
    def packages(self):
        entities = [
            {"category": "LabTest", "name": "血红蛋白", "aliases": ["Hb"], "synonyms": [], "parent": None, "depends_on": []},
            {"category": "LabTest", "name": "变异系数", "aliases": ["RDW-CV"], "synonyms": [], "parent": "红细胞容积分布宽度", "depends_on": []},
            {"category": "LabTest", "name": "红细胞容积分布宽度", "aliases": ["RDW"], "synonyms": [], "parent": None, "depends_on": []},
            {"category": "Population", "name": "性别", "aliases": [], "synonyms": [], "parent": None, "depends_on": []},
            {"category": "Population", "name": "男性", "aliases": [], "synonyms": [], "parent": "性别", "depends_on": []},
            {"category": "Disease", "name": "贫血", "aliases": [], "synonyms": ["贫血症"], "parent": None, "depends_on": []},
        ]
        evidence = {"chunk_id": "c1", "chunk_sha256": "a" * 64, "source_quote": "血红蛋白 男性 130~175g/L"}
        ontology = {"approved": 0, "relations": [
            {"relation": "IS_A", "source": "变异系数", "target": "红细胞容积分布宽度", "evidence": {}},
            {"relation": "IS_A", "source": "男性", "target": "性别", "evidence": {}},
        ]}
        catalog = {"approved": 0, "entries": [
            {"page_id": "p1", "candidate_key": "血红蛋白", "entity_type": "TestItem", "text": "血红蛋白", "source": {"chunk_id": "c1", "chunk_sha256": "a" * 64, "exact_quote": "血红蛋白"}},
            {"page_id": "p1", "candidate_key": "贫血", "entity_type": "MedicalConcept", "text": "贫血", "source": {"chunk_id": "c1", "chunk_sha256": "a" * 64, "exact_quote": "贫血"}},
        ]}
        relations = {"approved": 0, "candidates": []}
        rules = {"approved": 0, "candidates": [{
            "candidate_id": "r1", "page_id": "p1", "rule_key": "血红蛋白降低见于贫血",
            "semantic_type": "SEEN_IN", "subject_logic": "SINGLE",
            "subject_candidate_keys": ["血红蛋白"], "conclusion_candidate_key": "贫血",
            "population_candidate_keys": [], "method_candidate_keys": [], "components": {},
            "source": {"chunk_id": "c1", "chunk_sha256": "a" * 64, "exact_quote": "血红蛋白降低见于贫血"},
        }]}
        references = {"approved": 0, "rules": [{
            "rule_id": "ref1", "function": {"output": "血红蛋白参考区间"},
            "applicability": {"specimen": "全血", "population": None, "method": None},
            "cases": [{"selector": {"性别": "男"}, "interval": {"lower": 130, "upper": 175, "unit": "g/L"}}],
            "evidence": evidence,
        }]}
        core = {"approved": 0, "execution_policy": {"missing_input": "INSUFFICIENT_DATA"}, "rules": [{
            "rule_id": "chapter01:formula:hgb-copy", "rule_type": "formula",
            "execution_readiness": "executable",
            "function": {
                "expression": "血红蛋白计算值 = 示例计算(血红蛋白)",
                "rule_name": "示例计算",
                "output": {"name": "血红蛋白计算值", "abbreviation": "HGB-C", "value_type": "number"},
                "inputs": [{"name": "血红蛋白", "abbreviation": "HGB", "value_type": "number"}],
            },
            "applicability": {"required_inputs": ["HGB"], "temporal": False},
            "cases": [], "formula_ast": {"input": "HGB"}, "source_ambiguity": None,
            "evidence": evidence,
        }, {
            "rule_id": "chapter01:classification:hgb-sex", "rule_type": "classification",
            "execution_readiness": "boundary_review_required",
            "function": {
                "expression": "贫血程度 = 示例分类(血红蛋白, 性别)",
                "rule_name": "示例分类",
                "output": {"name": "贫血程度", "value_type": "category"},
                "inputs": [
                    {"name": "血红蛋白", "abbreviation": "HGB", "value_type": "number"},
                    {"name": "性别", "value_type": "category"},
                ],
            },
            "applicability": {"required_inputs": ["HGB", "性别"], "temporal": False},
            "cases": [{"condition_ast": {"input": "HGB", "op": "LT", "value": 90}, "result": "贫血"}],
            "source_ambiguity": {"type": "boundary"}, "evidence": evidence,
        }]}
        temporal = {"approved": 0, "execution_policy": {"single_report": "NOT_APPLICABLE"}, "rules": [{
            "rule_id": "chapter01:temporal:hgb", "rule_type": "temporal",
            "execution_readiness": "trend_definition_review_required",
            "function": {
                "expression": "血红蛋白趋势 = 示例趋势(HGB时序)", "rule_name": "示例趋势",
                "output": {"name": "血红蛋白趋势", "value_type": "category"},
                "inputs": [{"name": "HGB时序", "value_type": "time_series"}],
            },
            "applicability": {"required_inputs": ["HGB时序"], "temporal": True,
                              "single_report_executable": False},
            "cases": [{"condition_ast": {"input": "HGB时序", "op": "TREND_EQ", "value": "持续下降"},
                       "result": "下降"}],
            "source_ambiguity": {"type": "trend_not_quantified"}, "evidence": evidence,
        }]}
        quality = {"approved": 0, "review_items": [{
            "rule_id": "chapter01:classification:hgb-sex", "reason": "边界待审核",
            "runtime_behavior": "AMBIGUOUS_MATCH",
        }]}
        manual = {"approved": 0, "executable_rules": [{
            "source_candidate_id": "indicator-rule:28fb53487c62c3932d5d7642",
            "rule_type": "threshold_decision",
        }], "temporal_rules": [], "semantic_facts": []}
        return entities, ontology, catalog, relations, rules, references, core, temporal, quality, manual

    def test_merge_triples_and_reference_ranges(self):
        graph = ChapterGraphBuilder().build(*self.packages())
        by_name = {(node["node_type"], node["name"]): node for node in graph["nodes"]}
        self.assertIn(("TestItem", "血红蛋白"), by_name)
        self.assertIn(("ReferenceRange", "血红蛋白参考范围/ref1/1"), by_name)
        predicates = [edge["predicate"] for edge in graph["edges"]]
        self.assertIn("IS_A", predicates)
        self.assertIn("ITEM_HAS_REFERENCE_RANGE", predicates)
        self.assertIn("RANGE_APPLIES_TO_POPULATION", predicates)
        self.assertIn("RULE_HAS_SUBJECT", predicates)
        self.assertIn("RULE_HAS_CONCLUSION", predicates)
        self.assertEqual(predicates.count("RULE_HAS_CONCLUSION"), 4)
        self.assertEqual(graph["approved"], 0)
        self.assertEqual(len({edge["triple_id"] for edge in graph["edges"]}), len(graph["edges"]))
        self.assertEqual(predicates.count("IS_A"), 2)
        book_rules = [node for node in graph["nodes"]
                      if "book-rule-library-v0.1" in node["origins"]]
        self.assertEqual(len([node for node in book_rules if node["node_type"] == "InterpretationRule"]), 3)
        formula_output = by_name[("TestItem", "血红蛋白计算值")]
        formula_rule = next(node for node in book_rules
                            if node["properties"].get("rule_id") == "chapter01:formula:hgb-copy")
        self.assertTrue(any(edge["subject_id"] == formula_rule["node_id"]
                            and edge["predicate"] == "RULE_HAS_CONCLUSION"
                            and edge["object_id"] == formula_output["node_id"]
                            for edge in graph["edges"]))
        self.assertEqual(graph["superseded_rules"][0]["new_rule_id"],
                         "chapter01:threshold:thrombocytopenia")
        self.assertTrue(all("review_id" in item for item in graph["review_items"]))

    def test_wbc_table_selectors_create_derived_test_items(self):
        packages = list(self.packages())
        references = dict(packages[5])
        references["rules"] = [*references["rules"], {
            "rule_id": "chapter01:reference:wbc-differential",
            "function": {"output": "白细胞分类计数参考范围"},
            "applicability": {"specimen": "全血", "population": None, "method": None},
            "cases": [
                {"selector": {"细胞类型": "中性粒细胞", "数值类型": "百分数"},
                 "interval": {"lower": 40, "upper": 75, "unit": "%"}},
                {"selector": {"细胞类型": "中性粒细胞", "数值类型": "绝对值"},
                 "interval": {"lower": 1.8, "upper": 6.3, "unit": "10^9/L"}},
            ],
            "evidence": {"chunk_id": "c1", "chunk_sha256": "a" * 64,
                         "source_quote": "中性粒细胞 百分数 绝对值"},
        }]
        packages[5] = references
        graph = ChapterGraphBuilder().build(*packages)
        nodes = {(node["node_type"], node["name"]): node for node in graph["nodes"]}
        self.assertIn(("TestItem", "中性粒细胞百分数"), nodes)
        self.assertIn(("TestItem", "中性粒细胞绝对值"), nodes)
        percentage = nodes[("TestItem", "中性粒细胞百分数")]
        self.assertEqual(percentage["properties"]["derivation_kind"], "table_row_column")
        self.assertTrue(any(
            edge["subject_id"] == percentage["node_id"]
            and edge["predicate"] == "ITEM_HAS_REFERENCE_RANGE"
            for edge in graph["edges"]
        ))

    def test_sqlite_has_no_dangling_edges(self):
        graph = ChapterGraphBuilder().build(*self.packages())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_graph_package(output, graph, {"fixture": "b" * 64},
                                {"book_rule_evidence_anchors_replayed": 3})
            with sqlite3.connect(output / "knowledge.sqlite") as db:
                self.assertEqual(db.execute("PRAGMA integrity_check").fetchone()[0], "ok")
                self.assertEqual(db.execute("PRAGMA foreign_key_check").fetchall(), [])
                self.assertEqual(db.execute("SELECT value FROM metadata WHERE key='approved'").fetchone()[0], "0")
            manifest = json.loads((output / "run-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["counts"]["edges"], len(graph["edges"]))
            superseded = json.loads((output / "superseded-rules.json").read_text(encoding="utf-8"))
            self.assertEqual(superseded["count"], 1)
            self.assertEqual(manifest["schema_version"], "chapter-graph-run/v0.2")
            self.assertEqual(manifest["validation"]["book_rule_evidence_anchors_replayed"], 3)

    def test_rule_evidence_replays_chunk_hash_and_quote(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunk_dir = root / "chunks" / "0001"
            chunk_dir.mkdir(parents=True)
            chunk = chunk_dir / "0000.md"
            chunk.write_text("血红蛋白 男性 130~175g/L", encoding="utf-8")
            import hashlib
            digest = hashlib.sha256(chunk.read_bytes()).hexdigest()
            manifest = {"chunks": [{"chunk_id": "c1", "chunk_path": "chunks/0001/0000.md",
                                     "chunk_sha256": digest}]}
            package = {"rules": [{"rule_id": "r1", "evidence": {
                "chunk_id": "c1", "chunk_sha256": digest,
                "source_quote": "血红蛋白 男性 130~175g/L",
            }}]}
            self.assertEqual(validate_rule_evidence([package], manifest, root), 1)
            package["rules"][0]["evidence"]["source_quote"] = "不存在"
            with self.assertRaisesRegex(ValueError, "quote cannot be replayed"):
                validate_rule_evidence([package], manifest, root)

    def test_entity_checkpoint_replays_exact_page_line(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            page = root / "0000.md"
            page.write_text("血红蛋白(Hb)参考区间。\n", encoding="utf-8")
            import hashlib
            digest = hashlib.sha256(page.read_bytes()).hexdigest()
            manifest = {"pages": [{"page_id": "p1", "chapter_page_index": 0,
                "printed_page_number": 4, "source_pdf_page_number": 21,
                "cleaned_path": "0000.md", "cleaned_sha256": digest}]}
            checkpoint = {"source_manifest_sha256": "bound", "page_results": {"page:0000": {
                "status": "success", "source_sha256": digest,
                "output": {"entities": [{"name": "血红蛋白", "aliases": ["Hb"],
                    "mentions": ["血红蛋白", "Hb"]}]}}}}
            evidence = build_entity_evidence(checkpoint, manifest, root)
            self.assertEqual(evidence["血红蛋白"][0]["exact_quote"], "血红蛋白(Hb)参考区间。")
            self.assertEqual(evidence["Hb"][0]["line_number"], 1)


if __name__ == "__main__":
    unittest.main()
