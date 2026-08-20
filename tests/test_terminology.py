from __future__ import annotations

from medical_kg_sourceprep.extraction.terminology import (
    annotate_with_dictionary,
    discover_candidates,
    load_reviewed_dictionary,
)


def test_discover_candidates_counts_frequency_and_page_frequency() -> None:
    pages = [
        (0, "p0", "血红蛋白降低，检查血红蛋白。"),
        (1, "p1", "血红蛋白正常。"),
    ]

    candidates = discover_candidates(
        pages,
        seed_terms={"血红蛋白"},
        min_frequency=2,
        max_ngram=1,
    )

    hemoglobin = next(item for item in candidates if item["term"] == "血红蛋白")
    assert hemoglobin["frequency"] == 3
    assert hemoglobin["document_frequency"] == 2
    assert hemoglobin["review_status"] == "PENDING"
    assert hemoglobin["candidate_source"] == "EXISTING_MENTION"


def test_seed_term_is_retained_below_frequency_threshold() -> None:
    candidates = discover_candidates(
        [(0, "p0", "仅见一次粒细胞缺乏症。")],
        seed_terms={"粒细胞缺乏症"},
        min_frequency=2,
        max_ngram=1,
    )

    assert [item["term"] for item in candidates] == ["粒细胞缺乏症"]


def test_dictionary_annotation_requires_approval_and_respects_latin_boundary() -> None:
    pages = [(0, "p0", "MCH 与 MCHC 均为指标，血红蛋白也可测量。")]
    entries = [
        {
            "term": "MCH",
            "entity_type": "LabIndicator",
            "review_status": "APPROVED",
            "match_policy": "TOKEN_BOUNDARY",
            "aliases": [],
        },
        {
            "term": "血红蛋白",
            "entity_type": "LabIndicator",
            "review_status": "PENDING",
            "match_policy": "EXACT",
            "aliases": [],
        },
    ]

    documents = annotate_with_dictionary(pages, entries)

    assert [(item["exact_quote"], item["start"], item["end"])
            for item in documents[0]["mentions"]] == [("MCH", 0, 3)]


def test_dictionary_annotation_keeps_nested_and_coordinated_entities() -> None:
    pages = [(0, "p0", "缺铁性贫血和白血病均需进一步检查。")]
    entries = [
        {"term": term, "entity_type": "Disease", "review_status": "APPROVED",
         "match_policy": "EXACT", "aliases": []}
        for term in ("贫血", "缺铁性贫血", "白血病")
    ]

    mentions = annotate_with_dictionary(pages, entries)[0]["mentions"]

    assert [(item["exact_quote"], item["start"], item["end"]) for item in mentions] == [
        ("缺铁性贫血", 0, 5),
        ("贫血", 3, 5),
        ("白血病", 6, 9),
    ]


def test_load_reviewed_dictionary_from_tsv(tmp_path) -> None:
    review_path = tmp_path / "review.tsv"
    review_path.write_text(
        "term\tentity_type\treview_status\tmatch_policy\taliases\n"
        "血红蛋白\tLabIndicator\tAPPROVED\tEXACT\tHb|HGB\n",
        encoding="utf-8",
    )

    entries = load_reviewed_dictionary(review_path)

    assert entries[0]["aliases"] == ["Hb", "HGB"]
