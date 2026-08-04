"""Small controlled vocabulary for normalizing common laboratory test names."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LaboratoryTerm:
    standard_name: str
    abbreviation: str
    aliases: tuple[str, ...]


_TERMS = (
    LaboratoryTerm("丙氨酸氨基转移酶", "ALT", ("丙氨酸氨基转移酶", "谷丙转氨酶")),
    LaboratoryTerm(
        "天冬氨酸氨基转移酶",
        "AST",
        ("天冬氨酸氨基转移酶", "天门冬氨酸氨基转移酶", "谷草转氨酶"),
    ),
    LaboratoryTerm("总胆红素", "TBIL", ("总胆红素",)),
    LaboratoryTerm("直接胆红素", "DBIL", ("直接胆红素",)),
    LaboratoryTerm("间接胆红素", "IBIL", ("间接胆红素",)),
    LaboratoryTerm("总蛋白", "TP", ("总蛋白", "总蛋白定量", "总蛋白量")),
    LaboratoryTerm("白蛋白", "ALB", ("白蛋白", "白蛋白定量")),
    LaboratoryTerm(
        "白蛋白/球蛋白比值",
        "A/G",
        ("A/G", "白球比", "白蛋白/球蛋白", "白蛋白/球蛋白比值"),
    ),
    LaboratoryTerm("球蛋白", "GLO", ("球蛋白",)),
    LaboratoryTerm("前白蛋白", "PA", ("前白蛋白",)),
    LaboratoryTerm("γ-谷氨酰转移酶", "GGT", ("γ-谷氨酰转移酶", "γ-谷氨酰转肽酶")),
    LaboratoryTerm("碱性磷酸酶", "ALP", ("碱性磷酸酶",)),
    LaboratoryTerm("总胆汁酸", "TBA", ("总胆汁酸", "血清总胆汁酸")),
    LaboratoryTerm("乳酸脱氢酶", "LDH", ("乳酸脱氢酶",)),
    LaboratoryTerm("α-羟丁酸脱氢酶", "HBDH", ("α-羟丁酸脱氢酶",)),
    LaboratoryTerm("肌酸激酶", "CK", ("肌酸激酶",)),
    LaboratoryTerm(
        "肌酸激酶-MB同工酶",
        "CK-MB",
        ("肌酸激酶-MB同工酶", "肌酸激酶-MB同工酶活性", "肌酸激酶同工酶"),
    ),
    LaboratoryTerm("乳酸", "LA", ("乳酸",)),
    LaboratoryTerm("C反应蛋白", "CRP", ("C反应蛋白",)),
    LaboratoryTerm("胱抑素C", "Cys C", ("胱抑素C", "血清胱抑素C测定")),
    LaboratoryTerm("尿素", "Urea", ("尿素",)),
    LaboratoryTerm("肌酐", "CR", ("肌酐", "肌酐（酶法）", "肌酐(酶法)")),
    LaboratoryTerm("估算肾小球滤过率", "eGFR", ("估算肾小球滤过率",)),
    LaboratoryTerm("尿酸", "UA", ("尿酸",)),
    LaboratoryTerm("葡萄糖", "GLU", ("葡萄糖",)),
    LaboratoryTerm("二氧化碳", "CO2", ("二氧化碳",)),
    LaboratoryTerm("钾", "K", ("钾",)),
    LaboratoryTerm("钠", "Na", ("钠",)),
    LaboratoryTerm("氯", "CL", ("氯",)),
    LaboratoryTerm("钙", "Ca", ("钙", "总钙")),
    LaboratoryTerm("校正钙", "cCa", ("校正钙",)),
    LaboratoryTerm("无机磷", "IP", ("无机磷",)),
    LaboratoryTerm("β2微球蛋白", "β2-MG", ("β2微球蛋白", "β2-微球蛋白", "血β2微球蛋白")),
    LaboratoryTerm("C1q循环复合物", "C1q", ("C1q循环复合物",)),
    LaboratoryTerm("甘油三酯", "TG", ("甘油三酯",)),
    LaboratoryTerm("总胆固醇", "TC", ("总胆固醇",)),
    LaboratoryTerm("高密度脂蛋白胆固醇", "HDL-C", ("高密度脂蛋白", "高密度脂蛋白胆固醇",)),
    LaboratoryTerm("低密度脂蛋白胆固醇", "LDL-C", ("低密度脂蛋白", "低密度脂蛋白胆固醇",)),
)

_BY_ALIAS = {alias: term for term in _TERMS for alias in term.aliases}


def canonicalize_laboratory_term(raw_name: str, abbreviation: str | None) -> tuple[str, str | None]:
    """Return a canonical name/code while keeping unknown terms unchanged."""
    candidates = [raw_name]
    for prefix in ("血清", "血浆", "全血"):
        if raw_name.startswith(prefix) and len(raw_name) > len(prefix):
            candidates.append(raw_name[len(prefix):])
    term = next((_BY_ALIAS[name] for name in candidates if name in _BY_ALIAS), None)
    if term is None:
        return raw_name, abbreviation
    return term.standard_name, term.abbreviation
