"""从 v0.6 原文 mention 生成展开、去重后的规范实体候选层。"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "evaluation/chapter-01/chapter-01-entity-mentions-v0.6.json"
MANUAL_GRAPH_PATH = ROOT / "evaluation/chapter-01/chapter-01-graph-test-set-v0.3.json"
OUTPUT_PATH = ROOT / "evaluation/chapter-01/chapter-01-canonical-entities-v0.8.json"

COORDINATOR_PATTERN = re.compile(r"\s*(?:、|或|和|与|及|以及)\s*")
PAREN_ALIAS_PATTERN = re.compile(
    r"^(?P<label>.+?)[（(](?P<alias>[A-Za-z][A-Za-z0-9-]{1,15})[）)]$"
)
HEADING_PATTERN = re.compile(
    r"^\s*(?:[（(][一二三四五六七八九十]+[）)]|[一二三四五六七八九十]+、|\d+[.、])\s*"
)
SPACE_PATTERN = re.compile(r"\s+")
LATEX_WRAPPER_PATTERN = re.compile(r"\\\(|\\\)|[{}]")

# 协调结构需要恢复省略的共同成分，不能只用连接词机械切分。这里覆盖 v0.6 中
# 全部 28 条真实协调 mention；标题“二、缺铁性贫血检验”不属于协调结构。
# 每个原子项显式声明类型，避免错误继承外层 mention 的类型。
COORDINATION_EXPANSIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "红细胞和血红蛋白减少": (("红细胞减少", "IndicatorState"), ("血红蛋白减少", "IndicatorState")),
    "红细胞膜的结构和红细胞内血红蛋白结构异常": (
        ("红细胞膜结构异常", "ClinicalContext"),
        ("红细胞膜", "ClinicalContext"),
        ("红细胞内血红蛋白结构异常", "ClinicalContext"),
        ("红细胞内血红蛋白", "LabIndicator"),
        ("血红蛋白", "LabIndicator"),
    ),
    "急性或慢性失血": (("急性失血", "ClinicalContext"), ("慢性失血", "ClinicalContext")),
    "红细胞和血红蛋白增多": (("红细胞增多", "IndicatorState"), ("血红蛋白增多", "IndicatorState")),
    "急、慢性感染": (("急性感染", "ClinicalContext"), ("慢性感染", "ClinicalContext")),
    "高温或严寒": (("高温", "ClinicalContext"), ("严寒", "ClinicalContext")),
    "急性感染或炎症": (("急性感染", "ClinicalContext"), ("炎症", "ClinicalContext")),
    "广泛的组织损伤或坏死": (("广泛的组织损伤", "ClinicalContext"), ("广泛的组织坏死", "ClinicalContext")),
    "急、慢性淋巴细胞性白血病": (("急性淋巴细胞性白血病", "Disease"), ("慢性淋巴细胞性白血病", "Disease"), ("淋巴细胞", "LabIndicator"), ("白血病", "Disease")),
    "药物和食物过敏": (("药物过敏", "Disease"), ("食物过敏", "Disease")),
    "止血和凝血功能": (("止血功能", "ClinicalContext"), ("凝血功能", "ClinicalContext")),
    "血小板破坏或消耗过多": (("血小板破坏过多", "ClinicalContext"), ("血小板消耗过多", "ClinicalContext")),
    "胃、十二指肠溃疡出血": (("胃溃疡出血", "Disease"), ("十二指肠溃疡出血", "Disease")),
    "妊娠中、后期": (("妊娠中期", "ClinicalContext"), ("妊娠后期", "ClinicalContext")),
    # 本句跨 chunk 承接“缺乏叶酸”，因此恢复的是同一个叶酸缺乏事件及两个妊娠阶段。
    "中、晚期缺乏": (("叶酸缺乏", "ClinicalContext"), ("妊娠中期", "ClinicalContext"), ("妊娠晚期", "ClinicalContext")),
    "准备怀孕的妇女和孕妇": (("准备怀孕的妇女", "ClinicalContext"), ("孕妇", "ClinicalContext")),
    "血液循环与微循环障碍": (("血液循环障碍", "ClinicalContext"), ("微循环障碍", "ClinicalContext")),
    "组织和脏器缺血": (("组织缺血", "ClinicalContext"), ("脏器缺血", "ClinicalContext")),
    "白细胞和血小板流变性": (("白细胞流变性", "LabIndicator"), ("血小板流变性", "LabIndicator")),
    "红细胞和血小板聚集性增强": (("红细胞聚集性增强", "IndicatorState"), ("红细胞聚集性", "LabIndicator"), ("血小板聚集性增强", "IndicatorState"), ("血小板聚集性", "LabIndicator")),
    "血浆纤维蛋白原和球蛋白含量增高": (
        ("血浆纤维蛋白原含量增高", "IndicatorState"),
        ("血浆纤维蛋白原含量", "LabIndicator"),
        ("球蛋白含量增高", "IndicatorState"),
        ("球蛋白含量", "LabIndicator"),
    ),
    "子宫内膜破损及出血": (("子宫内膜破损", "ClinicalContext"), ("子宫内膜出血", "ClinicalContext")),
    "组织损伤及坏死": (("组织损伤", "ClinicalContext"), ("组织坏死", "ClinicalContext")),
    "A、B凝集原": (("A凝集原", "LabIndicator"), ("B凝集原", "LabIndicator")),
    "抗A、抗B凝集素": (("抗A凝集素", "LabIndicator"), ("抗B凝集素", "LabIndicator")),
    "A、B、O、AB四型": (("A型血", "IndicatorState"), ("B型血", "IndicatorState"), ("O型血", "IndicatorState"), ("AB型血", "IndicatorState")),
    "先天性凝血因子I、II、V、VII、X缺乏": tuple(
        candidate
        for factor in ("I", "II", "V", "VII", "X")
        for candidate in (
            (f"先天性凝血因子{factor}缺乏", "ClinicalContext"),
            (f"凝血因子{factor}", "ClinicalContext"),
        )
    ),
}

# 非逐字但语义确定的规范化。外层事件与内层实体同时返回；例如
# “人体缺少叶酸”既产生“叶酸缺乏”，也产生其内层实体“叶酸”。
CONTEXT_NORMALIZATIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "人体缺少叶酸": (("叶酸缺乏", "ClinicalContext"), ("叶酸", "ClinicalContext")),
    "怀孕早期缺乏叶酸": (("叶酸缺乏", "ClinicalContext"), ("叶酸", "ClinicalContext"), ("怀孕早期", "ClinicalContext")),
    "补充叶酸": (("叶酸", "ClinicalContext"),),
    "维生素B_12缺乏": (("维生素B12缺乏", "ClinicalContext"), ("维生素B12", "ClinicalContext")),
    "维生素B12缺乏": (("维生素B12缺乏", "ClinicalContext"), ("维生素B12", "ClinicalContext")),
    "铁缺乏": (("铁缺乏", "ClinicalContext"), ("铁", "ClinicalContext")),
    "内因子缺乏": (("内因子缺乏", "ClinicalContext"), ("内因子", "ClinicalContext")),
    "维生素K缺乏": (("维生素K缺乏", "ClinicalContext"), ("维生素K", "ClinicalContext")),
    "获得性凝血因子缺乏": (("获得性凝血因子缺乏", "ClinicalContext"), ("凝血因子", "ClinicalContext")),
    "遗传性转铁蛋白缺乏症": (("遗传性转铁蛋白缺乏症", "Disease"), ("转铁蛋白", "LabIndicator")),
    "G6PD缺乏症": (("G6PD缺乏症", "Disease"), ("G6PD", "LabIndicator")),
    "粒细胞缺乏症": (("粒细胞缺乏症", "Disease"), ("粒细胞", "LabIndicator")),
    "粒细胞缺乏症恢复期": (("粒细胞缺乏症恢复期", "ClinicalContext"), ("粒细胞缺乏症", "Disease"), ("粒细胞", "LabIndicator")),
    "排除深静脉血栓(DVT)有重要价值": (("深静脉血栓", "Disease"),),
    "血小板破坏增多但骨髓代偿功能良好": (("血小板破坏增多", "ClinicalContext"), ("骨髓代偿功能良好", "ClinicalContext")),
    "脱氧核糖核酸(DNA)合成障碍": (("脱氧核糖核酸合成障碍", "ClinicalContext"), ("脱氧核糖核酸", "ClinicalContext")),
    "中性粒细胞比例比淋巴细胞高": (("中性粒细胞比例", "LabIndicator"), ("淋巴细胞比例", "LabIndicator"), ("中性粒细胞比例高于淋巴细胞比例", "IndicatorState")),
    "婴幼儿生长发育需铁量增加": (("婴幼儿生长发育", "ClinicalContext"), ("需铁量增加", "ClinicalContext")),
    "红细胞膜表面积/血细胞容积的比值": (("红细胞膜表面积/血细胞容积比值", "LabIndicator"), ("红细胞膜表面积", "LabIndicator"), ("血细胞容积", "LabIndicator")),
    "标准血清+受检者红细胞": (("标准血清+受检者红细胞", "LabIndicator"), ("标准血清", "ClinicalContext"), ("受检者红细胞", "ClinicalContext")),
    "标准红细胞+受检者血清": (("标准红细胞+受检者血清", "LabIndicator"), ("标准红细胞", "ClinicalContext"), ("受检者血清", "ClinicalContext")),
    "抗A+抗B(O型血清)": (("抗A+抗B", "LabIndicator"), ("抗A", "LabIndicator"), ("抗B", "LabIndicator"), ("O型血清", "ClinicalContext")),
    "口服抗凝药是否适量": (("口服抗凝药", "ClinicalContext"),),
    "血液循环中有异常抗凝血物质": (("异常抗凝血物质", "ClinicalContext"),),
    "比较少见的红细胞增多": (("红细胞增多", "IndicatorState"),),
    "某些严重的过敏性疾病": (("过敏性疾病", "Disease"),),
    "单个血小板的平均大小": (("平均血小板体积", "LabIndicator"),),
    "红细胞表面电荷密度降低": (("红细胞表面电荷密度降低", "IndicatorState"), ("红细胞表面电荷密度", "LabIndicator")),
    "血浆中的一些大分子蛋白质": (("血浆大分子蛋白质", "ClinicalContext"), ("大分子蛋白质", "ClinicalContext")),
    "长期使用肾上腺皮质激素": (("长期使用肾上腺皮质激素", "ClinicalContext"), ("肾上腺皮质激素", "ClinicalContext")),
    "组织内铁蛋白释放增加": (("组织内铁蛋白释放增加", "ClinicalContext"), ("铁蛋白", "LabIndicator")),
    "相叠成串似“缗钱状”": (("红细胞缗钱状聚集", "IndicatorState"), ("红细胞", "LabIndicator")),
    "某些有毒有害化学物质": (("有毒有害化学物质", "ClinicalContext"),),
    "制造红细胞的骨髓减少": (("红细胞生成", "ClinicalContext"), ("骨髓减少", "ClinicalContext"), ("红细胞", "LabIndicator"), ("骨髓", "ClinicalContext")),
    "骨髓制造红细胞增多": (("红细胞生成增多", "ClinicalContext"), ("红细胞生成", "ClinicalContext"), ("红细胞", "LabIndicator"), ("骨髓", "ClinicalContext")),
    "血浆纤维蛋白原增高": (("血浆纤维蛋白原增高", "IndicatorState"), ("血浆纤维蛋白原", "LabIndicator")),
    "肝脏铁蛋白合成减少": (("肝脏铁蛋白合成减少", "ClinicalContext"), ("铁蛋白", "LabIndicator")),
    "细胞形态呈巨型改变": (("粒细胞系细胞形态巨型改变", "IndicatorState"), ("粒细胞系", "ClinicalContext"), ("巨核细胞系细胞形态巨型改变", "IndicatorState"), ("巨核细胞系", "ClinicalContext")),
    "纤维蛋白原含量增加": (("纤维蛋白原含量增加", "IndicatorState"), ("纤维蛋白原", "LabIndicator")),
    "急性传染病的恢复期": (("急性传染病恢复期", "ClinicalContext"), ("急性传染病", "Disease")),
    "应用肾上腺皮质激素": (("应用肾上腺皮质激素", "ClinicalContext"), ("肾上腺皮质激素", "ClinicalContext")),
    "增长迅速的恶性肿瘤": (("恶性肿瘤", "Disease"), ("肿瘤生长迅速", "ClinicalContext")),
    "其他组织器官的疾病": (("组织器官疾病", "ClinicalContext"),),
    "促红细胞生成素增高": (("促红细胞生成素增高", "IndicatorState"), ("促红细胞生成素", "LabIndicator")),
    "中性粒细胞相对偏低": (("中性粒细胞相对偏低", "IndicatorState"), ("中性粒细胞比例", "LabIndicator")),
    "转铁蛋白释放增加": (("转铁蛋白释放增加", "ClinicalContext"), ("转铁蛋白", "LabIndicator")),
    "转铁蛋白合成增加": (("转铁蛋白合成增加", "ClinicalContext"), ("转铁蛋白", "LabIndicator")),
    "转铁蛋白合成减少": (("转铁蛋白合成减少", "ClinicalContext"), ("转铁蛋白", "LabIndicator")),
    "转铁蛋白合成不足": (("转铁蛋白合成不足", "ClinicalContext"), ("转铁蛋白", "LabIndicator")),
    "转铁蛋白丢失增加": (("转铁蛋白丢失增加", "ClinicalContext"), ("转铁蛋白", "LabIndicator")),
    "血红蛋白结构异常": (("血红蛋白结构异常", "ClinicalContext"), ("血红蛋白", "LabIndicator")),
    "血红蛋白合成正常": (("血红蛋白合成正常", "ClinicalContext"), ("血红蛋白", "LabIndicator")),
    "血红蛋白合成减少": (("血红蛋白合成减少", "ClinicalContext"), ("血红蛋白", "LabIndicator")),
    "急性期时相反应蛋白": (("急性期反应蛋白", "ClinicalContext"),),
    "器官移植后的排斥反应": (("器官移植排斥反应", "ClinicalContext"), ("器官移植", "ClinicalContext"), ("排斥反应", "ClinicalContext")),
    "单核巨噬细胞系统功能亢进": (("单核巨噬细胞系统功能亢进", "ClinicalContext"), ("单核巨噬细胞系统", "ClinicalContext")),
    "骨髓造血功能衰竭": (("骨髓造血功能衰竭", "ClinicalContext"), ("骨髓", "ClinicalContext"), ("造血功能衰竭", "ClinicalContext")),
    "骨髓造血功能受损": (("骨髓造血功能受损", "ClinicalContext"), ("骨髓", "ClinicalContext"), ("造血功能受损", "ClinicalContext")),
    "骨髓造血功能不良": (("骨髓造血功能不良", "ClinicalContext"), ("骨髓", "ClinicalContext"), ("造血功能不良", "ClinicalContext")),
    "中枢神经系统的损伤": (("中枢神经系统损伤", "ClinicalContext"), ("中枢神经系统", "ClinicalContext")),
    "平均红细胞血红蛋白浓度": (("平均红细胞血红蛋白浓度", "LabIndicator"), ("红细胞", "LabIndicator"), ("血红蛋白", "LabIndicator")),
    "平均红细胞血红蛋白含量": (("平均红细胞血红蛋白含量", "LabIndicator"), ("红细胞", "LabIndicator"), ("血红蛋白", "LabIndicator")),
    "INR最适值为2.0~3.0": (("INR最适值为2.0~3.0", "IndicatorState"), ("INR", "LabIndicator")),
    "原发性巨球蛋白血症": (("原发性巨球蛋白血症", "Disease"), ("巨球蛋白", "LabIndicator")),
    "原发性血小板增多症": (("原发性血小板增多症", "Disease"), ("血小板增多", "IndicatorState"), ("血小板", "LabIndicator")),
    "原发性血小板减少性紫癜": (("原发性血小板减少性紫癜", "Disease"), ("血小板减少", "IndicatorState"), ("血小板", "LabIndicator")),
    "血红蛋白(男性)": (("血红蛋白(男性)", "LabIndicator"), ("血红蛋白", "LabIndicator"), ("男性", "ClinicalContext")),
    "血红蛋白(女性)": (("血红蛋白(女性)", "LabIndicator"), ("血红蛋白", "LabIndicator"), ("女性", "ClinicalContext")),
    "骨髓代偿功能良好": (("骨髓代偿功能良好", "ClinicalContext"), ("骨髓", "ClinicalContext"), ("代偿功能良好", "ClinicalContext")),
    "造血功能逐渐减退": (("造血功能逐渐减退", "ClinicalContext"), ("造血功能", "ClinicalContext")),
    "造血原料相对不足": (("造血原料相对不足", "ClinicalContext"), ("造血原料", "ClinicalContext")),
    "进行体外循环手术": (("体外循环手术", "ClinicalContext"), ("体外循环", "ClinicalContext")),
    "红细胞容积分布宽度": (("红细胞容积分布宽度", "LabIndicator"), ("红细胞", "LabIndicator")),
    "骨髓增生异常综合征": (("骨髓增生异常综合征", "Disease"), ("骨髓", "ClinicalContext")),
    "革兰氏阴性杆菌感染": (("革兰氏阴性杆菌感染", "ClinicalContext"), ("革兰氏阴性杆菌", "ClinicalContext")),
    "发绀型先天性心脏病": (("发绀型先天性心脏病", "Disease"), ("先天性心脏病", "Disease"), ("发绀", "ClinicalContext")),
    "珠蛋白生成障碍性贫血": (("珠蛋白生成障碍性贫血", "Disease"), ("珠蛋白生成障碍", "ClinicalContext"), ("珠蛋白", "ClinicalContext")),
    "珠蛋白合成障碍性贫血": (("珠蛋白合成障碍性贫血", "Disease"), ("珠蛋白合成障碍", "ClinicalContext"), ("珠蛋白", "ClinicalContext")),
    "小红细胞低色素性贫血": (("小红细胞低色素性贫血", "Disease"),),
    "小细胞低色素性贫血": (("小细胞低色素性贫血", "Disease"),),
    "正细胞不均一性贫血": (("正细胞不均一性贫血", "Disease"),),
    "小细胞不均一性贫血": (("小细胞不均一性贫血", "Disease"),),
    "大细胞不均一性贫血": (("大细胞不均一性贫血", "Disease"),),
    "遗传性球形红细胞增多症": (("遗传性球形红细胞增多症", "Disease"), ("球形红细胞", "IndicatorState"), ("红细胞增多", "IndicatorState"), ("红细胞", "LabIndicator")),
    "遗传性椭圆形红细胞增多症": (("遗传性椭圆形红细胞增多症", "Disease"), ("椭圆形红细胞", "IndicatorState"), ("红细胞增多", "IndicatorState"), ("红细胞", "LabIndicator")),
    "红细胞电泳时间延长": (("红细胞电泳时间延长", "IndicatorState"), ("红细胞电泳时间", "LabIndicator"), ("红细胞", "LabIndicator")),
    "亚急性感染性心内膜炎": (("亚急性感染性心内膜炎", "Disease"), ("感染性心内膜炎", "Disease")),
    "后天性肺源性心脏病": (("后天性肺源性心脏病", "Disease"), ("肺源性心脏病", "Disease")),
    "嗜酸性粒细胞白血病": (("嗜酸性粒细胞白血病", "Disease"), ("嗜酸性粒细胞", "LabIndicator"), ("白血病", "Disease")),
    "嗜碱性粒细胞白血病": (("嗜碱性粒细胞白血病", "Disease"), ("嗜碱性粒细胞", "LabIndicator"), ("白血病", "Disease")),
    "急性淋巴细胞性白血病": (("急性淋巴细胞性白血病", "Disease"), ("淋巴细胞", "LabIndicator"), ("白血病", "Disease")),
    "慢性淋巴细胞性白血病": (("慢性淋巴细胞性白血病", "Disease"), ("淋巴细胞", "LabIndicator"), ("白血病", "Disease")),
    "血液中含氧量减少": (("血液含氧量减少", "IndicatorState"), ("血液含氧量", "LabIndicator")),
    "单纯小细胞性贫血": (("单纯小细胞性贫血", "Disease"),),
    "大细胞均一性贫血": (("大细胞均一性贫血", "Disease"),),
    "小细胞均一性贫血": (("小细胞均一性贫血", "Disease"),),
    "正细胞均一性贫血": (("正细胞均一性贫血", "Disease"),),
    "应用某些化学药物": (("化学药物暴露", "ClinicalContext"), ("化学药物", "ClinicalContext")),
    "急性感染的恢复期": (("急性感染恢复期", "ClinicalContext"), ("急性感染", "ClinicalContext")),
    "急性白血病化疗后": (("急性白血病化疗后", "ClinicalContext"), ("急性白血病", "Disease")),
    "白细胞一过性增高": (("白细胞一过性增高", "IndicatorState"), ("白细胞", "LabIndicator")),
    "红细胞内液的黏度": (("红细胞内液黏度", "LabIndicator"),),
    "红细胞流动性变差": (("红细胞流动性变差", "IndicatorState"), ("红细胞流动性", "LabIndicator")),
    "红细胞的几何形状": (("红细胞几何形状", "LabIndicator"),),
    "红细胞绝对值增多": (("红细胞绝对值增多", "IndicatorState"), ("红细胞计数", "LabIndicator")),
    "红细胞聚集性增加": (("红细胞聚集性增加", "IndicatorState"), ("红细胞聚集性", "LabIndicator")),
    "红细胞聚集性增强": (("红细胞聚集性增强", "IndicatorState"), ("红细胞聚集性", "LabIndicator")),
    "红细胞聚集性增高": (("红细胞聚集性增高", "IndicatorState"), ("红细胞聚集性", "LabIndicator")),
    "血小板聚集性增强": (("血小板聚集性增强", "IndicatorState"), ("血小板聚集性", "LabIndicator")),
    "一次血黏度升高": (("血黏度升高", "IndicatorState"), ("血黏度", "LabIndicator")),
    "国际灵敏性指数": (("国际灵敏性指数", "LabIndicator"),),
    "制造红细胞减少": (("红细胞生成减少", "ClinicalContext"), ("红细胞生成", "ClinicalContext"), ("红细胞", "LabIndicator")),
    "异常增生性增多": (("中性粒细胞异常增生性增多", "IndicatorState"), ("中性粒细胞", "LabIndicator")),
    "形态学分类诊断": (("贫血形态学分类", "ClinicalContext"),),
    "红细胞刚性增高": (("红细胞刚性增高", "IndicatorState"), ("红细胞刚性", "LabIndicator")),
    "红细胞膜的弹性": (("红细胞膜弹性", "LabIndicator"),),
    "血中球蛋白增加": (("血中球蛋白增加", "IndicatorState"), ("球蛋白", "LabIndicator")),
    "血浆球蛋白增高": (("血浆球蛋白增高", "IndicatorState"), ("球蛋白", "LabIndicator")),
    "血液流动性下降": (("血液流动性下降", "IndicatorState"), ("血液流动性", "LabIndicator")),
    "血细胞浓度增加": (("血细胞浓度增加", "IndicatorState"), ("血细胞浓度", "LabIndicator")),
    "通过血黏度参数": (("血黏度参数", "LabIndicator"), ("血黏度", "LabIndicator")),
    "铁蛋白合成增加": (("铁蛋白合成增加", "ClinicalContext"), ("铁蛋白", "LabIndicator")),
    "血浆的黏度增加": (("血浆黏度增加", "IndicatorState"), ("血浆黏度", "LabIndicator")),
}

VAGUE_BOUNDARY_PATTERN = re.compile(r"一些|某些|各种|其他组织器官|有毒有害")

# 已核实为文档结构或表格角色的 mention。保留排除记录，但不生成可入图实体。
EXCLUDED_MENTIONS: dict[tuple[str, str], str] = {
    ("LabPanel", "临床血液检验"): "CHAPTER_HEADING_IS_SECTION_PATH",
    ("LabIndicator", "父母血型"): "RULE_TABLE_HEADER_NOT_INDICATOR",
    ("LabIndicator", "子女可能的血型"): "RULE_TABLE_HEADER_NOT_INDICATOR",
    ("LabIndicator", "子女不可能的血型"): "RULE_TABLE_HEADER_NOT_INDICATOR",
}

# 只合并本章原文明确定义的缩写、全称和同一方向状态表达。这里不使用模糊
# 相似度，避免把 RDW-CV/RDW-SD、不同标本或不同程度的状态误并。
CANONICAL_SYNONYM_MERGES: dict[tuple[str, str], str] = {
    ("LabPanel", "血液流变学"): "血液流变学检查",
    ("LabIndicator", "ESR"): "红细胞沉降率",
    ("LabIndicator", "INR"): "国际标准化比值",
    ("LabIndicator", "PCT"): "血小板压积",
    ("LabIndicator", "PDW"): "血小板体积分布宽度",
    ("LabIndicator", "PTR"): "凝血酶原时间比值",
    ("LabIndicator", "SF"): "血清铁蛋白",
    ("LabIndicator", "SI"): "血清铁",
    ("LabIndicator", "TIBC"): "总铁结合力",
    ("LabIndicator", "血沉"): "红细胞沉降率",
    ("LabIndicator", "血小板计数"): "血小板数量",
    ("LabIndicator", "血浆的黏度"): "血浆黏度",
    ("LabIndicator", "红细胞的变形性"): "红细胞变形性",
    ("LabIndicator", "红细胞的浓度"): "红细胞浓度",
    ("LabIndicator", "血红蛋白(男性)"): "血红蛋白",
    ("LabIndicator", "血红蛋白(女性)"): "血红蛋白",
    ("IndicatorState", "AB型"): "AB型血",
    ("IndicatorState", "D-二聚体为阳性"): "D-二聚体阳性",
    ("IndicatorState", "全血黏度增高"): "全血黏度升高",
    ("IndicatorState", "变形能力降低"): "红细胞变形能力降低",
    ("IndicatorState", "红细胞刚性增加"): "红细胞刚性增高",
    ("IndicatorState", "红细胞变形性愈差"): "红细胞变形能力降低",
    ("IndicatorState", "红细胞变形能力下降"): "红细胞变形能力降低",
    ("IndicatorState", "红细胞变形能力差"): "红细胞变形能力降低",
    ("IndicatorState", "红细胞聚集性增加"): "红细胞聚集性增高",
    ("IndicatorState", "红细胞聚集性增强"): "红细胞聚集性增高",
    ("IndicatorState", "血浆黏度增加"): "血浆黏度升高",
    ("IndicatorState", "血浆黏度增高"): "血浆黏度升高",
    ("IndicatorState", "血清铁蛋白水平升高"): "血清铁蛋白升高",
    ("IndicatorState", "血清铁蛋白水平降低"): "血清铁蛋白降低",
    ("IndicatorState", "白细胞减少"): "白细胞计数减少",
    ("IndicatorState", "血黏度愈高"): "血黏度升高",
    ("IndicatorState", "转铁蛋白升高"): "血清转铁蛋白升高",
    ("IndicatorState", "转铁蛋白降低"): "血清转铁蛋白降低",
}

# These states are computable from a single report value and the chapter table's
# reference/decision intervals. They are linked to the table's MCH/MCHC source
# mentions, rather than treated as direct literal mentions.
COMPUTED_INDICATOR_STATES: dict[str, tuple[str, ...]] = {
    "MCH": ("MCH增大", "MCH正常", "MCH减小", "MCH显著减小(<23pg)"),
    "MCHC": ("MCHC正常", "MCHC减小"),
}
COMPUTED_INDICATOR_STATE_CHUNK = "clinical-hematology:chapter-01:0003:0000"


def _stable_id(*values: str) -> str:
    raw = "\0".join(values).encode("utf-8")
    return "entity:" + hashlib.sha256(raw).hexdigest()[:20]


def _display_name(value: str) -> str:
    """清除排版噪声，但不做医学同义词推断。"""
    value = HEADING_PATTERN.sub("", value.strip())
    value = LATEX_WRAPPER_PATTERN.sub("", value)
    value = re.sub(r"([A-Za-z])_([0-9]+)", r"\1\2", value)
    value = value.replace("，", ",").replace("：", ":")
    return SPACE_PATTERN.sub("", value)


def _identity(value: str) -> str:
    return _display_name(value).casefold()


def _indicator_base(value: str) -> str | None:
    """从指标状态中恢复指标名；最终还必须命中冻结的 LabIndicator 词表。"""
    normalized = _display_name(value)
    match = re.fullmatch(
        r"(.+?)(?:水平)?(?:相对)?(?:为阳性|减少|增多|增加|增高|降低|升高|下降|减低|偏低|减小|正常|异常|加快|延长|阳性)",
        normalized,
    )
    if match is not None:
        return match.group(1)
    match = re.fullmatch(r"(.+?)[<>≤≥].+", normalized)
    if match is not None:
        return match.group(1)
    match = re.fullmatch(r"(.+?)在正常范围内", normalized)
    return match.group(1) if match is not None else None


def _parenthetical_alias(value: str) -> tuple[str, list[str]]:
    normalized = _display_name(value)
    match = PAREN_ALIAS_PATTERN.fullmatch(normalized)
    if match is None:
        return normalized, []
    return match.group("label"), [match.group("alias")]


def _candidate_names(mention: dict[str, Any]) -> tuple[list[dict[str, str]], bool]:
    quote = str(mention["exact_quote"])
    base_name, aliases = _parenthetical_alias(quote)
    normalized = _display_name(base_name)
    context = CONTEXT_NORMALIZATIONS.get(normalized)
    if context:
        return ([{
            "canonical_name": name,
            "entity_type": entity_type,
            "derivation": (
                "PARENTHETICAL_ALIAS"
                if aliases and _identity(name) == _identity(base_name)
                else "CONTEXT_NORMALIZATION"
            ),
        } for name, entity_type in context], True)
    expansion = COORDINATION_EXPANSIONS.get(normalized)
    if expansion:
        return ([{
            "canonical_name": name,
            "entity_type": entity_type,
            "derivation": "COORDINATION_EXPANSION",
        } for name, entity_type in expansion], True)
    return ([{
        "canonical_name": base_name,
        "entity_type": str(mention["entity_type"]),
        "derivation": "PARENTHETICAL_ALIAS" if aliases else "DIRECT_MENTION",
    }], False)


def build_dataset(
    source: dict[str, Any], manual_graph: dict[str, Any] | None = None
) -> dict[str, Any]:
    entity_groups: dict[tuple[str, str], dict[str, Any]] = {}
    links: list[dict[str, Any]] = []
    review_items: list[dict[str, Any]] = []
    excluded_mentions: list[dict[str, str]] = []
    all_mentions: list[dict[str, Any]] = []
    mention_count = 0
    expanded_mention_count = 0
    known_indicators: dict[str, str] = {}
    for case in source["cases"]:
        for mention in case["mentions"]:
            if mention["entity_type"] != "LabIndicator":
                continue
            name, _ = _parenthetical_alias(str(mention["exact_quote"]))
            known_indicators.setdefault(_identity(name), name)
    for candidates in (*COORDINATION_EXPANSIONS.values(), *CONTEXT_NORMALIZATIONS.values()):
        for name, entity_type in candidates:
            if entity_type == "LabIndicator":
                known_indicators.setdefault(_identity(name), name)

    for case in source["cases"]:
        for mention in case["mentions"]:
            all_mentions.append(mention)
            mention_count += 1
            exclusion_reason = EXCLUDED_MENTIONS.get((
                str(mention["entity_type"]),
                _display_name(str(mention["exact_quote"])),
            ))
            if exclusion_reason is not None:
                excluded_mentions.append({
                    "mention_id": str(mention["mention_id"]),
                    "exact_quote": str(mention["exact_quote"]),
                    "reason_code": exclusion_reason,
                })
                continue
            candidates, expanded = _candidate_names(mention)
            # 指标状态同时保留内层指标，例如“血清铁蛋白降低”与“血清铁蛋白”。
            state_bases: list[dict[str, str]] = []
            for candidate in candidates:
                if candidate["entity_type"] != "IndicatorState":
                    continue
                base = _indicator_base(candidate["canonical_name"])
                if base is None or _identity(base) not in known_indicators:
                    continue
                state_bases.append({
                    "canonical_name": known_indicators[_identity(base)],
                    "entity_type": "LabIndicator",
                    "derivation": "NESTED_INDICATOR_BASE",
                })
            candidates.extend(state_bases)
            expanded_mention_count += int(expanded)
            quote = str(mention["exact_quote"])
            has_coordinator = COORDINATOR_PATTERN.search(_display_name(quote)) is not None
            if has_coordinator and not expanded:
                review_items.append({
                    "mention_id": mention["mention_id"],
                    "exact_quote": quote,
                    "reason_code": "AMBIGUOUS_COORDINATION_NOT_EXPANDED",
                })
            if (
                VAGUE_BOUNDARY_PATTERN.search(_display_name(quote)) is not None
                and _display_name(quote) not in CONTEXT_NORMALIZATIONS
            ):
                review_items.append({
                    "mention_id": mention["mention_id"],
                    "exact_quote": quote,
                    "reason_code": "VAGUE_CANONICAL_BOUNDARY_REQUIRES_REVIEW",
                })

            parenthetical_name, parenthetical_aliases = _parenthetical_alias(quote)
            for candidate in candidates:
                canonical_name = candidate["canonical_name"]
                entity_type = candidate["entity_type"]
                identity = (entity_type, _identity(canonical_name))
                group = entity_groups.setdefault(identity, {
                    "canonical_id": _stable_id(*identity),
                    "canonical_name": canonical_name,
                    "entity_type": entity_type,
                    "aliases": set(),
                    "mention_ids": set(),
                    "derivations": set(),
                })
                group["mention_ids"].add(mention["mention_id"])
                group["derivations"].add(candidate["derivation"])
                # 只有“全称(缩写)”属于 alias。协调展开和嵌套派生的原文是来源，
                # 不是每个原子实体的同义词，否则会把“叶酸缺乏”错误合并进“叶酸”。
                if (
                    candidate["derivation"] == "PARENTHETICAL_ALIAS"
                    and _identity(quote) != identity[1]
                ):
                    group["aliases"].add(_display_name(quote))
                if (
                    candidate["derivation"] == "PARENTHETICAL_ALIAS"
                    and canonical_name == parenthetical_name
                ):
                    group["aliases"].update(parenthetical_aliases)
                links.append({
                    "mention_id": mention["mention_id"],
                    "canonical_id": group["canonical_id"],
                    "derivation": candidate["derivation"],
                })

            normalized_quote = _display_name(quote)
            computed_states = (
                COMPUTED_INDICATOR_STATES.get(normalized_quote, ())
                if mention["chunk_id"] == COMPUTED_INDICATOR_STATE_CHUNK
                else ()
            )
            for state_name in computed_states:
                identity = ("IndicatorState", _identity(state_name))
                group = entity_groups.setdefault(identity, {
                    "canonical_id": _stable_id(*identity),
                    "canonical_name": state_name,
                    "entity_type": "IndicatorState",
                    "aliases": set(),
                    "mention_ids": set(),
                    "derivations": set(),
                })
                group["mention_ids"].add(mention["mention_id"])
                group["derivations"].add("TABLE_THRESHOLD_DERIVATION")
                links.append({
                    "mention_id": mention["mention_id"],
                    "canonical_id": group["canonical_id"],
                    "derivation": "TABLE_THRESHOLD_DERIVATION",
                })

    # 将同一原文位置已经标出的内层 mention 连接到外层 mention。这样既保留完整语境，
    # 又能把叶实体作为关系端点使用；这里只使用原文坐标，不猜测医学同义词。
    links_by_mention: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for link in links:
        links_by_mention[link["mention_id"]].append(link)
    mentions_by_chunk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for mention in all_mentions:
        mentions_by_chunk[str(mention["chunk_id"])].append(mention)
    for chunk_mentions in mentions_by_chunk.values():
        for outer in chunk_mentions:
            for inner in chunk_mentions:
                if outer["mention_id"] == inner["mention_id"]:
                    continue
                if not (
                    int(outer["start"]) <= int(inner["start"])
                    and int(inner["end"]) <= int(outer["end"])
                    and (
                        int(outer["start"]) < int(inner["start"])
                        or int(inner["end"]) < int(outer["end"])
                    )
                ):
                    continue
                for inner_link in links_by_mention[inner["mention_id"]]:
                    if inner_link["derivation"] == "TABLE_THRESHOLD_DERIVATION":
                        continue
                    target = next(
                        group for group in entity_groups.values()
                        if group["canonical_id"] == inner_link["canonical_id"]
                    )
                    target["mention_ids"].add(outer["mention_id"])
                    links.append({
                        "mention_id": outer["mention_id"],
                        "canonical_id": inner_link["canonical_id"],
                        "derivation": "NESTED_SOURCE_MENTION",
                    })

    # 复合疾病可连接到同一冻结集合中已经存在的词尾疾病类型。要求前置修饰语至少
    # 两个字符，避免把“副伤寒”错误解释成“伤寒”的嵌套；过泛类别也不生成端点。
    excluded_disease_bases = {"疾病", "增多症"}
    disease_groups = [
        group for group in entity_groups.values()
        if group["entity_type"] == "Disease"
    ]
    group_by_id = {
        group["canonical_id"]: group for group in entity_groups.values()
    }
    for outer_link in list(links):
        outer_group = group_by_id[outer_link["canonical_id"]]
        if outer_group["entity_type"] != "Disease":
            continue
        outer_name = outer_group["canonical_name"]
        for inner_group in disease_groups:
            inner_name = inner_group["canonical_name"]
            if inner_name in excluded_disease_bases or inner_name == outer_name:
                continue
            if not outer_name.endswith(inner_name):
                continue
            if len(outer_name[:-len(inner_name)]) < 2:
                continue
            inner_group["mention_ids"].add(outer_link["mention_id"])
            inner_group["derivations"].add("NESTED_DISEASE_BASE")
            links.append({
                "mention_id": outer_link["mention_id"],
                "canonical_id": inner_group["canonical_id"],
                "derivation": "NESTED_DISEASE_BASE",
            })

    # 用“中文全称(缩写)”建立的别名索引，合并同类型的独立缩写 mention。
    alias_index: dict[tuple[str, str], str] = {}
    for group in entity_groups.values():
        for alias in group["aliases"]:
            alias_index[(group["entity_type"], _identity(alias))] = group["canonical_id"]

    by_id = {group["canonical_id"]: group for group in entity_groups.values()}
    redirects: dict[str, str] = {}
    for identity, group in list(entity_groups.items()):
        target_id = alias_index.get(identity)
        if target_id is None or target_id == group["canonical_id"]:
            continue
        target = by_id[target_id]
        target["aliases"].add(group["canonical_name"])
        target["aliases"].update(group["aliases"])
        target["mention_ids"].update(group["mention_ids"])
        target["derivations"].update(group["derivations"])
        target["derivations"].add("ALIAS_MERGE")
        redirects[group["canonical_id"]] = target_id
        del entity_groups[identity]

    synonym_merge_count = 0
    for (entity_type, source_name), target_name in CANONICAL_SYNONYM_MERGES.items():
        source_identity = (entity_type, _identity(source_name))
        target_identity = (entity_type, _identity(target_name))
        source_group = entity_groups.get(source_identity)
        target_group = entity_groups.get(target_identity)
        if source_group is None or target_group is None or source_group is target_group:
            continue
        target_group["aliases"].add(source_group["canonical_name"])
        target_group["aliases"].update(source_group["aliases"])
        target_group["mention_ids"].update(source_group["mention_ids"])
        target_group["derivations"].update(source_group["derivations"])
        target_group["derivations"].add("CANONICAL_SYNONYM_MERGE")
        redirects[source_group["canonical_id"]] = target_group["canonical_id"]
        del entity_groups[source_identity]
        synonym_merge_count += 1

    for link in links:
        link["canonical_id"] = redirects.get(link["canonical_id"], link["canonical_id"])
    unique_links = {
        (item["mention_id"], item["canonical_id"], item["derivation"]): item
        for item in links
    }

    # v0.6 is a literal-span evaluation layer, not the semantic authority for the
    # graph. Preserve manually annotated derived/table entities even when their
    # canonical names do not occur verbatim in the source text.
    manual_entity_count = 0
    if manual_graph is not None:
        for case in manual_graph["cases"]:
            for entity_type, canonical_name in case["entities"]:
                identity = (str(entity_type), _identity(str(canonical_name)))
                group = entity_groups.get(identity)
                if group is None:
                    group = next((
                        candidate for candidate in entity_groups.values()
                        if candidate["entity_type"] == entity_type
                        and any(
                            _identity(alias) == identity[1]
                            for alias in candidate["aliases"]
                        )
                    ), None)
                if group is None:
                    group = {
                        "canonical_id": _stable_id(*identity),
                        "canonical_name": str(canonical_name),
                        "entity_type": str(entity_type),
                        "aliases": set(),
                        "mention_ids": set(),
                        "derivations": set(),
                    }
                    entity_groups[identity] = group
                if "MANUAL_GRAPH_GOLD" not in group["derivations"]:
                    manual_entity_count += 1
                group["derivations"].add("MANUAL_GRAPH_GOLD")

    entities = [{
        "canonical_id": group["canonical_id"],
        "canonical_name": group["canonical_name"],
        "entity_type": group["entity_type"],
        "aliases": sorted(group["aliases"], key=_identity),
        "mention_ids": sorted(group["mention_ids"]),
        "derivations": sorted(group["derivations"]),
        "review_status": (
            "MANUAL_GRAPH_GOLD"
            if "MANUAL_GRAPH_GOLD" in group["derivations"]
            else "ASSISTANT_EXPANDED_REQUIRES_USER_VALIDATION"
        ),
    } for group in entity_groups.values()]
    entities.sort(key=lambda item: (
        str(item["entity_type"]), _identity(str(item["canonical_name"]))
    ))
    links = sorted(unique_links.values(), key=lambda item: (
        item["mention_id"], item["canonical_id"], item["derivation"]
    ))
    type_counts = defaultdict(int)
    for entity in entities:
        type_counts[entity["entity_type"]] += 1
    derivation_counts = defaultdict(int)
    for link in links:
        derivation_counts[link["derivation"]] += 1
    derivation_mention_counts = {
        derivation: len({
            link["mention_id"] for link in links
            if link["derivation"] == derivation
        })
        for derivation in derivation_counts
    }
    review_reason_counts = defaultdict(int)
    for item in review_items:
        review_reason_counts[item["reason_code"]] += 1

    return {
        "schema_version": "medical-kg-canonical-entity-candidates/v0.8",
        "status": "MANUAL_GRAPH_GOLD_WITH_MENTION_DERIVATIONS",
        "source_mentions": str(INPUT_PATH.relative_to(ROOT)),
        "source_manual_graph": (
            str(MANUAL_GRAPH_PATH.relative_to(ROOT)) if manual_graph is not None else None
        ),
        "contract": {
            "source_mentions_unchanged": True,
            "deduplication_identity": "ENTITY_TYPE_PLUS_NORMALIZED_CANONICAL_NAME",
            "external_medical_knowledge_used": False,
            "all_v06_coordination_mentions_expanded": True,
            "nested_policy": "SOURCE_SPAN_PLUS_TYPED_NORMALIZATION",
            "canonical_synonym_policy": "EXPLICIT_CHAPTER_LOCAL_EQUIVALENCE_ONLY",
            "manual_graph_entities_preserved_without_literal_mentions": True,
        },
        "canonical_entities": entities,
        "mention_to_canonical_links": links,
        "excluded_mentions": sorted(excluded_mentions, key=lambda item: item["mention_id"]),
        "review_items": sorted(review_items, key=lambda item: item["mention_id"]),
        "statistics": {
            "mention_count": mention_count,
            "expanded_mention_count": expanded_mention_count,
            "canonical_entity_count": len(entities),
            "manual_graph_entity_count": manual_entity_count,
            "mention_to_canonical_link_count": len(links),
            "excluded_mention_count": len(excluded_mentions),
            "alias_merge_count": len(redirects),
            "canonical_synonym_merge_count": synonym_merge_count,
            "coordination_mention_count": derivation_mention_counts.get("COORDINATION_EXPANSION", 0),
            "context_normalized_mention_count": derivation_mention_counts.get("CONTEXT_NORMALIZATION", 0),
            "nested_indicator_mention_count": derivation_mention_counts.get("NESTED_INDICATOR_BASE", 0),
            "nested_source_link_count": derivation_counts.get("NESTED_SOURCE_MENTION", 0),
            "review_item_count": len(review_items),
            "ambiguous_coordination_count": review_reason_counts.get(
                "AMBIGUOUS_COORDINATION_NOT_EXPANDED", 0
            ),
            "vague_boundary_review_count": review_reason_counts.get(
                "VAGUE_CANONICAL_BOUNDARY_REQUIRES_REVIEW", 0
            ),
            "canonical_entity_type_counts": dict(sorted(type_counts.items())),
            "link_derivation_counts": dict(sorted(derivation_counts.items())),
        },
    }


def main() -> None:
    source = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    manual_graph = json.loads(MANUAL_GRAPH_PATH.read_text(encoding="utf-8"))
    payload = build_dataset(source, manual_graph)
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["statistics"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
