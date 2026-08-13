# 第二、六、十三、十八章负例集

`negative-cases-v0.1.json` 收录四章中原文明示不支持的过度推断。每条记录绑定一个
canonical EvidenceChunk 和逐字引文，当前状态为 `HUMAN_REVIEW_REQUIRED`。

这些记录只用于评测抽取器或 Judge 是否错误生成强诊断、错误替代、错误因果或忽略联合
条件的结论，不进入候选图，也不作为正向知识发布。
