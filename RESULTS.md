# 实验结果记录

测试集：`data/processed/test.jsonl`，498 个句对 / 方向（seed=42 划分）。
评估脚本：`python -m ccnlp.eval_runner`。

## 对照组 A：标准 Seq2Seq 基线（无回环约束）

- 模型：`IDEA-CCNL/Randeng-BART-139M-SUMMARY`
- 数据：NiuTrans 5 万句对（去重后 49826），双向展开
- 训练：3 epoch，batch 16，lr 5e-5，单卡
- checkpoint：`/data2/lyl/outputs/randeng-bart-niutrans`

| 方向 | BLEU | ChrF | ExactMatch | BERTScore F1 |
|------|------|------|------------|--------------|
| 古→今 | 22.33 | 23.38 | 0.006 | 待补 |
| 今→古 | 28.75 | 26.85 | 0.020 | 待补 |
| 总体 | 25.06 | 25.11 | 0.013 | 待补 |

观察：
- 今→古方向显著优于古→今。
- 古→今存在幻觉（如「温故而知新」→ 凭空出现人名「张温」），疑似受史书类训练数据的人名模式影响——正是回环一致性约束意图缓解的信息失真。
- BERTScore 因服务器无法访问 HF Xet CDN 暂缺，待 `bert-base-chinese` 预下载后补齐。

## 实验组 B：回环一致性约束（待训练）

_待任务8 完成后填入。_
