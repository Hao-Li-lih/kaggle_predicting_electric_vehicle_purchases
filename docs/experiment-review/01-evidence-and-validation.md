# 01｜证据、评价指标与比较边界

[返回目录](README.md) · [下一篇：特征工程](02-feature-engineering.md)

## 1. 为什么使用 ROC-AUC

目标 `Will_Buy_EV` 是 `Yes`/`No`；模型输出 `Yes` 的概率。ROC-AUC 可以理解为：随机取一个真正会购买的人和一个不会购买的人，模型把前者排在后者之前的概率。`0.5` 近似随机排序，越接近 `1` 越好。它不要求选定 `0.5` 这样的分类阈值，也不直接奖励“概率数值恰好等于真实发生率”。因此概率平均和百分位排名平均都值得检验，但最终仍须计算 AUC，而不能仅凭直觉判断。[Notebook 02](../../notebooks/02_feature_engineering_model_comparison.ipynb)；[Notebook 03](../../notebooks/03_advanced_models_and_deep_learning.ipynb)

训练集共有 `668,665` 行、测试集 `286,571` 行；`Yes` 比例约 `17.46%`。`id` 仅是提交键。数据探索发现原始 CSV 无缺失与整行重复，训练/测试字段一致；预处理仍保留缺失值填补，以便流程稳健。[Notebook 01](../../notebooks/01_data_exploration.ipynb)

## 2. 三种不能混用的分数

| 名称 | 来源 | 能说明什么 | 不能说明什么 |
|---|---|---|---|
| Tuning AUC | 训练池里划出的调优集 | 候选参数、早停轮数、融合权重的选择 | 最终泛化表现；它已被用于选择 |
| Holdout AUC | 留出的有标签数据 | 在当前切分上的事后比较 | 反复查看后仍完全无偏；跨不同训练规模的单因素归因 |
| Public AUC | 用户提供的 Kaggle 截图 | 具体提交在公开榜样本上的实际表现 | 私有榜最终表现；极小差距的统计显著性 |

早期 Notebook `02` 用 20% 验证集，模型比较最多使用 12 万训练行、调参最多 8 万行；Notebook `03` 仍保留 20% holdout，但树模型训练最多使用 14 万行，另有 4 万 tuning。后期脚本通常先留出 15%，再从剩余部分划出约 10 万 tuning，模型拟合约 46.8 万行。[Notebook 02](../../notebooks/02_feature_engineering_model_comparison.ipynb)；[Notebook 03](../../notebooks/03_advanced_models_and_deep_learning.ipynb)；[优化脚本](../../optimize_xgboost_leaderboard.py)

这意味着：`0.940255 → 0.941855` 同时改变了训练规模、特征和部分参数，不能写成“某一个特征提升了 `0.001600`”。后期多个脚本使用同样的 15% 划分，因此后期候选之间更可比；不过这一 holdout 被反复查看，仍有适应性选择风险。

## 3. 最可信的项目内对照

| 对照 | 保持一致的条件 | 观察结果 | 证据强度 |
|---|---|---|---|
| 原始特征 vs 整组工程特征 | 同一个逻辑回归、训练样本、20% 验证集 | `0.937955 → 0.938371`，`+0.000416` | 可归因于**整组**特征；无法分解单列贡献 |
| 旧特征 vs 精细特征 | 同一个 depth-5 XGBoost 基础参数与 15% 切分 | `0.941714 → 0.941855`，`+0.000141` | 可归因于**精细特征组**，同时早停轮数略有变化 |
| `max_bin=256…8192` | 设计上沿用同一后期特征、切分和 XGBoost 参数 | 留出 AUC 逐步升至 `0.943136` | 趋势清晰；`256–4096` 行是探针脚本中的历史数值 |
| V2 旧 1800 轮 vs 延长 depth-5 | 后期同一切分、相同 `max_bin=4096` 和 depth-5 主参数 | `0.942993 → 0.943028`，`+0.000035` | 小幅改善；早停设置和训练运行也不同 |

来源：[Notebook 02](../../notebooks/02_feature_engineering_model_comparison.ipynb)、[精细特征结果](../../artifacts/xgboost_optimization_results.csv)、[fine-bin 结果](../../artifacts/ranking_model_probe_results.csv)、[V2 结果](../../artifacts/main_xgboost_v2_results.csv)。

## 4. 怎样理解“为什么变好”

本项目的证据通常是模型之间的预测分数，不是对数据生成过程的直接观察。因此要把三层说法分开：

1. **观察**：提高 `max_bin` 后，在记录的同一后期切分上 AUC 上升。这是表格直接支持的。
2. **机制推断**：更细的直方图切点可能保留收入、通勤等连续变量的局部排序，这是 XGBoost `hist` 的工作机制所支持的解释。
3. **未证实的细节**：究竟是收入、通勤还是其他字段贡献了多少提升；当前没有逐列消融、不同随机切分或最终私榜证据，不能给出定量归因。

同理，深树表现下降可以与容量增加、噪声拟合相容，却不能仅凭一张 AUC 表断定“已证明过拟合”。本复盘会用“可能”“与结果相符”标记这样的机制解释。

## 5. 证据层级与来源索引

| 层级 | 具体材料 | 用途 |
|---|---|---|
| 原始实验输出 | [Notebook 02](../../notebooks/02_feature_engineering_model_comparison.ipynb)、[Notebook 03](../../notebooks/03_advanced_models_and_deep_learning.ipynb) 的保存输出 | 早期模型、消融、候选调参、神经网络训练曲线 |
| 机器可读结果 | [`artifacts/`](../../artifacts) 下的 CSV/JSON | 后期 AUC、最优轮数、融合权重、训练时间 |
| 可复核实现 | [早期优化](../../optimize_xgboost_leaderboard.py)、[fine-bin 探针](../../ranking_model_probe.py)、[异构融合](../../compare_diverse_ensembles.py)、[V2](../../optimize_main_xgboost_v2.py) 等脚本 | 确认实际使用的特征、切分、参数与权重搜索方式 |
| 外部反馈 | 用户在对话里提供的 Kaggle 截图 | 只用于标注已观察到的 Public 分数；仓库内没有榜单导出作为独立复核 |

特别注意：`ranking_model_probe.py` 中 `256–4096` 的 `rows` 在脚本内先以固定数字构造，随后循环只训练 `8192`。它们与前期实验记录一致，但如果要达到逐次运行级别的完全溯源，还需要当时的原始日志或重新运行完整网格。
