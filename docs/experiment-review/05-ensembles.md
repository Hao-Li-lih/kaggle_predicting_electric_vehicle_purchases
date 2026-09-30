# 05｜集成学习：哪些融合有效，哪些只增加噪声

[返回目录](README.md) · [上一篇：提升树与神经网络](04-boosting-and-neural.md) · [下一篇：逐轮账本](06-iteration-audit.md)

集成是把若干模型的预测合成一个分数。它有用的前提是：参与者各自足够准确，而且它们在关键样本上的错误不完全相同。仅仅把算法名称凑在一起，通常不足以提升 ROC-AUC。本项目依次尝试等权概率平均、调优集优化权重、百分位 rank 融合，以及同模型多随机种子平均。

## 1. 第一轮：三种 GBDT 等权在本地略好，公开榜没有保持优势

Notebook `03` 将选出的 XGBoost、CatBoost、LightGBM 概率各占三分之一。20% holdout AUC：XGBoost `0.940255`，三 GBDT 等权 `0.940314`，本地增量 `+0.000059`。这说明三者在该验证集上有轻微互补，但增量很小。加入 PyTorch 网络后四模型等权降为 `0.940112`，比三 GBDT 低 `0.000202`；弱模型即使有不同预测，也可能破坏原来的排序。[Notebook 03，代码单元 24](../../notebooks/03_advanced_models_and_deep_learning.ipynb)

用户提供的公开榜截图恰好显示相反的极小差距：XGBoost `0.94154`，三 GBDT 等权 `0.94151`，XGBoost 高 `0.00003`。这不能证明本地融合算法有错，只说明 `+0.000059` 的本地收益并未稳定转化到公开榜。因此后续更偏向以 XGBoost 为主的保守融合。[Notebook 04](../../notebooks/04_leaderboard_optimization.ipynb)

## 2. 随机种子平均：目标是降低单次训练波动

后期 fine-bin 脚本在相同 `max_bin=4096` 参数下训练种子 `42` 与 `314`，对测试概率取均值；`8192` 只训练种子 `42`。用户提供的公开榜分数分别为 `0.94313` 与 `0.94311`。因两者同时改变 `max_bin` 与种子数量，不能把公开榜差距专门归因于平均种子。[训练脚本](../../train_finebin_xgboost.py)；[Notebook 07](../../notebooks/07_rf_svm_and_top_model_blends.ipynb)

V2 将 depth-5、`4096` 模型扩成五种子：`42, 314, 2026, 2718, 8675`，对全量训练所得测试概率求均值。脚本确实生成了提交文件，但仓库没有对应的五种子本地 AUC 或 Public 分数；“方差可能下降”是方法上的预期，不能写成已证实的分数提升。[全量训练脚本](../../train_xgboost_v2_ensemble.py)；[训练记录](../../artifacts/xgboost_v2_training_summary.csv)

## 3. 强模型之间的概率和 rank 融合

在 `4096` 双种子与 `8192` 单种子公开榜接近时，脚本生成 `75/25`、`90/10` 概率融合，以及 `75/25` rank 融合。概率融合保留两个模型原有分数的尺度；rank 融合先把每个测试样本映射为各模型内部的百分位，再平均排序，能减弱概率刻度差异。[脚本](../../create_top_model_blends.py)

后续 V2 还生成 `80%` 五种子 depth-5 与 `20%` 旧 `8192` 的概率混合。**这些权重没有在共同的本地 holdout 上学习或验证，也没有记录公开榜分数**；它们是待提交检验的候选，不是已经观察到提升的结论。[V2 训练脚本](../../train_xgboost_v2_ensemble.py)

早期优化版还生成 `90%` 优化 XGBoost、各 `5%` CatBoost/LightGBM 的混合，以及新旧 XGBoost `75/25` 混合。用户给出的公开榜截图中，两者分别为 `0.94186` 与 `0.94183`，略低于优化 XGBoost 单独的 `0.94187`。这些差距很小，但至少没有提供“融合明确胜出”的证据。[Notebook 04](../../notebooks/04_leaderboard_optimization.ipynb)

## 4. 异构模型：优化器几乎只保留 XGBoost

`compare_diverse_ensembles.py` 在同一后期划分上产生 XGBoost depth 5、depth 3、LightGBM GOSS、Extra Trees、CatBoost Lossguide 的 tuning/holdout 预测。它分别做两件事：[代码](../../compare_diverse_ensembles.py)

1. 对模型概率做非负且和为 1 的权重搜索，以 tuning **Log Loss** 为目标。
2. 把各模型转换为百分位 rank，按 tuning **ROC-AUC** 前向加入模型，每轮尝试 `2%–30%`。

结果：[权重](../../artifacts/diverse_ensemble_weights.json)；[模型与融合 AUC](../../artifacts/diverse_model_comparison.csv)

| 方案 | 主要权重 | Tuning AUC | Holdout AUC | 如何解读 |
|---|---|---:|---:|---|
| depth-5 XGBoost 单模 | 100% | `0.942951` | **`0.942993`** | 该轮主基准 |
| Log Loss 优化概率融合 | depth 5 `98.898%`、depth 3 `1.062%`，其他合计约 `0.04%` | `0.942948` | `0.942992` | 为改善 Log Loss 得到的权重，没有改善 AUC |
| rank 前向融合 | depth 5 `100%` | `0.942951` | `0.942993` | 其他候选在给定搜索网格上没有带来 tuning AUC 增益 |

这里特别要分清目标：概率权重优化的是 **Log Loss**，所以不能说优化器直接证明“按 AUC 最优权重是 98.898%”。rank 搜索才针对 AUC，但只搜索了有限轮次和离散权重。rank 后的数值不是校准概率；结果表中 rank 方案的 Log Loss 不应与真实概率的 Log Loss 用同一含义比较。

## 5. RF、SVM、XGBRF 的两两融合

`compare_rf_svm_models.py` 以 `4096` depth-5 XGBoost 为基准，将 RF、XGBRF 和三档线性 SVM 的分数都转成 rank，逐个搜索 `1%–30%` 候选权重。五个候选的最佳权重**全是零**；holdout AUC 因而与基准完全相同，为 `0.942993`。这不是“RF/SVM 预测与 XGBoost 完全一样”，而是在当前 tuning 集、搜索范围和这些参数下，它们的差异没有提高排序质量。[脚本](../../compare_rf_svm_models.py)；[权重表](../../artifacts/rf_svm_pairwise_ensembles.csv)

## 6. V2：质量接近的结构变体带来极小本地提升

V2 先按 tuning AUC 选中延长版 depth 5，然后对其他 XGBoost 结构做最多三轮 rank 前向搜索，每轮尝试 `1%–30%` 的新权重。得到 depth 5 `62.3%`、低学习率 depth 5 `26.7%`、Lossguide `11.0%`，depth 4 与 depth 6 为零。Holdout AUC 从单模型 `0.943028` 增至 `0.943040`，增量约 `+0.000012`。[脚本](../../optimize_main_xgboost_v2.py)；[权重](../../artifacts/main_xgboost_v2_summary.json)；[结果](../../artifacts/main_xgboost_v2_results.csv)

这是记录中**有同轮本地验证的结构融合增益**，但幅度很小，且仍低于历史 `8192` 单模型的 `0.943136`。全量提交将五种子 depth-5 平均作为 `62.3%` 的主输入，而验证时学权重所用的是单种子 depth-5；最终提交与验证对象并不完全相同，因此不能把 `0.943040` 直接称为该提交的已验证 AUC。[V2 训练脚本](../../train_xgboost_v2_ensemble.py)

## 7. 从这些结果得到的通用判断

模型融合前先检查两件事：单模型质量是否接近主模型；在**未用于训练的 tuning/OOF 预测**上加入它是否提高目标指标。若权重优化反复退化为零，增加相似或较弱模型通常没有价值。微小本地增益应通过新切分或多折 OOF 验证稳定性；公开榜上几万分之一的差距不能单独决定模型机制孰优孰劣。
