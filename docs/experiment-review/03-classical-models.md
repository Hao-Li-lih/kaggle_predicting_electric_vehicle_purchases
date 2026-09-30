# 03｜经典机器学习：逐模型调参和得失

[返回目录](README.md) · [上一篇：特征工程](02-feature-engineering.md) · [下一篇：提升树与神经网络](04-boosting-and-neural.md)

下面按实际运行过的模型列出参数与结果。早期 Notebook `02` 的模型只用 12 万行开发样本；后期 RF/SVM 探针使用约 46.8 万行和不同特征，不能将两次成绩直接解释为某个参数带来的改善。参数后若写“尝试”，表示确实运行；“可能原因”是与模型结构相符的解释，不是单独因果检验。[验证边界](01-evidence-and-validation.md)

## 1. 常数概率基线

Notebook `02` 使用 `DummyClassifier(strategy="prior")`，始终预测训练部分的正类比例。验证 ROC-AUC 为 `0.500000`，Average Precision 约 `0.174646`。它不是参赛候选，作用是确认所有后续模型确实学到了可用排序，也为 Log Loss、Brier Score 提供最低比较起点。[Notebook 02，代码单元 14、17](../../notebooks/02_feature_engineering_model_comparison.ipynb)

## 2. 逻辑回归：工程特征有用，正则强度影响很小

逻辑回归将预处理后的特征按线性权重相加，再映射为概率。类别 one-hot、数值标准化与人工交互使它既容易训练，也能测试特征是否表达了有用结构。基线使用 `C=1.0`、`max_iter=600`，验证 AUC 为 `0.938371`。[Notebook 02](../../notebooks/02_feature_engineering_model_comparison.ipynb)

对 `C∈{0.03,0.10,0.30,1.00,3.00}` 进行三折分层交叉验证，最高平均 CV AUC 是 `C=0.30` 的 `0.938590`；将该模型在相同开发样本重拟合后，独立验证 AUC 为 `0.938379`，仅比基线高 `0.000008`。`C` 越小代表更强的 L2 正则，但五档 CV 分数极接近，现有实验不支持“精细调 `C` 会产生大幅提升”。真正较明确的改进是整组工程特征的 `+0.000416`。[Notebook 02，代码单元 21、24、29](../../notebooks/02_feature_engineering_model_comparison.ipynb)

## 3. 样条逻辑模型：更灵活的主效应仍不足以匹配树模型

`structured_model_probe.py` 对 `Age`、`Annual_Income_USD`、`Daily_Commute_km` 做三次样条变换，分别测试 `n_knots∈{5,8,12}` 和逻辑回归 `C∈{0.1,1,10}`，共九个组合；其他类别使用 one-hot，并额外表示核心上下文、站点与城市等类别组合。最高记录 AUC 为 `12` 个结点、`C=1` 的 `0.939556`；`8` 结点、`C=1` 为 `0.939395`，`5` 结点、`C=1` 为 `0.939330`。增加结点与改善相伴，但幅度很小；同一结点数内调 `C` 的效果更小。[脚本](../../structured_model_probe.py)；[结果表](../../artifacts/structured_model_probe_results.csv)

**为什么可能落后**：样条给单个连续变量平滑曲线，但线性分类头仍主要按加法组合各项；本数据中补贴、焦虑和环保关注形成强条件分段，树模型更容易表达高阶交互。需要说明的是，这九个候选直接按同一个 15% holdout 比较，没有另设 tuning，因此最高值带有参数选择偏差；它也不是对样条模型潜力的穷尽搜索。[脚本](../../structured_model_probe.py)

## 4. HistGradientBoosting：早期最好的 sklearn 模型

Notebook `02` 的首轮设定为 `learning_rate=0.08`、`max_iter=180`、`max_leaf_nodes=31`、`min_samples_leaf=20`、`l2_regularization=1.0`，验证 AUC `0.939849`，高于当时的逻辑回归、RF 和 Extra Trees。[Notebook 02，代码单元 14、17](../../notebooks/02_feature_engineering_model_comparison.ipynb)

随后用 8 次随机搜索、三折分层 CV，在如下候选空间中调参：

| 参数 | 搜索值 | 通俗作用 |
|---|---|---|
| `learning_rate` | `0.03, 0.05, 0.08, 0.12` | 每棵树修正多少；小步长通常需要更多树 |
| `max_iter` | `120, 180, 260` | 最多训练多少次提升 |
| `max_leaf_nodes` | `15, 31, 63` | 每棵树可分成多少局部区域 |
| `max_depth` | `None, 8, 12` | 限制树的最大层数 |
| `min_samples_leaf` | `20, 50, 100` | 叶子至少需要多少样本，控制小群体噪声 |
| `l2_regularization` | `0, 1, 5` | 限制叶子输出的极端程度 |

这 8 次搜索选中 `learning_rate=0.03`、`max_iter=260`、`max_leaf_nodes=15`、`max_depth=None`、`min_samples_leaf=100`、`l2_regularization=0`，平均 CV AUC 为 `0.939410`。独立验证 AUC 达到 `0.939935`，比该 Notebook 首轮设定高 `0.000086`。较小步长、更少叶子、更多叶样本与当前小幅改善相符，但六项参数同时变化，不能把收益单独归因于某一项。搜索只抽了 8 个组合，不能称为全局最优。[Notebook 02，代码单元 26、29](../../notebooks/02_feature_engineering_model_comparison.ipynb)

## 5. Random Forest：两轮设定都落后于提升树

Random Forest 对不同 bootstrap 样本建立多棵树并平均。它擅长降低单棵树方差，但与逐棵纠错的 boosting 有不同学习机制。

| 阶段 | 主要参数 | 相应验证结果 |
|---|---|---:|
| Notebook `02` 基线 | `n_estimators=120`、`max_depth=16`、`min_samples_leaf=5`、`max_features="sqrt"` | 20% 验证 AUC `0.937868` |
| 后期探针 | `n_estimators=160`、`criterion="log_loss"`、`max_depth=22`、`min_samples_leaf=10`、`max_features="sqrt"`、`max_samples=0.80` | 15% holdout AUC `0.939658` |

第二轮分数更高，**不能说是这些参数让 RF 提升**：训练样本、特征和切分都变了。与同轮 fine-bin XGBoost `0.942993` 相比，RF 仍低 `0.003335`；在 tuning 集尝试给 RF `1%–30%` 的 rank 融合权重，最好是 `0%`。可能的原因是 RF 平均后较难像 boosting 一样不断修正核心条件分段内的细微排序，但没有训练误差分析可以证明这就是唯一原因。[Notebook 02](../../notebooks/02_feature_engineering_model_comparison.ipynb)；[后期实现](../../compare_rf_svm_models.py)；[模型结果](../../artifacts/rf_svm_model_comparison.csv)；[权重结果](../../artifacts/rf_svm_pairwise_ensembles.csv)

## 6. Extra Trees：随机切点没有带来需要的细粒度排序

Notebook `02` 使用 `n_estimators=120`、`max_depth=18`、`min_samples_leaf=3`、`max_features="sqrt"`，20% 验证 AUC 为 `0.936880`。后期异构实验使用 `n_estimators=60`、`criterion="entropy"`、`max_features=0.60`、`min_samples_leaf=5`，15% holdout AUC 为 `0.935314`。两轮设计不同，不能把下降归咎于某一参数。[Notebook 02](../../notebooks/02_feature_engineering_model_comparison.ipynb)；[异构脚本](../../compare_diverse_ensembles.py)；[异构结果](../../artifacts/diverse_model_comparison.csv)

后期对数损失优化给它几乎零权重，rank 融合给零权重。Extra Trees 的随机分裂可提供不同预测，但目前这种差异不足以抵消单模型质量差距；这解释了为什么“模型多样”不自动等于“集成有益”。[异构权重](../../artifacts/diverse_ensemble_weights.json)

## 7. 线性 SVM：比较的是排序，不是概率

后期脚本先用训练数据拟合 `MaxAbsScaler`，再训练 `LinearSVC`，测试 `C∈{0.01,0.1,1.0}`，共用 `max_iter=5000`、`tol=1e-5`。评估的是 `decision_function` 决策分数的 AUC；该分数不是校准概率，因此脚本只报告 AUC 与 Average Precision，没有为 SVM 报 Log Loss。[实现](../../compare_rf_svm_models.py)

| `C` | Tuning AUC | Holdout AUC |
|---:|---:|---:|
| `0.01` | `0.936524` | `0.937074` |
| `0.1` | `0.935259` | `0.936183` |
| `1.0` | `0.936444` | **`0.937208`** |

调优集最高是 `C=0.01`，留出集最高却是 `C=1`，说明这一小范围的排名不稳定，不能根据 holdout 反向声称 `C=1` 是事先选出的最佳参数。三个版本与 XGBoost 的 rank 融合最优权重均为零。线性边界即使接收手工交互，也难以充分拟合复杂分段，是与结果相符的解释；没有运行 RBF 核 SVM，因此不能评价“所有 SVM”。[结果](../../artifacts/rf_svm_model_comparison.csv)；[融合结果](../../artifacts/rf_svm_pairwise_ensembles.csv)

## 8. XGBoost Random Forest 变体

`XGBRFClassifier` 虽属于 XGBoost 库，这里按 bagging 机制与 RF 一起比较。后期设定为 `n_estimators=350`、`learning_rate=1.0`、`max_depth=8`、`min_child_weight=10`、`subsample=0.80`、`colsample_bynode=0.80`、`reg_lambda=4.0`、`max_bin=1024`，holdout AUC 为 `0.939390`，pairwise rank 融合最优权重同样为零。这只评价了该组参数，不能排除所有 XGBRF 设定。[脚本](../../compare_rf_svm_models.py)；[结果](../../artifacts/rf_svm_model_comparison.csv)
