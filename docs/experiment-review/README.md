# 电动汽车购买预测：实验复盘目录

这组文档回答三个问题：特征和模型是如何逐轮变化的；哪些改变确实伴随分数改善或下降；现有证据能解释到什么程度。适合不熟悉本项目的读者按顺序阅读，也可直接跳到需要的主题。

## 阅读目录

1. [证据、评价指标与比较边界](01-evidence-and-validation.md)：先弄清 AUC、数据切分、公开榜与本地分数，以及“观察到改善”和“证明原因”的区别。
2. [特征工程的每一次变化](02-feature-engineering.md)：从原始字段、基础衍生特征，到精细交互和 `Core_Context`；列出有消融证据与没有单独证据的部分。
3. [经典机器学习模型及调参](03-classical-models.md)：常数基线、逻辑回归、样条模型、HistGradientBoosting、Random Forest、Extra Trees 和线性 SVM。
4. [XGBoost、LightGBM、CatBoost 与神经网络](04-boosting-and-neural.md)：逐阶段参数、选择依据、改善与退化，以及未实际运行的模型边界。
5. [集成学习为什么有时有效、有时无效](05-ensembles.md)：等权融合、种子平均、概率融合、rank 融合和验证集学权重。
6. [逐轮实验账本与最终核对](06-iteration-audit.md)：把 Notebook `01`–`08`、结果文件、公开榜截图和可支持的结论逐一对齐。

## 先看结论

- 早期在**相同逻辑回归、相同训练与验证样本**下，整组工程特征使 ROC-AUC 从 `0.937955` 升至 `0.938371`。后期在**相同 XGBoost 基础参数与切分**下，精细特征组使留出 AUC 从 `0.941714` 升至 `0.941855`。两次实验均只能证明“整组特征有效”，不能给其中某一个特征单独记功。[Notebook 02](../../notebooks/02_feature_engineering_model_comparison.ipynb)；[结果表](../../artifacts/xgboost_optimization_results.csv)
- 目前记录中最明显的单一参数方向是提高 XGBoost 的 `max_bin`：在 fine-bin 结果表中，留出 AUC 从 `256` 的 `0.941832` 升至 `8192` 的 `0.943136`。其中 `256–4096` 行由既有实验数值写入探针脚本，`8192` 在脚本中重新拟合；因此这是强而有边界的实验线索，不能说每一行都由同一次脚本执行生成。[结果表](../../artifacts/ranking_model_probe_results.csv)；[探针脚本](../../ranking_model_probe.py)
- 最高**已记录本地留出 AUC**是 `max_bin=8192` XGBoost 的 `0.943136`。用户提供的公开榜截图中，`4096` 双种子提交为 `0.94313`，`8192` 单种子提交为 `0.94311`；差距只有 `0.00002`，不能证明双种子方案普遍优于 `8192`。[本地结果](../../artifacts/ranking_model_probe_results.csv)；[Notebook 07](../../notebooks/07_rf_svm_and_top_model_blends.ipynb)
- 多个不同名称的弱模型没有带来有效融合。RF、Extra Trees、线性 SVM、LightGBM GOSS、CatBoost Lossguide 的验证式融合权重多数落在零附近；相反，几个质量接近的 XGBoost 结构得到小幅本地增益。[异构权重](../../artifacts/diverse_ensemble_weights.json)；[RF/SVM 权重](../../artifacts/rf_svm_pairwise_ensembles.csv)；[V2 结果](../../artifacts/main_xgboost_v2_results.csv)

## 重要阅读约定

- 文中“本地”表示脚本或 Notebook 在有标签训练数据上计算的 AUC；“Public”表示用户提供的 Kaggle 公开榜截图，仓库没有可复核的榜单导出文件。
- 分数差异只在相同切分、样本量、特征和评价流程下才适合归因于某个单独改变。早期 `02`、`03` 与后期 `04`–`08` 的训练规模不同，不能把跨阶段差值解释为单一参数的效果。
- 对“为什么”的解释分为两级：实验直接支持的观察，以及与模型机制一致、但尚未做单特征或多随机切分实验验证的推断。
- 后期多次查看同一个留出集，因此它对本项目仍有比较价值，但不能再被称作跨所有实验完全独立、无选择偏差的最终评估。

文档基于仓库当前版本及用户在对话中提供的公开榜截图整理；没有重新训练全套模型，也没有对未记录的提交推测分数。
