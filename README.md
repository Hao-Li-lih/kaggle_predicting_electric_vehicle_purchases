# Predicting Electric Vehicle Purchases

[中文说明](#中文说明) | [English Documentation](#english-documentation)

本仓库记录 Kaggle 电动汽车购买预测竞赛的完整建模过程：从数据探索、特征工程和经典机器学习基线，到高分辨率 XGBoost、异构集成与类别嵌入神经网络。所有数据列名和代码均使用英文，Notebook 中使用中文 Markdown 解释方法、用法与实验思路。

This repository contains the complete modeling workflow for a Kaggle competition that predicts the probability of electric-vehicle purchase, covering exploration, feature engineering, classical baselines, high-resolution XGBoost, heterogeneous ensembles, and a categorical-embedding neural network.

---

# 中文说明

## 1. 项目目标

任务是根据用户收入、年龄、通勤距离、充电条件、环保意识、补贴认知、里程焦虑等信息，预测二分类目标 `Will_Buy_EV` 为 1 的概率。

- 评估指标：ROC-AUC
- 训练集规模：约 668,000 行
- 测试集规模：约 287,000 行
- 标识列：`id`
- 目标列：`Will_Buy_EV`
- 最终输出：每个测试样本购买电动汽车的概率

ROC-AUC 只关心预测排序，因此本项目不仅比较单模型精度，也重点研究 rank averaging、OOF/独立验证集权重学习以及不同随机种子的集成稳定性。

## 2. 当前成绩

### 2.1 Kaggle Public Leaderboard

| 模型或提交 | Public ROC-AUC |
|---|---:|
| XGBoost `max_bin=4096` 双种子集成 | **0.94313** |
| XGBoost `max_bin=8192` 单模型 | 0.94311 |
| 第一版优化 XGBoost 种子集成 | 0.94187 |
| 原始 XGBoost | 0.94154 |
| GBDT 等权融合 | 0.94151 |
| CatBoost | 0.94095 |
| PyTorch 类别嵌入网络 | 0.93805 |

### 2.2 本地独立验证集

| 实验 | ROC-AUC |
|---|---:|
| XGBoost `max_bin=8192` | **0.943136** |
| XGBoost V2 结构化 rank ensemble | 0.943040 |
| XGBoost V2 extended depth-5 | 0.943028 |
| XGBoost `max_bin=4096` baseline | 0.942993 |

本地与线上结果共同说明：该数据集的主要提升来自高分辨率直方图切分、可靠的条件分段以及稳定的多种子平均，而不是无限增加人工交叉项。

## 3. 数据与目录准备

从 Kaggle 下载以下文件并放入 `data/`：

```text
data/
  train.csv
  test.csv
  sample_submission.csv
```

数据文件、外部数据、虚拟环境、训练日志和生成的提交文件默认不会上传到 Git：

```text
data/
external_data/
.venv/
catboost_info/
submissions/
```

这样可以避免上传竞赛数据、体积较大的临时文件和可重复生成的预测结果。

## 4. 特征工程

特征设计围绕“购买能力、日常用车需求、充电便利性、政策与心理因素”展开。

### 4.1 充电条件

- 公共与家庭充电总量、最小值、最大值和差值
- 每个充电站对应的通勤距离
- 公共充电与家庭充电之间的平衡关系
- 充电条件与环境意识的交互

### 4.2 收入与车辆拥有能力

- `log1p` 收入变换
- 收入平方项
- 人均车辆或每辆车对应收入
- 单位通勤距离收入
- 收入与补贴、家庭充电条件的交互

### 4.3 年龄与通勤

- 年龄平方项，用于表达非线性生命周期效应
- 通勤距离对数与比例特征
- 里程焦虑与通勤距离、充电条件的交互

### 4.4 行为与心理变量

- 环保关注度与补贴、家庭充电的交互
- 环保关注度与里程焦虑的差值
- `Core_Context`：综合环保关注、补贴认知、里程焦虑和家庭充电的核心条件分段

所有派生列都保持英文命名，中文只用于 Markdown 说明。

## 5. 已尝试的模型

### 5.1 经典机器学习

- Logistic Regression
- Spline Logistic GAM
- HistGradientBoosting
- Random Forest
- Extra Trees
- Linear SVM

这些模型用于建立可解释基线、测试平滑非线性结构，并为异构集成提供误差差异。

### 5.2 梯度提升树

- XGBoost：histogram、Lossguide、Random Forest 变体及不同深度/学习率
- LightGBM：GBDT 与 GOSS
- CatBoost：Symmetric Tree 与 Lossguide

XGBoost 是当前主模型。关键改进是将 `max_bin` 从常见的 256 逐步提高到 4096/8192，使连续变量的细粒度排序得到保留。

### 5.3 深度学习

- PyTorch 类别嵌入网络
- 数值特征标准化、类别特征 embedding、MLP 分类头
- 使用验证集 ROC-AUC、early stopping 和多随机种子比较

神经网络表现低于主树模型，但可作为具有不同归纳偏置的候选模型。当前数据仍更适合高质量树模型。

### 5.4 外部数据实验

项目测试了原始 EV adoption 数据集的先验信息，但由于原始数据与合成竞赛数据存在分布偏移，外部先验降低了验证表现，因此没有用于最终提交。

## 6. 验证与集成策略

为避免把公开榜单当成调参验证集，实验遵循以下原则：

1. 使用固定的分层 train/tune/holdout 划分。
2. 超参数和融合权重只在 tune 集上选择。
3. holdout 仅用于最终无偏比较。
4. Kaggle Public Leaderboard 作为事后检查，不直接替代本地验证。
5. 概率平均与 rank averaging 同时比较；ROC-AUC 场景通常更偏向稳定的排序融合。
6. 只有在验证集上带来互补收益的模型才进入集成，模型“不同”本身并不代表有效。

最新 V2 结构化融合在本地学习到的主要权重约为：

- 62.3%：extended depth-5 XGBoost
- 26.7%：low-learning-rate depth-5 XGBoost
- 11.0%：Lossguide XGBoost

最终还会通过多个随机种子平均降低方差，并生成 rank ensemble 与少量历史最佳提交混合版本。

## 7. 实验演进

1. **基础探索与基线**：识别变量类型、缺失值、分布和目标关系。
2. **高级 GBDT 与神经网络**：比较 XGBoost、LightGBM、CatBoost 和 PyTorch embedding。
3. **排行榜优化**：加入领域交互、随机种子集成及外部数据检验。
4. **Fine-bin XGBoost**：系统比较 `max_bin=256` 到 `8192`，取得最明显提升。
5. **异构模型检验**：加入 RF、Extra Trees、SVM、GOSS、Lossguide 等模型，分析误差相关性。
6. **XGBoost V2**：延长训练、降低学习率、组合 depth/Lossguide 结构并训练五种子集成。

## 8. Notebook 说明

```text
notebooks/
  01_data_exploration.ipynb
  02_feature_engineering_model_comparison.ipynb
  03_advanced_models_and_deep_learning.ipynb
  04_leaderboard_optimization.ipynb
  05_model_rethinking_and_finebin_xgboost.ipynb
  06_diverse_models_and_validated_ensembles.ipynb
  07_rf_svm_and_top_model_blends.ipynb
  08_xgboost_v2_optimization_and_ensembles.ipynb
```

- `01`：数据质量、分布、类别/数值变量和目标关系探索。
- `02`：领域特征工程、预处理流水线和传统模型对比。
- `03`：XGBoost、LightGBM、CatBoost 与 PyTorch 网络。
- `04`：排行榜反馈后的特征、种子平均和提交策略。
- `05`：重新审视数据结构并验证高分辨率 XGBoost。
- `06`：异构树模型、验证式融合和相关性分析。
- `07`：RF、Extra Trees、SVM 与历史最佳模型混合。
- `08`：XGBoost V2 深度优化、结构化融合及五种子训练。

Notebook 已按从上到下可重复执行的方式组织，并保留关键输出用于复核。

## 9. 脚本说明

基础与高级模型：

```text
generate_advanced_submissions.py
optimize_xgboost_leaderboard.py
structured_model_probe.py
original_prior_probe.py
```

高分辨率 XGBoost：

```text
ranking_model_probe.py
train_finebin_xgboost.py
```

异构模型与集成：

```text
compare_diverse_ensembles.py
compare_rf_svm_models.py
train_validated_depth_ensemble.py
create_top_model_blends.py
```

最新主模型：

```text
optimize_main_xgboost_v2.py
train_xgboost_v2_ensemble.py
```

实验指标、权重和摘要保存在 `artifacts/`，便于比较并避免只依赖 Notebook 输出。

## 10. 安装与运行

推荐在 Windows PowerShell 中创建独立环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

运行最新优化与训练流程：

```powershell
python optimize_main_xgboost_v2.py
python train_xgboost_v2_ensemble.py
```

训练完成后，提交文件会写入 `submissions/`。当前优先验证的候选文件为：

1. `submission_xgb4096_extended_5seed_ensemble.csv`
2. `submission_xgb_v2_structural_rank_ensemble.csv`
3. `submission_xgb_v2_public_blend_80_20.csv`

提交前请确认列顺序与 `sample_submission.csv` 完全一致、`id` 无变化、预测值位于 `[0, 1]` 且没有缺失值。

## 11. 复现与注意事项

- 固定 Python、依赖版本和所有模型随机种子。
- 先运行 Notebook 中的数据检查，再开始长时间训练。
- 高 `max_bin` 会显著增加内存和训练时间，应根据设备资源调整。
- 公开榜单只覆盖部分测试集；小幅领先可能来自抽样波动，应优先相信稳定的本地验证结果。
- 竞赛数据与生成提交文件不包含在仓库中，需由使用者从 Kaggle 获取或自行生成。

---

# English Documentation

## 1. Objective

The task is to predict the probability that `Will_Buy_EV` equals 1 from information such as income, age, commute distance, charging access, environmental concern, subsidy awareness, and range anxiety.

- Metric: ROC-AUC
- Training size: approximately 668,000 rows
- Test size: approximately 287,000 rows
- Identifier: `id`
- Target: `Will_Buy_EV`
- Output: one EV-purchase probability per test row

Because ROC-AUC evaluates ordering rather than a fixed classification threshold, the project studies probability blending, rank averaging, validation-based ensemble weighting, and seed averaging in addition to standalone models.

## 2. Current Results

### 2.1 Kaggle Public Leaderboard

| Model or submission | Public ROC-AUC |
|---|---:|
| XGBoost `max_bin=4096`, two-seed ensemble | **0.94313** |
| XGBoost `max_bin=8192`, single model | 0.94311 |
| First optimized XGBoost seed ensemble | 0.94187 |
| Original XGBoost | 0.94154 |
| Equal-weight GBDT blend | 0.94151 |
| CatBoost | 0.94095 |
| PyTorch categorical-embedding network | 0.93805 |

### 2.2 Local Holdout

| Experiment | ROC-AUC |
|---|---:|
| XGBoost `max_bin=8192` | **0.943136** |
| XGBoost V2 structural rank ensemble | 0.943040 |
| XGBoost V2 extended depth-5 | 0.943028 |
| XGBoost `max_bin=4096` baseline | 0.942993 |

The local and public results suggest that high-resolution histogram splits, reliable conditional segmentation, and seed averaging are more valuable than continually adding handcrafted interactions.

## 3. Data Setup

Download the Kaggle files and place them under `data/`:

```text
data/
  train.csv
  test.csv
  sample_submission.csv
```

Competition data, external datasets, virtual environments, model logs, and generated submissions are intentionally excluded from Git:

```text
data/
external_data/
.venv/
catboost_info/
submissions/
```

This keeps licensed competition data, large temporary files, and reproducible predictions out of version control.

## 4. Feature Engineering

The feature set follows four domain themes: purchasing power, daily driving demand, charging convenience, and policy or behavioral factors.

### 4.1 Charging access

- Total, minimum, maximum, and gap across public and home charging variables
- Commute distance per available station
- Balance between public and home charging access
- Charging access interacted with environmental concern

### 4.2 Income and vehicle affordability

- `log1p` income transformation
- Squared income
- Income per owned vehicle
- Income per unit of commute distance
- Income interactions with subsidy awareness and home charging

### 4.3 Age and commuting

- Squared age for nonlinear life-cycle effects
- Logarithmic and ratio-based commute features
- Range anxiety interacted with commute distance and charging access

### 4.4 Behavioral variables

- Environmental concern interacted with subsidy awareness and home charging
- Difference between environmental concern and range anxiety
- `Core_Context`, a compact segment combining concern, subsidy awareness, anxiety, and home charging

All source and engineered column names remain in English. Chinese is used only in explanatory Markdown cells.

## 5. Models Evaluated

### 5.1 Classical machine learning

- Logistic Regression
- Spline Logistic GAM
- HistGradientBoosting
- Random Forest
- Extra Trees
- Linear SVM

These models establish interpretable baselines, test smooth nonlinear structure, and provide potentially different errors for heterogeneous ensembles.

### 5.2 Gradient-boosted trees

- XGBoost: histogram, Lossguide, random-forest variants, and multiple depth/learning-rate settings
- LightGBM: GBDT and GOSS
- CatBoost: Symmetric Tree and Lossguide

XGBoost remains the strongest model family. The most important improvement was increasing `max_bin` from the usual 256 to 4096/8192, preserving finer ordering within continuous variables.

### 5.3 Deep learning

- PyTorch categorical-embedding network
- Standardized numeric inputs, categorical embeddings, and an MLP head
- Validation ROC-AUC, early stopping, and multiple-seed comparisons

The neural network trails the main tree models, although it remains a useful candidate with a different inductive bias. The current tabular structure still favors well-tuned boosting.

### 5.4 External-data experiment

The original EV-adoption dataset was evaluated as a source of prior information. Distribution shift between the original and synthetic competition data reduced validation performance, so the external prior is not used in final submissions.

## 6. Validation and Ensembling

The workflow avoids treating the public leaderboard as the primary validation set:

1. Use a fixed stratified train/tune/holdout split.
2. Select hyperparameters and ensemble weights on the tune split only.
3. Use the holdout split once for final, less-biased comparisons.
4. Treat the Kaggle public leaderboard as posterior evidence rather than a replacement for local validation.
5. Compare probability averaging with rank averaging; stable rank blends are often effective for ROC-AUC.
6. Add a model to an ensemble only when it provides validated complementary value. Diversity alone is insufficient.

The latest V2 structural blend learned approximate local weights of:

- 62.3% extended depth-5 XGBoost
- 26.7% low-learning-rate depth-5 XGBoost
- 11.0% Lossguide XGBoost

Multiple seeds are then averaged to reduce variance. Rank ensembles and conservative blends with the previous public best are also produced.

## 7. Experiment Progression

1. **Exploration and baselines**: inspect types, missingness, distributions, and target relationships.
2. **Advanced GBDT and neural models**: compare XGBoost, LightGBM, CatBoost, and a PyTorch embedding network.
3. **Leaderboard optimization**: add domain interactions, seed averaging, and an external-data check.
4. **Fine-bin XGBoost**: compare `max_bin=256` through `8192`, producing the largest improvement.
5. **Diverse model audit**: evaluate RF, Extra Trees, SVM, GOSS, and Lossguide variants and their error correlations.
6. **XGBoost V2**: extend training, reduce learning rates, combine depth/Lossguide structures, and train a five-seed ensemble.

## 8. Notebooks

```text
notebooks/
  01_data_exploration.ipynb
  02_feature_engineering_model_comparison.ipynb
  03_advanced_models_and_deep_learning.ipynb
  04_leaderboard_optimization.ipynb
  05_model_rethinking_and_finebin_xgboost.ipynb
  06_diverse_models_and_validated_ensembles.ipynb
  07_rf_svm_and_top_model_blends.ipynb
  08_xgboost_v2_optimization_and_ensembles.ipynb
```

- `01`: data quality, distributions, variable types, and target analysis.
- `02`: domain feature engineering, preprocessing pipelines, and classical model comparison.
- `03`: XGBoost, LightGBM, CatBoost, and the PyTorch network.
- `04`: leaderboard-informed features, seed averaging, and submission strategy.
- `05`: structural reassessment and high-resolution XGBoost experiments.
- `06`: diverse tree models, validated blends, and correlation analysis.
- `07`: RF, Extra Trees, SVM, and blends with historical top models.
- `08`: XGBoost V2 optimization, structural blending, and five-seed training.

The notebooks are organized for top-to-bottom reproducibility and retain key outputs for review.

## 9. Scripts

Baseline and advanced modeling:

```text
generate_advanced_submissions.py
optimize_xgboost_leaderboard.py
structured_model_probe.py
original_prior_probe.py
```

High-resolution XGBoost:

```text
ranking_model_probe.py
train_finebin_xgboost.py
```

Diverse models and ensembles:

```text
compare_diverse_ensembles.py
compare_rf_svm_models.py
train_validated_depth_ensemble.py
create_top_model_blends.py
```

Latest main-model workflow:

```text
optimize_main_xgboost_v2.py
train_xgboost_v2_ensemble.py
```

Metrics, learned weights, and experiment summaries are stored in `artifacts/` so that results can be compared without relying only on notebook output.

## 10. Installation and Execution

Create an isolated environment in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the latest optimization and training workflow:

```powershell
python optimize_main_xgboost_v2.py
python train_xgboost_v2_ensemble.py
```

Generated files are written to `submissions/`. The current priority candidates are:

1. `submission_xgb4096_extended_5seed_ensemble.csv`
2. `submission_xgb_v2_structural_rank_ensemble.csv`
3. `submission_xgb_v2_public_blend_80_20.csv`

Before uploading, verify that the column order exactly matches `sample_submission.csv`, IDs are unchanged, predictions are within `[0, 1]`, and no values are missing.

## 11. Reproducibility Notes

- Pin Python and dependency versions and keep all random seeds fixed.
- Run the notebook data checks before starting long training jobs.
- Large `max_bin` values substantially increase memory use and training time; adjust them to the available hardware.
- The public leaderboard covers only part of the test set. Small differences may be sampling noise, so stable local validation should remain the primary decision criterion.
- Competition data and generated submissions are not included and must be downloaded from Kaggle or recreated locally.

## License and Usage

The code is provided for educational and competition research purposes. Users are responsible for complying with Kaggle competition rules and the licenses of the underlying datasets and libraries.
