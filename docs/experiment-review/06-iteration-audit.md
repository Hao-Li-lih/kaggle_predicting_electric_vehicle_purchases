# 06｜逐轮实验账本与最终核对

[返回目录](README.md) · [上一篇：集成学习](05-ensembles.md)

本页按实际迭代顺序记录每轮的问题、操作、观察结果和解释边界。数字以保存的 Notebook 输出和 `artifacts` 为准；公开榜来自用户在对话中提供的截图，仓库没有对应导出文件。

## 1. 第一轮：先确认数据是否能直接建模

[Notebook 01](../../notebooks/01_data_exploration.ipynb) 记录训练 `668,665` 行、测试 `286,571` 行、正类率约 `17.46%`、无缺失、无整行重复。补贴、里程焦虑、环保关注和收入是强关联字段。因此后续采用分层切分，保留 `id` 做提交对齐，并围绕购买能力、充电便利与行为因素设计特征。这里没有模型提升数字，也不能从单变量正类率推断因果关系。

## 2. 第二轮：整组工程特征和 sklearn 基线

[Notebook 02](../../notebooks/02_feature_engineering_model_comparison.ipynb) 将 13 个原始预测字段扩成 32 个建模字段，并在同一逻辑回归上做消融：AUC `0.937955 → 0.938371`（`+0.000416`）。首轮 HistGradientBoosting `0.939849` 优于逻辑回归 `0.938371`、RF `0.937868`、Extra Trees `0.936880`。经过 8 次随机搜索、三折 CV，HGB 在独立验证达到 `0.939935`（相对首轮 `+0.000086`）。

这轮支持的判断是：整组领域特征有用，较受控的提升树比这些经典基线更适合当前数据。无法证明 19 个新增特征中的任何一列单独有效；调参幅度也很小。

## 3. 第三轮：三种 GBDT 与类别嵌入网络

[Notebook 03](../../notebooks/03_advanced_models_and_deep_learning.ipynb) 分别试两组 CatBoost、LightGBM、XGBoost 参数，并训练一个 PyTorch embedding 网络。20% holdout AUC：GBDT 等权 `0.940314`、XGBoost `0.940255`、CatBoost `0.940118`、LightGBM `0.940113`、神经网络 `0.938041`。四模型等权为 `0.940112`，比三 GBDT 融合低 `0.000202`。

用户最初的公开榜截图给出：XGBoost `0.94154`、GBDT 等权 `0.94151`、CatBoost `0.94095`、神经网络 `0.93805`。这说明等权融合在本地极小优势没有稳定反映到 Public，神经网络加入融合更没有帮助。两组参数只是小范围试验，不能说某个模型族被全面调优到极限。

## 4. 第四轮：扩大训练量、精细特征与外部数据检验

后期[优化脚本](../../optimize_xgboost_leaderboard.py) 使用约 `468,065` 行拟合、`100,300` 行 tuning、`100,300` 行 holdout。`max_bin=256` 的 depth-5 XGBoost，旧特征 holdout `0.941714`，精细特征组 `0.941855`，同轮提升 `+0.000141`。depth 4、更强正则的 depth 6、31 叶 Lossguide 均未超过 tuning 上的 depth-5 基准。[结果表](../../artifacts/xgboost_optimization_results.csv)

最终用选中轮数在全部竞赛训练数据重训，并对种子 `42, 314, 2026` 求平均。[元数据](../../artifacts/xgboost_optimization_metadata.json) 用户后来的公开榜截图中，优化 XGBoost 的提交为 `0.94187`，相比早期 `0.94154` 高 `0.00033`；由于训练规模、特征、早停轮数和种子一起改变，不能将这 `0.00033` 单独记在精细特征名下。两种偏 XGBoost 的融合在 Public 分别为 `0.94186` 和 `0.94183`，没有超过该主模型。[Notebook 04](../../notebooks/04_leaderboard_optimization.ipynb)

## 5. 外部原始数据为什么被排除

“竞赛数据受原始 EV 数据启发”不等于两个来源拥有相同的条件标签关系。主优化脚本把原始数据追加到竞赛训练部分，仅改变其样本权重 `0,1,3,5`，使用同一竞赛 tuning 和 holdout 评估：[结果表](../../artifacts/xgboost_optimization_results.csv)

| 原始数据权重 | Tuning AUC | Holdout AUC | 最优迭代 |
|---:|---:|---:|---:|
| 0 | **`0.941753`** | **`0.941855`** | 1377 |
| 1 | `0.934115` | `0.934545` | 25 |
| 3 | `0.935673` | `0.935920` | 56 |
| 5 | `0.933748` | `0.934178` | 52 |

加入原始数据后，AUC 明显下降且早停轮数骤减；最佳权重为零。独立的[原始数据探针](../../original_prior_probe.py)记录：仅用原始数据训练、在竞赛 holdout 上为 `0.932944`；仅用竞赛数据训练为 `0.941616`。[结果](../../artifacts/original_prior_probe_results.csv) 这与来源间的分布或标签机制差异相符，但现有代码没有分解出具体是哪一种差异。探针还在无标签预处理阶段拼接了 holdout 特征，因此它的数值不是严格训练折预处理的独立复现；主加权实验使用训练部分拟合预处理器，是拒绝混入原始数据的更直接依据。

## 6. 第五轮：样条模型、外部先验与 fine-bin XGBoost

[Notebook 05](../../notebooks/05_model_rethinking_and_finebin_xgboost.ipynb) 汇总了样条逻辑模型最高 `0.939556`、原始数据训练先验 `0.932944`，都低于主 XGBoost。最关键的是 fine-bin 表：[结果](../../artifacts/ranking_model_probe_results.csv)

| `max_bin` | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---:|---:|---:|---:|---:|---:|
| Holdout AUC | `0.941832` | `0.942196` | `0.942520` | `0.942808` | `0.942993` | **`0.943136`** |

`256 → 8192` 增量约 `+0.001304`，是当前记录中最明显的参数方向。但[探针脚本](../../ranking_model_probe.py)将 `256–4096` 历史数值写在 `rows` 常量中，只训练 `8192`；这组趋势应按保存的实验记录使用，不应误写为“六个版本均在这次脚本运行中重新拟合”。后续公开榜截图显示 `4096` 双种子为 `0.94313`、`8192` 单种子为 `0.94311`，都高于第一版优化 XGBoost `0.94187`；公开榜增量同时包含特征和训练方案变化，不能完全归因于 `max_bin`。[Notebook 07](../../notebooks/07_rf_svm_and_top_model_blends.ipynb)

## 7. 第六轮：异构树与验证式权重

[异构结果](../../artifacts/diverse_model_comparison.csv)在统一后期划分上显示：`4096` depth-5 XGBoost `0.942993`、depth-3 XGBoost `0.942544`、CatBoost Lossguide `0.941687`、LightGBM GOSS `0.941111`、Extra Trees `0.935314`。概率融合按 tuning Log Loss 学到约 `98.90%` 主模型与 `1.06%` depth-3，holdout AUC `0.942992`；rank 融合回到 `100%` 主模型。[权重](../../artifacts/diverse_ensemble_weights.json)

因此不能写“异构集成提高了分数”。它检验了备选方向，却说明当次候选缺乏足够强的互补排序。若某个弱模型特别慢而且权重接近零，继续为它生成全量提交没有本地证据支持。[Notebook 06](../../notebooks/06_diverse_models_and_validated_ensembles.ipynb)

## 8. 第七轮：RF、XGBRF、线性 SVM 和两个 Public 最佳提交

[RF/SVM 结果](../../artifacts/rf_svm_model_comparison.csv)显示：sklearn RF `0.939658`、XGBRF `0.939390`、线性 SVM 最高 `0.937208`，均低于同轮 XGBoost `0.942993`。对每个模型搜索 `1%–30%` rank 权重，最佳都为 `0%`。[融合结果](../../artifacts/rf_svm_pairwise_ensembles.csv)

据 Public `0.94313` 与 `0.94311` 生成的 `75/25`、`90/10` 概率融合和 `75/25` rank 融合文件，只记录了文件创建和预测相关性，没有相应可核对的本地或公开榜 AUC。不能说这些提交已提升分数。[Notebook 07](../../notebooks/07_rf_svm_and_top_model_blends.ipynb)；[生成脚本](../../create_top_model_blends.py)

## 9. 第八轮：V2 延长训练与相近结构融合

[V2 结果](../../artifacts/main_xgboost_v2_results.csv)显示，`4096` depth-5 把训练上限从 1800 提高到 2200 后，最佳迭代为 2159，holdout AUC 从旧参考 `0.942993` 到 `0.943028`。低学习率 depth-5 `0.943003`；depth 4、depth 6、Lossguide 单模都更低。tuning 上学习的 rank 组合以 `62.3%/26.7%/11.0%` 组合三个相近结构，holdout `0.943040`，比本轮最好单模高约 `0.000012`，仍低于历史 `8192` 的 `0.943136`。[V2 权重](../../artifacts/main_xgboost_v2_summary.json)

[全量训练脚本](../../train_xgboost_v2_ensemble.py)生成五种子、结构 rank、与旧 `8192` 的 `80/20` 混合提交；仓库没有这些新文件的已验证 Public 分数，也没有五种子文件的本地 AUC。这些只能称为“生成的候选”，不能称为“已提升的最终模型”。[Notebook 08](../../notebooks/08_xgboost_v2_optimization_and_ensembles.ipynb)

## 10. 审计清单：本复盘没有写成已证实的事情

| 常见误读 | 实际记录 |
|---|---|
| “`Core_Context` 单独让 AUC 增加” | 没有只切换这一列的消融；fine-bin 阶段同时变了模型方案 |
| “所有 8192 训练都优于 4096 集成” | 本地单模型 `8192` 较高；用户给的 Public 是双种子 `4096` 略高 `0.00002`，配置不只差分箱 |
| “RF、SVM、Extra Trees 从未试过” | 都有实际实验；在记录的参数与切分下较弱，融合权重为零 |
| “RBF SVM、FT-Transformer、TabNet、`pytorch-tabular` 已训练” | 只在说明中被提及或建议，没有训练与分数记录 |
| “`XGBRanker` 已用于最终排名模型” | 探针脚本导入了 `XGBRanker`，但当前执行路径只训练 `XGBClassifier`；rank 融合是预测后处理 |
| “五种子 V2 已得到 `0.943040`” | `0.943040` 是单种子验证预测构成的 V2 结构融合；全量五种子提交没有对应本地或 Public 分数 |
| “holdout 在整个项目中只看过一次” | 后期多个迭代查看同一切分，存在选择偏差 |
| “样条模型九组参数经过独立 tuning 选择” | 九组直接在同一 holdout 上比较，最高值有选择偏差 |
| “外部原始数据绝对无用” | 当前权重、预处理和模型下明显有害；不代表一切外部数据方法都无效 |

## 11. 最终归纳

有直接同轮证据的改善是：整组基础特征、整组精细特征、提高 fine-bin 分辨率、以及小幅延长 depth-5 训练。异构弱模型的简单或优化融合没有超过主模型；原始数据追加训练显著伤害本地 AUC。最稳妥的解释是，这个少量强条件变量加连续变量的表格任务，适合能细致分裂并逐轮修正误差的树模型；具体因果机制和私榜优势仍需新的控制实验或最终榜单验证。
