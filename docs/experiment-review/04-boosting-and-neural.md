# 04｜提升树与神经网络：参数路线和实验结果

[返回目录](README.md) · [上一篇：经典模型](03-classical-models.md) · [下一篇：集成学习](05-ensembles.md)

## 参数先读懂

提升树是一棵棵追加的小决策树：新树着重修正现有模型的错误。`n_estimators`/`iterations` 是最多追加多少棵，`learning_rate` 是每棵树走多大一步；步长变小后往往需要更多树，因此这两个参数通常一起调。`max_depth`、`num_leaves`、`max_leaves` 决定单棵树能切出多复杂的群体；`min_child_weight`、`min_child_samples`、`min_data_in_leaf` 则防止对很小的群体过度拟合。`subsample` 和 `colsample_bytree` 让每棵树只看部分行或列，`reg_alpha`/`reg_lambda` 与 `l2_leaf_reg` 用正则限制模型过于激进。这些参数在各库中的数学定义并不完全相同，表格只能比较同一实验内的候选。[XGBoost 参数实现](../../optimize_main_xgboost_v2.py)；[LightGBM/CatBoost 参数实现](../../compare_diverse_ensembles.py)

早停指训练时持续检查 tuning AUC；若若干轮没有改善，就保留此前最好的迭代。它可以避免盲目使用树数上限，但 tuning 集也因此参与了模型选择。若最优迭代一直撞到上限，例如 `1799/1800`，可以合理地测试更高上限；这仍须看新的验证结果。[fine-bin 探针](../../ranking_model_probe.py)；[V2 优化](../../optimize_main_xgboost_v2.py)

## 1. 第一轮 GBDT 对比：XGBoost 略胜，差距很小

Notebook `03` 在相同的最多 14 万训练样本、4 万 tuning 和 20% holdout 上，各为 CatBoost、LightGBM、XGBoost 试了两个候选。候选在 tuning 上选出，holdout 只报告选中模型的分数。[Notebook 03，代码单元 14–16、24](../../notebooks/03_advanced_models_and_deep_learning.ipynb)

| 家族 | tuning 候选及主要参数 | tuning 选择 | 选中模型 holdout AUC |
|---|---|---|---:|
| XGBoost | depth 5：`800` 树、`lr=0.03`、`min_child_weight=5`、`reg_lambda=2`；depth 7：`600` 树、`lr=0.04`、`min_child_weight=10`、`reg_lambda=5`；均 `subsample=colsample_bytree=0.85` | depth 5：`0.940649` > depth 7：`0.940528` | **`0.940255`** |
| CatBoost | depth 6：`600` 轮、`lr=0.05`、`l2_leaf_reg=5`；depth 8：`500` 轮、`lr=0.05`、`l2_leaf_reg=8` | depth 6：`0.940404` > depth 8：`0.940109` | `0.940118` |
| LightGBM GBDT | 31 leaves：`800` 树、`lr=0.03`、`min_child_samples=50`、`reg_lambda=2`；63 leaves：`600` 树、`lr=0.04`、`min_child_samples=80`、`reg_lambda=5` | 31 leaves：`0.940595` > 63 leaves：`0.940390` | `0.940113` |

三家都让容量较受控的候选胜出：depth 5 胜 depth 7，depth 6 胜 depth 8，31 leaves 胜 63 leaves。因为候选同时改变了树数、学习率与正则，不能只说“较浅必然更好”；更准确的结论是**这一整组较保守设定**在 tuning 上较好。

## 2. XGBoost：从精细特征到高分辨率直方图

### 2.1 扩大训练数据并优化特征

后期 `optimize_xgboost_leaderboard.py` 将训练部分扩大到约 46.8 万行。基础 XGBoost 为 `n_estimators=1400`、`learning_rate=0.025`、`max_depth=5`、`min_child_weight=5`、`subsample=0.85`、`colsample_bytree=0.85`、`reg_lambda=2`、`tree_method="hist"`、`max_bin=256`。它在固定切分下比较旧特征与精细特征，holdout AUC `0.941714 → 0.941855`。与 Notebook `03` 的 `0.940255` 相比，变化同时包含训练量和特征，不能做单因素归因。[脚本](../../optimize_xgboost_leaderboard.py)；[结果](../../artifacts/xgboost_optimization_results.csv)

同一阶段的参数族搜索如下；数值为 tuning AUC，只有最终选中方案及外部数据权重实验完整报告了 holdout：

| 候选 | 与 depth-5 基准相比改变了什么 | Tuning AUC | 结论 |
|---|---|---:|---|
| `depth5_reference` | 基准：`1400` 树，`lr=0.025`，depth 5 | **`0.941753`** | 该组最好；最优迭代 `1377`（全量重训用 `1378` 棵） |
| `lossguide_31` | depth 0、`max_leaves=31`、Lossguide、`1600` 树、`lr=0.02`、更强正则 | `0.941687` | 没超过基准 |
| `depth4_regularized` | depth 4、`1800` 树、`lr=0.02`、`min_child_weight=10`、更强正则 | `0.941675` | 没超过基准 |
| `depth6_regularized` | depth 6、`min_child_weight=10`、`gamma=0.02`、更强正则 | `0.941594` | 该组最低 |

这些结果支持继续用 depth-5 方案，但不能孤立判定 Lossguide 或 depth 6 一定无效，因为每一行同时改变了多个参数。用户随后提供的公开榜截图中，第一版优化 XGBoost 三种子平均达到 `0.94187`，高于之前 XGBoost 的 `0.94154`；这是**整套训练、特征、种子方案**的公开榜提升，不是某一个参数的效果。[元数据](../../artifacts/xgboost_optimization_metadata.json)；[Notebook 04](../../notebooks/04_leaderboard_optimization.ipynb)

### 2.2 `max_bin`：最明显的参数方向

在 XGBoost `hist` 算法中，连续数值先被映射到有限数量的候选箱，树再寻找分裂点。`max_bin` 较小更省资源，但可能把收入、通勤等变量里有区分力的细节合在同一箱。后期记录的同一模型族结果为：[结果表](../../artifacts/ranking_model_probe_results.csv)

| `max_bin` | 最优迭代 | Holdout AUC | 相对 256 | 训练秒数 |
|---:|---:|---:|---:|---:|
| 256 | 1511 | `0.941832` | 基准 | 60.7 |
| 512 | 1799 | `0.942196` | `+0.000364` | 73.7 |
| 1024 | 1789 | `0.942520` | `+0.000688` | 77.3 |
| 2048 | 1791 | `0.942808` | `+0.000976` | 74.5 |
| 4096 | 1799 | `0.942993` | `+0.001161` | 93.6 |
| 8192 | 1799 | **`0.943136`** | **`+0.001304`** | 314.3 |

这里 `n_estimators=1800`、`lr=0.025`、depth 5、`min_child_weight=5`、采样率各 `0.85`、`reg_lambda=2`。`8192` 相比 `4096` 只再增加约 `0.000142` AUC，却在该记录中花费约 `3.4` 倍训练时间。成绩上升与“更细切点改善连续变量排序”的解释相符，但没有逐变量消融证明具体由哪些字段贡献；而且 `256–4096` 是探针脚本中记录的历史结果行，`8192` 才由该脚本现场训练。[探针脚本](../../ranking_model_probe.py)

公开榜截图显示 `4096` 双种子集成 `0.94313`，`8192` 单种子 `0.94311`。这两个提交还改变了种子数量，无法用公开榜 `0.00002` 的差距单独比较 `max_bin`；私榜表现未知。[Notebook 07](../../notebooks/07_rf_svm_and_top_model_blends.ipynb)

### 2.3 V2：延长训练与结构变体

V2 固定 `max_bin=4096`，以 depth 5、`lr=0.025`、`min_child_weight=5`、采样率各 `0.85`、`reg_lambda=2` 为基准，把最大树数提高到 `2200`，并测试更低学习率、depth 4、depth 6 与 Lossguide。每组按 tuning AUC 早停，holdout 结果如下。[参数脚本](../../optimize_main_xgboost_v2.py)；[结果](../../artifacts/main_xgboost_v2_results.csv)

| V2 候选 | 重要参数差异 | 最优迭代 | Holdout AUC |
|---|---|---:|---:|
| `Depth5_Baseline` | `2200` 树上限、`lr=0.025`、depth 5 | 2159 | **`0.943028`** |
| `Depth5_Long_LowLR` | `3000` 树、`lr=0.018`、采样 `0.88`、`reg_alpha=0.02`、`reg_lambda=3` | 2986 | `0.943003` |
| `Depth4_Long` | `3200` 树、`lr=0.018`、depth 4、`min_child_weight=8`、`reg_lambda=4` | 3199 | `0.942910` |
| `Lossguide_31` | `3000` 树、`lr=0.018`、`max_leaves=31`、`min_child_weight=8`、`reg_lambda=4` | 2313 | `0.942891` |
| `Depth6_Regularized` | `2200` 树、`lr=0.020`、depth 6、`min_child_weight=12`、`gamma=0.01`、`reg_lambda=5` | 2172 | `0.942839` |

此前 `4096`、1800 树版本在相同后期切分上为 `0.942993`。V2 depth-5 为 `0.943028`，增量只有 `0.000035`；原版本最优迭代撞到 `1799` 上限，V2 达 `2159`，与“原训练轮数稍少”一致。小幅改善不能保证超过 `8192`：后者仍以 `0.943136` 保持本地最高。其他变体有多个参数同时变化，所以不能把得失只归于深度或学习率。[旧结果](../../artifacts/ranking_model_probe_results.csv)；[V2 结果](../../artifacts/main_xgboost_v2_results.csv)

## 3. LightGBM：标准 GBDT 接近，但 GOSS 版本未追上

第一轮标准 `LGBMClassifier` 的 31 leaves 候选 tuning AUC 为 `0.940595`，优于 63 leaves 的 `0.940390`；选中方案 holdout 为 `0.940113`。后期为测试不同采样机制，GOSS 方案使用 `n_estimators=3000`、`lr=0.02`、`num_leaves=31`、`min_child_samples=100`、`max_bin=1023`、`data_sample_strategy="goss"`、`top_rate=0.20`、`other_rate=0.10`、`reg_lambda=4`，holdout 为 `0.941111`。由于训练量、特征、切分、正则和采样机制同时改变，`0.941111` 高于早期 `0.940113` **不证明 GOSS 比普通 GBDT 好**；在同轮比较中，它仍落后于 `4096` XGBoost 的 `0.942993`。[Notebook 03](../../notebooks/03_advanced_models_and_deep_learning.ipynb)；[异构实现](../../compare_diverse_ensembles.py)；[结果](../../artifacts/diverse_model_comparison.csv)

尝试 GOSS 的理由是让训练更关注当前预测困难的样本，同时保留部分其余样本，以减少计算；但“更快”不能补偿这里 `0.001882` 的同轮 AUC 差距。31 leaves 胜 63 leaves 只说明这两组完整设定的调优集排序；不是所有较少叶子的 LightGBM 都会更好。

## 4. CatBoost：常规对称树较强，Lossguide 实验没有超越主模型

早期用原生类别列训练，比较 depth 6（`600` 轮、`lr=0.05`、`l2_leaf_reg=5`）与 depth 8（`500` 轮、`lr=0.05`、`l2_leaf_reg=8`），按 tuning 选中 depth 6；holdout AUC `0.940118`。用户提供的公开榜截图中，CatBoost 单模型约 `0.94095`。[Notebook 03](../../notebooks/03_advanced_models_and_deep_learning.ipynb)

后期 Lossguide 候选设 `iterations=2200`、`learning_rate=0.03`、`max_leaves=31`、`min_data_in_leaf=100`、`l2_leaf_reg=5`、`random_strength=0.5`、Bernoulli `subsample=0.85`，按 tuning 早停；holdout AUC `0.941687`，仍低于同轮 XGBoost `0.942993`。与早期 CatBoost 的数值不可直接归因于 grow policy，因为训练规模和特征均改变。它在异构融合中几乎没有获得权重。[异构实现](../../compare_diverse_ensembles.py)；[结果](../../artifacts/diverse_model_comparison.csv)；[权重](../../artifacts/diverse_ensemble_weights.json)

CatBoost 在代码里直接接收原生类别字段，这是选择它的实际理由：无需先把全部类别手工 one-hot。早期 depth 8 候选更复杂却略差，提示该配置可能没有带来有益的条件细节；Lossguide 则改变树的生长方式，希望形成不同的局部划分。两个实验都只覆盖有限参数，不能把失败推广为“CatBoost 不适合这个主题”。

## 5. PyTorch 类别嵌入网络：一次具体架构实验

Notebook `03` 为类别字段分别建立 embedding，维度根据类别数量开平方后限制在 `2–16`；数值列先填补并标准化。网络为 `128 → 64 → 1` 的 MLP，含 BatchNorm、ReLU 和 `0.20/0.10` Dropout；`AdamW(lr=0.001, weight_decay=0.0001)`、batch size `2048`、最多 `12` epoch、tuning AUC 耐心 `3` 个 epoch。第 `7` 轮的 tuning AUC 最高为 `0.938154`，holdout 为 `0.938041`；用户提供的公开榜截图约 `0.93805`。[Notebook 03，代码单元 19–24](../../notebooks/03_advanced_models_and_deep_learning.ipynb)

它低于三种提升树，也使四模型等权融合从三 GBDT 的 `0.940314` 降到 `0.940112`。**只做过这套架构、一次随机种子和 early stopping**；没有记录学习率网格、embedding 维度搜索、多种子神经网络训练，也没有运行 FT-Transformer、TabNet 或 `pytorch-tabular`。较弱结果可能与样本量、类别基数、结构偏置或调参预算有关，不能据此断言深度学习在所有同类表格任务上都不合适。[Notebook 03](../../notebooks/03_advanced_models_and_deep_learning.ipynb)

embedding 的动机是让类别拥有可训练的低维表示，MLP 再组合这些表示与标准化数值输入；Dropout 和权重衰减用于限制过拟合。当前网络在第 7 轮后 tuning AUC 不再改善，早停避免继续消耗训练时间。相比能直接对收入与条件变量寻找局部切点的提升树，这一套网络没有取得更好的排序；究竟是模型表达方式还是训练预算造成差距，当前实验无法拆分。
