# 02｜特征工程：做了什么、证据支持什么

[返回目录](README.md) · [上一篇：证据与验证](01-evidence-and-validation.md) · [下一篇：经典模型](03-classical-models.md)

## 1. 从原始数据理解任务

训练 CSV 的原始预测字段只有 13 个。探索结果中，`Environmental_Concern_Level` 与目标的线性相关约 `0.464`，`Annual_Income_USD` 约 `0.226`；有补贴组正类率约 `27.47%`，无补贴组约 `0.58%`；低里程焦虑组约 `18.90%`，高焦虑组约 `0.14%`。这些是**单变量关联**，说明这些字段值得优先建模，却不证明补贴、焦虑或环保态度对购买意愿的因果作用。[Notebook 01](../../notebooks/01_data_exploration.ipynb)

`id` 被排除：它是训练/测试样本的标识与提交对齐依据，没有可解释的购买行为含义。`Will_Buy_EV` 只用于训练标签，绝不能进入特征。类别字段在经典模型和大部分树模型中做 one-hot，未知类别采用忽略策略；数值缺失用训练部分中位数填补，类别缺失用训练部分众数填补。早期线性模型额外标准化数值字段。这样保证预处理器只从训练折学习参数。[Notebook 02](../../notebooks/02_feature_engineering_model_comparison.ipynb)

## 2. 第一轮：把单行信息组合成可学习的特征

Notebook `02` 的 `EVFeatureEngineer` 把 13 个原始预测字段扩展为 32 个建模字段，增加 19 列。核心做法如下。[Notebook 02，代码单元 11](../../notebooks/02_feature_engineering_model_comparison.ipynb)

| 特征组 | 代表列或公式 | 为什么尝试 | 需要注意 |
|---|---|---|---|
| 充电供给 | `Total_Charging_Stations = Home + Work`；`Min_Charging_Stations`、`Max_Charging_Stations`、`Home_Work_Station_Gap` | 总量和最薄弱场景可能比单一地点的站数更贴近日常便利程度 | 家庭与工作地点并非完全可替代；这些变量之间高度相关 |
| 充电负担 | `Commute_Per_Station = Daily_Commute_km / (Total_Charging_Stations + 1)`；`Has_Nearby_Charging` | 把通勤需求与可用充电资源放在同一尺度比较 | `+1` 是避免除零的设计选择，不是物理单位换算 |
| 购买能力 | `Log_Annual_Income = log1p(Annual_Income_USD)`；`Income_Per_Car`；`Income_Per_Commute` | 收入通常长尾，相对收入可能比绝对值更能表达负担 | 分母为零时采用截断，避免无穷大；比例值仍需检查异常 |
| 行为交互 | `Environment_Income_Interaction`、`Environment_Charging_Interaction`、`Age_Income_Interaction` | 同样的收入或充电资源，对环保关注程度不同的人可能有不同意义 | 对树模型并非必需；对线性模型尤其有助于表达非加性关系 |
| 语义编码 | `Home_Charging_Flag`、`Subsidy_Flag`、`Range_Anxiety_Ordinal` | 给 Yes/No 或 Low/Medium/High 一个有序、可交互的表示 | 焦虑级别间距被编码为相等，这是一种近似 |
| 固定分箱与组合类别 | `Age_Band`、`Income_Band`、`Commute_Band`、`Charging_Context` | 让线性模型直接学习区间效应和“家庭充电 × 补贴”的条件分段 | 切点是预设规则；无法单凭整组消融判断每个分箱是否有益 |

**直接证据**：固定同一逻辑回归、同一训练样本和 20% 验证集，只切换工程特征开关，ROC-AUC 从 `0.937955` 增至 `0.938371`（`+0.000416`）；Log Loss 从 `0.233600` 降至 `0.232859`。这是整组特征的收益，不能把 `+0.000416` 分配给某一列。[Notebook 02，代码单元 21 的保存输出](../../notebooks/02_feature_engineering_model_comparison.ipynb)

## 3. 第二轮：面向强条件关系的精细特征

`optimize_xgboost_leaderboard.py` 在已有特征上加入 `refined=True` 的一组特征：

| 设计 | 实际字段 | 模型表达上的作用 |
|---|---|---|
| 非线性数值 | `Age_Squared`、`Income_Squared_Scaled`、`Log_Daily_Commute` | 让模型更容易区分年龄、收入、通勤的非线性区间 |
| 充电平衡 | `Charging_Balance_Ratio = (Home + 1)/(Work + 1)` | 区分“总量相同，但家庭/工作地点分布不同”的用户 |
| 收入 × 条件 | `Income_x_Subsidy`、`Income_x_Home_Charging` | 检验收入排序是否随补贴和家庭充电条件改变 |
| 环境关注 × 条件 | `Concern_x_Subsidy`、`Concern_x_Home_Charging` | 检验行为偏好是否依赖购买支持条件 |
| 焦虑 × 需求/供给 | `Anxiety_x_Commute`、`Anxiety_x_Charging` | 检验焦虑在长通勤或低充电可达性下的不同影响 |
| 相对心理强度 | `Concern_Minus_Anxiety` | 用一个差值概括推动与阻碍因素的相对强弱 |
| 组合类别 | `Subsidy_Range_Context`、`Home_Range_Context`、`City_Charging_Context`、`Concern_Level_Context` | 为类别组合建立明确分段，便于 one-hot 后的树学习 |

同一 XGBoost 基础设置与后期划分下，旧特征的 tuning/holdout AUC 为 `0.941590/0.941714`，精细特征为 `0.941753/0.941855`；两个切分方向一致，holdout 增量为 `+0.000141`。Log Loss 同时下降。但这一轮加入的是**整组**变量，没有逐列消融；不能宣称 `Income_x_Subsidy` 或任何一个组合类别单独提升了分数。[实现](../../optimize_xgboost_leaderboard.py)；[结果](../../artifacts/xgboost_optimization_results.csv)

## 4. 第三轮：为高分辨率 XGBoost 简化成核心上下文

`ranking_model_probe.py` 和随后的 fine-bin、异构模型、V2 脚本沿用充电总量/极值/差值、收入与通勤比例、环境交互以及上述主要数值交互，另加入：

```text
Core_Context = Environmental_Concern_Level
             + Subsidy_Available
             + Range_Anxiety_Level
             + Home_Charging_Possible
```

这是一列字符串类别，随后做 one-hot。它的意图是让模型直接识别“环保关注 × 补贴 × 焦虑 × 家庭充电”的核心条件组合，而不是依赖多次树分裂才能形成这个群体。[fine-bin 特征实现](../../ranking_model_probe.py)；[V2 特征实现](../../optimize_main_xgboost_v2.py)

**证据边界**：没有保持其他条件不变、只切换 `Core_Context` 的单独消融。fine-bin 阶段的高分不能归因于这一列；可能同时来自特征集合、`max_bin`、训练轮数和模型实现。复盘只把它写作建模思路，不把它写成经证实的单列贡献。

## 5. 哪些特征调整可能变差

仓库没有“某个具体新特征使 AUC 下降”的逐列结果表，因此不能编造坏特征排名。可以确认的是：

- 人工交互增加后收益仍较小（精细特征组仅 `+0.000141`），继续无约束叠加交叉项没有现成证据支持。[精细特征结果](../../artifacts/xgboost_optimization_results.csv)
- `Core_Context` 等类别组合会增加 one-hot 维度；稀少组合可能导致估计不稳，这是方法上的潜在风险，当前结果没有单独测出损失。
- 高分辨率树模型可以自己寻找复杂阈值，所以“逻辑回归受益的特征”不能自动推断为“XGBoost 也同幅受益”。[早期消融](../../notebooks/02_feature_engineering_model_comparison.ipynb)；[fine-bin 结果](../../artifacts/ranking_model_probe_results.csv)
- 外部原始数据并不是特征工程。它作为追加训练数据时显著降低 AUC；详见[逐轮账本](06-iteration-audit.md#5-外部原始数据为什么被排除)。

## 6. 可复用的特征工程原则

先根据数据理解提出“需求、供给、能力、心理”相关假设，再按特征组做同条件消融。比例特征处理除零，类别组合留意稀有取值，所有统计变换只在训练折拟合。若某组特征与参数同时改变，应把结果记为新方案整体效果，不能拆成虚假的单因素收益。
