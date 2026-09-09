# XAI01-02：扰动类与博弈论类归因方法比较

> 机器学习综合实践 · XAI-T01「特征归因方法选择与评价」
> 在给定预训练模型与数据配置下，系统实现并比较**扰动类**与**博弈论类**归因方法，
> 围绕**忠实性、稳定性、效率**三个维度回答 RQ1--RQ3。

---

## 当前进度

仓库骨架已建立，LIME 基线算法及合成数值测试已实现；尚未运行真实数据归因实验。

| 阶段 | 内容 | 状态 |
|---|---|---|
| M1 问题分析 | 需求分析与任务建模、方法选择、实验设计 | 进行中 |
| M2 初步结果 | 冻结数据与模型输入，完成值函数、预算计数、Ablation、LIME 及小规模端到端验证 | 进行中：LIME 核心已实现，其余待完成 |
| M3 中期结果 | 完成 KernelSHAP、统一评价指标及默认设置下的核心方法初步比较 | 未开始 |
| M4 完整实验 | 完成核心全量实验、参数敏感性、梯度对照、扩展探索与配对统计 | 未开始 |
| M5 最终成果 | 冻结实验结果，完成可追溯的统计图表、完整报告与答辩 PPT | 未开始 |

---

## 1. 项目概览

本项目不提出新的归因算法，也不复现前沿论文，而是回答一个方法选择问题：
**博弈论类方法（SHAP）的公理性保证，在实践中是否值得其高昂的计算成本？**

| 研究问题 | 内容 |
|---|---|
| **RQ1** | 扰动类与博弈论类方法在忠实性与效率上如何权衡？计算成本增加是否带来质量提升？ |
| **RQ2** | 超像素大小、遮蔽率、模型前向预算如何影响归因质量？什么粒度最优？ |
| **RQ3** | SHAP 的公理保证是否转化为更优的归因质量？与梯度类、扰动类相比如何？ |

**方法范围**：核心方法包括 Ablation、LIME、KernelSHAP、Integrated Gradients 和
Grad-CAM，用于形成主要实验结论；RISE、Shapley 置换采样、Faith-Shap 二阶交互、
Saliency、Expected Gradients 和 Grad-CAM++ 作为扩展方法，只报告探索性结果。

**数据与模型**：ImageNet 验证集子集（自然图像）与 CHNCXR 胸片（医学影像）
均使用 ResNet-50、VGG13，形成 2 数据域 $\times$ 2 模型的对照设计。

---

## 2. 目录结构

```
.
├── configs/              配置层：实验参数、方法超参、评价口径
│   ├── data/             数据集定义与划分策略
│   ├── model/            模型口径、预处理、解释目标
│   ├── method/           各归因方法的参数与扫描网格
│   └── experiment/       各项实验的设计
├── data/                 数据说明与划分清单（原始影像不入库）
│   ├── raw/              原始数据存放处
│   ├── processed/        预处理产物
│   ├── splits/           划分清单与数据指纹
│   └── external/         外部或跨组产物
├── preprocessing/        预处理、超像素分割、掩码生成、遮蔽基线
├── models/               模型加载与指纹、值函数与前向计数
├── attribution/          归因方法实现
│   ├── perturbation/     LIME / RISE / Ablation
│   ├── game_theoretic/   KernelSHAP / 置换采样 / Faith-Shap 交互
│   ├── gradient/         对照组：Saliency / IG / Grad-CAM 系列
│   └── hybrid/           混合策略
├── evaluation/           忠实性、稳定性、效率、定位性、统计推断
├── experiments/          各实验入口与统一驱动
├── results/              指标 JSONL、聚合表、图表、归因图
├── reports/              LaTeX 报告
├── scripts/              数据获取与校验、图表生成
├── tests/                值函数、可复现性、公理检查、指标、预算对齐
└── utils/                种子派生、配置加载、计时、结果读写
```

---

## 3. 环境配置

### Conda（推荐）

```bash
conda env create -f environment.yml
conda activate xai01-02
```

### pip

```bash
# 先按 https://pytorch.org 的官方指令安装对应 CUDA 版本的 torch，再执行：
pip install -r requirements.txt
```

开发与交叉验证依赖（可选）：`pip install -r requirements-dev.txt`

**硬件**：单卡 NVIDIA GPU，建议显存 ≥ 12 GB。

---

## 4. 数据

原始影像与模型权重**不入 Git**，仓库只提交 `data/splits/` 下的划分清单与数据指纹；
成员各自获取数据后运行预处理脚本即可重建一致的实验输入。

| 数据集 | 用途 | 存放位置 |
|---|---|---|
| ILSVRC2012 验证集子集 | 自然图像主实验集 | `data/raw/imagenet/` |
| CHNCXR (Shenzhen CXR Set) | 医学影像实验集 | `data/raw/chncxr/` |

数据由课程统一提供；CHNCXR 亦可从 NLM 官方源获取：
<https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/>

M1 固化的数据使用口径如下：

- ImageNet 以当前 500 张图像为候选池，CHNCXR 以全部 662 张胸片为候选池；
- 各数据域的主实验集取 ResNet-50 与 VGG13 均分类正确的样本交集，并将样本 ID 固化到 `data/splits/`；
- 解释目标统一为正确类别的预 Softmax logit；在主实验集中，正确类别同时也是两个模型的预测类别；
- CHNCXR 定位性评价仅使用具有病灶掩码的阳性样本，负样本仍参与忠实性、稳定性和效率评价；
- 前届提供的 51 张全阳性预测样例仅作为独立参考队列，不替代从 662 张候选池筛选的主实验集；
- 课程资源中的 VOC 数据不属于当前 M1 与后续主实验范围。

### M1 方法与评价口径

- 所有方法解释正确类别的预 Softmax logit；区域方法共享同一 SLIC 划分，并以归一化输入空间中的零张量作为主遮蔽基线；
- 主设置采用 100 个名义超像素、0.50 遮蔽率和 1024 次模型前向总预算；敏感性实验依次考察超像素数 `{50, 100, 200, 300}`、遮蔽率 `{0.25, 0.50, 0.75}` 和模型前向预算 `{256, 512, 1024, 2048}`；RISE 使用保留概率 `p_keep = 1 - 遮蔽率`；
- 随机方法统一使用种子 `{0, 1, 2, 3, 4}`，按逐样本方式保存结果和开展配对分析；
- 忠实性按超像素内归因总和排序并报告删除/插入 AUC；稳定性报告区域排序 Spearman 相关和 Top 20% Jaccard；确定性方法不参与随机稳定性的显著性比较；效率按实际输入样本数统计模型前向/反向次数，并报告归因耗时；
- 定位性报告正归因的掩码内能量占比与 Pointing Game；ImageNet 使用 bbox 区域，CHNCXR 仅使用具有病灶掩码的阳性样本；
- 四个“数据域 × 模型”组合分别分析；默认主设置以配对差值中位数为主要估计量，使用配对 bootstrap 置信区间、双侧 Wilcoxon 检验、秩二列相关效应量和 Holm 多重比较校正；RQ2 扫描不重复进行显著性检验，扩展方法只作探索性报告。

---

## 5. 运行方式

### 已实现：LIME 基线代码（暂不执行真实数据归因与可视化）

从 `code/` 目录运行以下命令，仅检查默认参数与单图计算预算：

```bash
python -m experiments.lime_baseline --config configs/method/lime.yaml --dry-run
python -m pytest tests/test_lime.py -q -p no:cacheprovider
```

配置入口不会加载权重、读取影像或保存归因结果。数值测试仅使用合成输入和
可解析的预测函数。该部分依赖 NumPy、scikit-learn、scikit-image、PyYAML
和 pytest；无需 PyTorch。模型加载与真实数据实验尚待接入。

实现位置：

- `attribution/perturbation/lime.py`：独立随机采样、余弦距离核、加权岭回归、区域系数及拟合诊断；
- `preprocessing/superpixels.py`：统一 SLIC 分割，默认名义区域数 100、compactness=10、sigma=1；
- `configs/method/lime.yaml`：1024 次前向总预算、批大小 32、遮蔽率 0.5、核宽 0.25、岭惩罚 1.0、种子 0；
- `tests/test_lime.py`：数值正确性、预算、可复现性与接口检查。

后续模型模块通过 `fit_lime(image, segments, predict_logits, target, config)` 接入：

- `image` 为已按权重规范裁剪/缩放并归一化的浮点 `H×W×C` 数组；
- `segments` 为与输入严格对齐的整数 `H×W` 标签，可使用非连续标签；
- SLIC 在归一化之前的 `[0, 1]` RGB 图像上执行，使用实际产生的区域数；
- `predict_logits` 接收归一化后的 NumPy `N×H×W×C` 批次，返回 `N×K` 原始 logit；PyTorch 接入方负责转为 NCHW、设备转换、`eval()` 和关闭梯度；
- `target` 显式传入正确类别的零起始索引，函数不会额外推理选择类别。

每个遮蔽区域在归一化输入空间置零。预算包含首行完整输入；其余行独立按
Bernoulli 分布采样，因此遮蔽率是期望的区域比例，不保证每行恰好遮蔽同样数量，
也不是像素面积比例。重复掩码仍计入预算，不做缓存或额外端点推理。

算法采用全部区域拟合（`feature_selection='none'`），截距不正则化，目标函数为
`sum(w * (y - intercept - Z @ coefficients)^2) + alpha * ||coefficients||²`。
核权重为 `exp(-0.5 * (cosine_distance / kernel_width)^2)`。这些数学约定参考
[LIME 官方图像实现](https://github.com/marcotcr/lime/blob/master/lime/lime_image.py)及
[局部回归实现](https://github.com/marcotcr/lime/blob/master/lime/lime_base.py)；本项目改用共享
SLIC、logit 和归一化零遮蔽，不保证与原库默认配置或随机数序列逐位一致。

返回值包含按 `region_ids` 对齐的有符号系数、截距、原图 logit、代理模型在原图的
预测、训练邻域加权 R²、实际前向样本数/批次数、实际遮蔽率和不同掩码数。
加权 R² 仅诊断局部拟合，不替代忠实性评价；区域系数未投影为像素归因图，
未归一化或截断正负值。采样预算小或区域缺少变化时，岭回归仍可拟合，但不能据此
认定局部效应可靠。耗时、Insertion/Deletion、稳定性评价与真实模型验证留待实验阶段。

### 后续实验入口约定

以下为后续模块的入口规范，尚未全部实现：

```bash
python -m experiments.<name> --config configs/experiment/<name>.yaml [--dry-run]
```

`--dry-run` 只打印任务规模与计算预算估计，不实际运行。

可复现性约定：固定随机种子且各随机源独立派生；每次运行冻结完整配置并随结果落盘；
每条指标记录携带 git commit、模型权重与数据指纹、以及运行环境元数据；
所有图表由脚本从落盘表格重新生成，不手工绘图。

---

## 6. 报告

LaTeX 报告位于 `reports/`，使用 XeLaTeX 编译（中文排版依赖 `ctex`）：

```bash
cd reports/requirements_modeling && latexmk -xelatex main.tex
```

---

## 7. 团队与协作

五人小组，按数据、模型、实验、分析、文档分工，成员之间交叉互审。

| 姓名 | 角色 | 主要职责 | 主要目录 | 互审 |
|---|---|---|---|---|
| 杜孟泽 | R1 数据 | 数据获取与合规、划分清单、预处理、超像素与掩码模块、EDA | `data/` `preprocessing/` | 审 R2 |
| 张泰瑜 | R2 模型 | 模型加载与指纹、值函数与前向计数、缓存与批处理 | `models/` | 审 R3 |
| 邹研泽 | R3 方法 | LIME / RISE / Ablation / KernelSHAP / 交互分解 / 梯度基线 | `attribution/` | 审 R4 |
| 陈纪仰(组长) | R4 实验 | 评价指标、实验驱动、预算对齐、断点续跑、结果落盘 | `evaluation/` `experiments/` | 审 R5 |
| 朱炳政 | R5 分析与文档 | 统计推断、图表生成、报告与 PPT | `results/` `reports/` | 审 R1 |

组长兼任进度管理与对外沟通。

**分支**：`main` 保持随时可运行；功能分支 `feat/<模块>-<简述>`、
修复分支 `fix/<简述>`、实验分支 `exp/<exp_id>-<简述>`。

**提交信息**：写清改了什么、为什么，避免只写 `update` / `fix`。

```
feat(attribution): 实现 KernelSHAP 的成对联盟采样，方差较独立采样下降约 40%
fix(evaluation): Insertion 起点改为模糊图，修正全零起点导致的曲线前段噪声
```

**评审**：合入 `main` 需至少一名交叉互审人 approve。

**在线协作文档**：维护分工表、会议纪要与进度追踪。

---

## 8. 引用

- Ribeiro et al. *"Why Should I Trust You?": Explaining the Predictions of Any Classifier.* KDD 2016. (LIME)
- Petsiuk et al. *RISE: Randomized Input Sampling for Explanation of Black-box Models.* BMVC 2018.
- Lundberg & Lee. *A Unified Approach to Interpreting Model Predictions.* NeurIPS 2017. (SHAP/KernelSHAP)
- Sundararajan et al. *Axiomatic Attribution for Deep Networks.* ICML 2017. (Integrated Gradients)
- Selvaraju et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017.
- Tsai et al. *Faith-Shap: The Faithful Shapley Interaction Index.* JMLR 2023.
- Covert et al. *Explaining by Removing: A Unified Framework for Model Explanation.* JMLR 2021.
- Jaeger et al. *Two public chest X-ray datasets for computer-aided screening of pulmonary diseases.* QIMS 2014. (CHNCXR)
