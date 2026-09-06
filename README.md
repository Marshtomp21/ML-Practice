# XAI01-02：扰动类与博弈论类归因方法比较

> 机器学习综合实践 · XAI-T01「特征归因方法选择与评价」
> 在给定预训练模型与数据配置下，系统实现并比较**扰动类**与**博弈论类**归因方法，
> 围绕**忠实性、稳定性、效率**三个维度回答 RQ1--RQ3。

---

## 当前进度

仓库骨架已建立，各模块的实现将按里程碑逐步填充。

| 阶段 | 内容 | 状态 |
|---|---|---|
| M1 问题分析 | 需求分析与任务建模、方法选择、实验设计 | 进行中 |
| M2 初步结果 | 数据预处理、值函数与前向计数、LIME + Ablation | 未开始 |
| M3 中期结果 | KernelSHAP、评价指标、多方法初步比较 | 未开始 |
| M4 完整实验 | 粒度实验、消融链、梯度类交叉对比、选做与探索 | 未开始 |
| M5 最终成果 | 完整报告、答辩 PPT | 未开始 |

---

## 1. 项目概览

本项目不提出新的归因算法，也不复现前沿论文，而是回答一个方法选择问题：
**博弈论类方法（SHAP）的公理性保证，在实践中是否值得其高昂的计算成本？**

| 研究问题 | 内容 |
|---|---|
| **RQ1** | 扰动类与博弈论类方法在忠实性与效率上如何权衡？计算成本增加是否带来质量提升？ |
| **RQ2** | 超像素大小、扰动比例、采样数如何影响归因质量？什么粒度最优？ |
| **RQ3** | SHAP 的公理保证是否转化为更优的归因质量？与梯度类、扰动类相比如何？ |

**计划实现的方法**：LIME、RISE、Ablation（扰动类）；KernelSHAP、Shapley 置换采样、
Faith-Shap 二阶交互（博弈论类）；Saliency、Integrated Gradients、Expected Gradients、
Grad-CAM、Grad-CAM++（梯度类对照）。

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

---

## 5. 运行方式

实验统一通过命令行入口执行：

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
