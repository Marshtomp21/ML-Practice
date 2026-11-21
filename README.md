# XAI01-02：扰动类与博弈论类归因方法比较

机器学习综合实践 · XAI-T01「特征归因方法选择与评价」。项目比较扰动类与博弈论类特征归因方法的忠实性、稳定性、定位性与效率，并与梯度类外部结果进行对照。

## 当前进度

核心方法（Ablation、LIME、RISE、KernelSHAP）与扩展方法（OrdShap、Region-RISE）均已实现并通过单元测试。M1–M3 已完成需求建模、端到端验证、默认设置比较及中期汇报；M4 已完成冻结划分下的参数扫描（实验 1–3）、最终评价集确认性检验与跨域迁移对比（选做实验 7）、5 种遮蔽基线敏感性分析、OrdShap 4×4 位置归因（选做实验 5）、CHNCXR 定位诊断及 ViT-B/16 小规模跨架构/划分对照，并接入 XAI01-01 梯度类组对比材料；现阶段进入对比分析与总结。

| 阶段 | 内容 | 状态 |
|---|---|---|
| M1 | 需求、数据和实验设计 | 已完成 |
| M2 | 三个扰动基线与端到端验证 | 已完成 |
| M3 | 博弈论类方法、默认比较与中期汇报 | 已完成 |
| M4 | 参数扫描、最终评价、基线敏感性、OrdShap 与 ViT 扩展 | 主实验已完成 |
| M5 | 结题报告与答辩 | 待开展 |

## 项目设置

- 核心方法：LIME、RISE、Ablation、KernelSHAP；
- 扩展与对照方法：OrdShap（4×4 图块位置归因）、Region-RISE（SLIC/Patch 划分对照），以及外部梯度类对照（XAI01-01 组：Saliency、IG-blur、EG、Grad-CAM）；
- 数据：ImageNet 验证子集（500 候选 → 374 共同正确：M4 调参集 75 / 最终评价集 299）与 CHNCXR 深圳胸片（662 候选 → M4 调参集 86 / 最终评价集 339，其中最终集含病灶并集掩码的阳性图 198 张）；
- 模型：ResNet-50、VGG13（主实验），ViT-B/16（ImageNet 跨架构与划分对照）；
- 目标：正确类别的 pre-Softmax logit；主协议遮蔽基线为归一化空间零张量（M4 基线敏感性另对照 4 种替代基线）；
- 指标：Insertion/Deletion AUC（含跨粒度去混淆的统一 SLIC=100 重算）、Max-Sensitivity、Cosine Similarity、Spearman、Top 20% Jaccard、耗时、前向次数、掩码内正能量占比和 Pointing Game；
- CHNCXR 定位性评价只使用具有病灶并集掩码的阳性样本，其他指标使用全部共同正确样本。

主设置为 SLIC 名义区域数 100、遮蔽率 0.50、前向预算 1024、随机种子 `{0,1,2,3,4}`；M4 冻结划分见 `data/splits/m4_frozen_20260917.json`。

## 目录

```text
configs/          实验与方法配置
data/             数据说明与固定划分
docs/             跨组对比与项目分析文档
preprocessing/    图像预处理、超像素与 Patch 划分
models/           模型加载与推理计数
attribution/      归因方法（扰动类、博弈论类与遮蔽基线）
evaluation/       忠实性、稳定性与定位性评价指标
experiments/      实验入口
results/          实验指标、统计表格与可视化样例
reports/          阶段报告与图表
scripts/          运行、汇总、校验与绘图脚本
tests/            自动化单元测试
```

原始数据和模型权重不提交到 Git，默认位置为 `data/raw/` 和 `models/checkpoints/`。

## 环境

```bash
conda activate cjy_mob
python -m pip install -r requirements-experiments.txt
```

本机 GPU 实验环境的补充依赖见 `requirements-experiments.txt`。

## 运行与结果索引

### 单元测试

```bash
conda run -n cjy_mob python -m pytest tests -q -p no:cacheprovider
```

### M2 扰动基线验证

在仓库根目录运行 M2 ImageNet 预实验与全部 CHNCXR 共同正确样本（中断后添加 `--resume` 续跑）：

```bash
conda run -n cjy_mob python -m experiments.perturbation_pilot \
  --config configs/experiment/perturbation_imagenet_m2.yaml \
  --output results/metrics/imagenet_m2_20260911
conda run -n cjy_mob python -m experiments.perturbation_pilot \
  --config configs/experiment/perturbation_full.yaml \
  --output results/metrics/chncxr_full_main
```

结果校验与数据复核：

```bash
conda run -n cjy_mob python scripts/validate_perturbation_run.py results/metrics/imagenet_m2_20260911
conda run -n cjy_mob python scripts/validate_perturbation_run.py results/metrics/chncxr_full_main
```

共同正确筛选前后的 M2 数据复核见 `results/tables/m2_postscreen_audit.json`。

### M3 默认设置比较

运行 M3 默认设置、合并分片并验证结果：

```bash
bash scripts/run_m3_default.sh
python -m scripts.merge_m3_shards results/metrics/m3_imagenet_default_20260913 \
  results/metrics/m3_imagenet_default_20260913_shard{0,1,2,3}
python -m scripts.merge_m3_shards results/metrics/m3_chncxr_default_20260913 \
  results/metrics/m3_chncxr_default_20260913_shard{0,1,2,3}
python scripts/validate_m3_run.py results/metrics/m3_imagenet_default_20260913
python scripts/validate_m3_run.py results/metrics/m3_chncxr_default_20260913
```

默认设置汇总分别位于 `results/metrics/m3_imagenet_default_20260913/REPORT.md` 与 `results/metrics/m3_chncxr_default_20260913/REPORT.md`。

### M4 参数实验、基线敏感性与扩展分析

主实验与分析脚本：

```bash
# 调参集扫描（实验 1–3）与最终评价集（含选做实验 7）
bash scripts/run_m4_core.sh
bash scripts/run_m4_final.sh
python scripts/analyze_m4_config.py
python scripts/analyze_m4_final.py
python scripts/analyze_m4_localization.py

# 5 种解释基线敏感性实验
bash scripts/run_m4_baseline_sensitivity.sh

# 选做实验 5：OrdShap 官方实现校验、分阶段运行与汇总
python scripts/verify_ordshap_reference.py
python -m experiments.m4_ordshap --config configs/experiment/m4_ordshap.yaml --phase pilot
python -m experiments.m4_ordshap --config configs/experiment/m4_ordshap.yaml --phase main
python -m experiments.m4_ordshap --config configs/experiment/m4_ordshap.yaml --phase moves
python scripts/analyze_m4_ordshap.py

# 基于汇总 JSON 绘制 M4 报告图表（输出至 reports/m4/figures/）
python scripts/plot_m4.py all
```

ViT-B/16 跨架构与 SLIC/Patch 划分对照（运行前先通过 `python scripts/download_vit_weights.py` 准备 TorchVision ViT-B/16 ImageNet-1K V1 权重；完整配置见 `configs/experiment/m4_vit_extension.yaml`）：

```bash
# 检查实验矩阵
python -m experiments.m4_vit_extension \
  --experiment architecture --phase main \
  --output results/metrics/m4_vit_architecture --dry-run
python -m experiments.m4_vit_extension \
  --experiment partition --phase main \
  --output results/metrics/m4_vit_partition --dry-run

# 运行跨架构与 SLIC/Patch 实验（支持中断后续跑；参数可换为 architecture 或 partition）
bash scripts/run_m4_vit_extension.sh all
```

`architecture` 直接复用 M3 的 ResNet-50/VGG13 逐图结果，仅新增运行 ViT；`partition` 中的 ResNet-50 SLIC-196/Patch-16 为新控制条件，与 ViT 同步运行。

M4 已落盘的主要分析产物：

| 实验模块 | 汇总报告 / 核心产物 |
|---|---|
| 调参集扫描与配置冻结（实验 1–3） | `results/metrics/m4_config_analysis/REPORT.md` |
| 最终评价集检验与域迁移（选做实验 7） | `results/metrics/m4_final_analysis/REPORT.md` |
| 遮蔽基线敏感性分析（5 种解释基线） | `results/metrics/m4_baseline_sensitivity/REPORT.md` |
| OrdShap 位置归因（选做实验 5） | `results/metrics/m4_ordshap/REPORT.md`、`results/tables/ordshap_reference_check.json` |
| CHNCXR 定位失败定量诊断 | `results/metrics/m4_config_analysis/localization_diagnostics.json` |
| ViT-B/16 跨架构与 SLIC/Patch 对照 | `results/metrics/m4_vit_architecture/`、`results/metrics/m4_vit_partition/` |
| XAI01-01 梯度类外部对照材料 | `docs/XAI01-01_与扰动博弈论组对比_实验结果.md` |

## 报告

各阶段文档与报告位于 `reports/`：

- M1 需求建模报告与 PPT：`reports/requirements_modeling/`
- M2 数据探索分析：`reports/m2_eda/M2_EDA.md`
- M3 中期检查报告与 PPT：`reports/midterm_review/`

编译示例：

```bash
cd reports/midterm_review
latexmk -xelatex main.tex
```

## 团队分工

| 姓名 | 角色 | 主要职责 |
|---|---|---|
| 杜孟泽 | 数据 | 数据、划分、预处理与 EDA |
| 张泰瑜 | 模型 | 模型加载、值函数与前向计数 |
| 邹研泽 | 方法 | 归因方法实现 |
| 陈纪仰（组长） | 实验 | 评价、实验驱动与结果落盘 |
| 朱炳政 | 分析与文档 | 统计、报告与 PPT |

## 引用

- Ribeiro et al. *Why Should I Trust You?* KDD 2016. (LIME)
- Petsiuk et al. *RISE.* BMVC 2018.
- Lundberg & Lee. *A Unified Approach to Interpreting Model Predictions.* NeurIPS 2017. (SHAP)
- Hill et al. *OrdShap: Feature Position Importance for Sequential Black-Box Models.* NeurIPS 2025.
- Jaeger et al. *Two public chest X-ray datasets for computer-aided screening of pulmonary diseases.* QIMS 2014.
