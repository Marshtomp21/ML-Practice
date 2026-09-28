# XAI01-02：扰动类与博弈论类归因方法比较

机器学习综合实践 · XAI-T01「特征归因方法选择与评价」。项目比较不同归因方法的忠实性、稳定性与效率。

## 当前进度

已实现并测试 LIME、RISE、Ablation、KernelSHAP。M2 已完成 ImageNet/CHNCXR × ResNet-50/VGG13 的端到端验证；M3 已完成四种核心方法在两个数据域和两种模型上的默认设置比较，以及输入稳定性、跨种子稳定性、效率和定位性评价。M4 的 ViT 扩展代码已实现，待在完整 ImageNet/GPU 环境运行。

| 阶段 | 内容 | 状态 |
|---|---|---|
| M1 | 需求、数据和实验设计 | 已完成 |
| M2 | 三个扰动基线与初步实验 | 已完成 |
| M3 | 博弈论类方法与初步比较 | 已完成 |
| M4 | 参数实验、ViT 跨架构与特征划分对照 | 代码已完成，待完整运行 |
| M5 | 报告与答辩 | 未开始 |

## 项目设置

- 方法：LIME、RISE、Ablation、KernelSHAP；
- 数据：ImageNet 子集和 CHNCXR；
- 模型：ResNet-50、VGG13；
- 目标：正确类别的 pre-Softmax logit；
- 指标：Insertion/Deletion、Max-Sensitivity、Cosine Similarity、Spearman、Top 20% Jaccard、耗时、前向次数、掩码内能量和 Pointing Game；
- CHNCXR 定位性评价只使用具有病灶掩码的阳性样本，其他指标使用全部共同正确样本。

M1 主设置为 SLIC 名义区域数 100、遮蔽率 0.50、前向预算 1024、随机种子 `{0,1,2,3,4}`。参数扫描在后续单因素实验中运行。

## 目录

```text
configs/          实验配置
data/             数据说明与固定划分
preprocessing/    图像预处理与超像素
models/           模型加载与推理
attribution/      归因方法
evaluation/       评价指标
experiments/      实验入口
results/          实验结果
reports/          项目报告
scripts/          辅助脚本
tests/            自动测试
```

原始数据和模型权重不提交到 Git。默认位置为 `data/raw/` 和 `models/checkpoints/`。

## 环境

```bash
conda activate cjy_mob
python -m pip install -r requirements-experiments.txt
```

本机 GPU 实验环境的补充依赖见 `requirements-experiments.txt`。

## 运行

在 `code/` 目录使用 `cjy_mob` 环境运行 M2 ImageNet 预实验：

```bash
conda run -n cjy_mob python -m experiments.perturbation_pilot \
  --config configs/experiment/perturbation_imagenet_m2.yaml \
  --output results/metrics/imagenet_m2_20260911
```

运行全部 CHNCXR 共同正确样本：

```bash
conda run -n cjy_mob python -m experiments.perturbation_pilot \
  --config configs/experiment/perturbation_full.yaml \
  --output results/metrics/chncxr_full_main
```

中断后添加 `--resume` 续跑。测试和结果校验：

```bash
conda run -n cjy_mob python -m pytest tests -q -p no:cacheprovider
conda run -n cjy_mob python scripts/validate_perturbation_run.py results/metrics/imagenet_m2_20260911
conda run -n cjy_mob python scripts/validate_perturbation_run.py results/metrics/chncxr_full_main
```

10 张胸片的首轮流程验证结果位于 `results/metrics/chncxr_pilot_20260910/`。
共同正确筛选前后的 M2 数据复核见 `results/tables/m2_postscreen_audit.json`。

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

M3 中期报告与演示材料位于 `reports/midterm_review/`。

运行 M4 ViT 扩展前，先准备 TorchVision ViT-B/16 ImageNet-1K V1 权重：

```bash
python scripts/download_vit_weights.py
```

检查实验矩阵（不读取数据和权重）：

```bash
python -m experiments.m4_vit_extension \
  --experiment architecture --phase main \
  --output results/metrics/m4_vit_architecture --dry-run
python -m experiments.m4_vit_extension \
  --experiment partition --phase main \
  --output results/metrics/m4_vit_partition --dry-run
```

运行全部跨架构和 SLIC/Patch 实验（支持中断后续跑）：

```bash
bash scripts/run_m4_vit_extension.sh all
```

也可将最后一个参数换成 `architecture` 或 `partition` 单独运行。`architecture` 直接复用 M3 的 ResNet-50/VGG13 逐图结果，本次只新增运行 ViT；`partition` 中的 ResNet-50 SLIC-196/Grid-196 是 M3 没有的新控制条件，仍需运行。完整协议、分阶段命令和输出说明见 `docx/M4_ViT扩展实验实施说明.md`。

## 报告

```bash
cd reports/requirements_modeling
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
- Jaeger et al. *Two public chest X-ray datasets for computer-aided screening of pulmonary diseases.* QIMS 2014.
