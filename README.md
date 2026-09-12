# XAI01-02：扰动类与博弈论类归因方法比较

机器学习综合实践 · XAI-T01「特征归因方法选择与评价」。项目比较不同归因方法的忠实性、稳定性与效率。

## 当前进度

已实现并测试 LIME、RISE、Ablation，并完成 ImageNet/CHNCXR × ResNet-50/VGG13 的 M2 端到端验证。CHNCXR 已运行全部 434 张共同正确样本，ImageNet 已完成 10 张固定共同正确样本的预实验；数值结果、归因数组和可视化均已落盘。SHAP/KernelSHAP 属于 M3，尚未实现。

| 阶段 | 内容 | 状态 |
|---|---|---|
| M1 | 需求、数据和实验设计 | 进行中 |
| M2 | 三个扰动基线与初步实验 | 已完成 |
| M3 | 博弈论类方法与初步比较 | 未开始 |
| M4 | 参数实验、消融与交叉对比 | 未开始 |
| M5 | 报告与答辩 | 未开始 |

## 项目设置

- 方法：LIME、RISE、Ablation；
- 数据：ImageNet 子集和 CHNCXR；
- 模型：ResNet-50、VGG13；
- 目标：正确类别的 pre-Softmax logit；
- 指标：Insertion/Deletion、Spearman、Top 20% Jaccard、耗时和前向次数；
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
