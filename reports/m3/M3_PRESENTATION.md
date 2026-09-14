# M3 中期检查演示提纲

## 第 1 页：M3 目标与完成状态

- 四种核心方法：Ablation、LIME、RISE、KernelSHAP
- 两个数据域：ImageNet、CHNCXR
- 两种模型：ResNet-50、VGG13
- 640/640 条默认设置结果完成
- Insertion/Deletion、输入稳定性、跨种子稳定性、效率、定位性全部落盘

## 第 2 页：统一实验协议

- 正确类别 Softmax 前 logit
- SLIC 名义区域数 100
- 归一化输入空间零基线
- 随机方法预算 1024、种子 0–4
- 输入稳定性 `ε=0.01、K=10`
- 同图、同模型、同目标类别配对比较

## 第 3 页：ImageNet 结果

- LIME 与 KernelSHAP 在忠实性上领先
- RISE 在输入与跨种子稳定性上领先
- Ablation 单图约 0.2 秒、约 86.5 次前向
- KernelSHAP 100/100 次满秩，效率约束残差为 0

![ImageNet 四方法对照](../../results/figures/m3_imagenet_default_20260913/resnet50_ILSVRC2012_val_00000319.png)

## 第 4 页：CHNCXR 结果

- LIME 与 KernelSHAP 在忠实性上领先
- RISE 在稳定性上领先
- 四种方法 Pointing Game 中位数均为 0
- 病灶内能量低于 ImageNet

![CHNCXR 四方法对照](../../results/figures/m3_chncxr_default_20260913/resnet50_CHNCXR_0499_1.png)

## 第 5 页：方法—目标映射

| 方法 | M3 观测优势 | M4 对应实验 |
|---|---|---|
| Ablation | 最低前向数与耗时 | 作为所有扫描的低成本参照 |
| LIME | 两域忠实性领先 | 扫描 SLIC 粒度、遮蔽率、预算 |
| RISE | 输入与跨种子稳定性领先 | 扫描网格、遮蔽率、预算 |
| KernelSHAP | 忠实性领先且满足效率约束 | 扫描预算并检查稳定性改善 |

## 第 6 页：M4 数据与任务

- CHNCXR 配置选择集：86 张
- CHNCXR 最终评价集：348 张
- 完成四项必做实验和统计分析
- 完成 OrdShap 位置归因与医学影像验证
- 开展单因素参数扫描和梯度类交叉比较
