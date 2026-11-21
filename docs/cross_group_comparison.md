# XAI01-01（梯度类）与 XAI01-02（扰动/博弈论类）跨组对比分析

本文整合梯度类小组（XAI01-01）与本组（XAI01-02）在 ImageNet 与 CHNCXR 上的正式实验结果，从实验口径、核心指标表现、基线敏感性与方法适用场景四个方面展开对比，为 RQ3（博弈论与扰动方法相对于梯度类对照的实际表现）提供依据。

原始数据来源：
- 梯度类组汇总：`docs/XAI01-01_与扰动博弈论组对比_实验结果.md`
- 本组最终评价集：`results/metrics/m4_final_analysis/REPORT.md`
- 本组遮蔽基线敏感性：`results/metrics/m4_baseline_sensitivity/REPORT.md`

---

## 1. 两组实验口径对照

两组均以 ImageNet 验证子集与 ResNet-50 作为核心交集，但在评分尺度、评价粒度与稳定性定义上采用不同协议。下表列出具体口径差异，便于正确解读后续分表数值：

| 对比维度 | XAI01-01（梯度类） | XAI01-02（扰动 / 博弈论类） |
|---|---|---|
| 核心方法 | Saliency、IG-blur、Expected Gradients（EG）、Grad-CAM | LIME、RISE、Ablation、KernelSHAP（另有 OrdShap 扩展） |
| ImageNet 样本 | 100 张（ResNet-50 Top-1 正确，seed=42 每类 1 张） | 299 张最终评价集（ResNet-50 与 VGG13 共同正确） |
| 模型权重 | `ResNet50_Weights.IMAGENET1K_V2` | 同款 `ResNet50_Weights.IMAGENET1K_V2`（另有 VGG13、ViT-B/16） |
| 忠实性目标得分 | 目标类别 **Softmax 概率** | 目标类别 **pre-Softmax logit** |
| 评价遮蔽基线 | 预处理后 RGB 图像 GaussianBlur（`radius=15`） | 归一化输入空间**全零张量**（主协议） |
| 评价单位与步数 | $224\times224$ 逐像素排序，10 区间 / 11 采样点 | SLIC=100 超像素区域排序，20 步 / 21 采样点 |
| 输入扰动设置 | RGB 空间高斯噪声 $\sigma=0.01$，每图 1 次（保存为 8-bit PNG） | 均匀扰动 $\pm 0.01$，每图重复 10 次 |
| 稳定性指标 | 原图与噪声图的 Spearman、Top-10% IoU | 输入扰动 Max-Sensitivity、Cosine；随机方法跨种子 Spearman、Top-20% Jaccard |
| 统计汇总方式 | 100 图算术均值（配对 Bootstrap + Holm 校正 Wilcoxon） | 随机方法图内 5 种子均值，跨图取中位数与 95% CI（Holm 校正配对检验） |
| 计时边界 | CUDA 秒/图（Grad-CAM 为 RTX 4070 Laptop，EG/评价为 RTX 4060 Laptop） | 独占 GPU 秒/图，同时记录精确模型前向次数 |

由于忠实性得分尺度（概率 AUC $\in [0,1]$ vs logit AUC）、评价粒度（像素 vs SLIC 区域）与稳定性定义不同，下文将两组指标按各自原始口径分表呈现，再在范式层面对比方法特征、相对排序与质量—效率权衡。

---

## 2. ImageNet · ResNet-50 跨范式核心结果对比

### 2.1 XAI01-01 梯度类方法（ImageNet 100 图均值）

| 方法 | Insertion 概率 AUC ↑ | Deletion 概率 AUC ↓ | 输入 Spearman ↑ | Top-10% IoU ↑ | CUDA 耗时（秒/图）↓ |
|---|---:|---:|---:|---:|---:|
| Saliency | 0.2619 | 0.1378 | 0.6300 | 0.3271 | 0.0253 |
| **IG-blur** | **0.3689** | **0.0496** | 0.8278 | 0.5260 | 2.3405 |
| EG | 0.3177 | 0.0812 | 0.8950 | 0.7100 | 1.6714 |
| **Grad-CAM** | 0.3671 | 0.1043 | **0.9909** | **0.9588** | **0.0154** |

- 统计检验要点：IG-blur 与 Grad-CAM 在 Insertion 上无显著差异（$p_{\mathrm{holm}}=0.672$）；IG-blur 在 Deletion 上显著领先（$p_{\mathrm{holm}}=2.6\times10^{-14}$）；Grad-CAM 在 Spearman 与 Top-10% IoU 上显著领先（$p_{\mathrm{holm}}=2.34\times10^{-17}$）。

### 2.2 XAI01-02 扰动与博弈论类方法（ImageNet 299 图最终评价集中位数，预算 1024）

| 方法 | Insertion logit AUC ↑ | Deletion logit AUC ↓ | Max-Sens ↓ | Cosine ↑ | 跨种子 Spearman ↑ | 定位能量 ↑ | 独占 GPU 耗时（秒/图）↓ | 前向数 ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ablation** | 6.2852 | 5.5035 | 0.4161 | 0.9452 | — | 0.4736 | **0.248** | **87** |
| **LIME** | **6.8032** | **2.3137** | 0.1421 | 0.9940 | 0.5652 | **0.6829** | 3.421 | 1024 |
| **KernelSHAP** | 6.7025 | 2.3796 | 0.1156 | 0.9953 | 0.3871 | 0.6260 | 3.379 | 1024 |
| **RISE** | 6.1726 | 4.5785 | **0.0091** | **1.0000** | **0.9929** | 0.3176 | 5.741 | 1024 |

- 统计检验要点：LIME 与 KernelSHAP 稳居忠实性第一梯队，显著优于 Ablation 与 RISE（$p_{\mathrm{holm}}<10^{-15}$），两者在 ResNet-50 Deletion 上无显著差异（$p_{\mathrm{holm}}=0.215$）；RISE 在所有稳定性指标上全面领先；Ablation 以不到随机方法十分之一的前向次数提供快速基线。

### 2.3 跨范式特征综合对照

| 评价维度 | 梯度类（XAI01-01）代表与规律 | 扰动 / 博弈论类（XAI01-02）代表与规律 | 跨范式结论 |
|---|---|---|---|
| **忠实性优先** | **IG-blur**：512 步路径积分精准定位关键像素，Deletion 显著最优 | **LIME / KernelSHAP**：区域代理回归与 Shapley 加权估计联合刻画区域贡献，Insertion/Deletion 显著领先 | 两组忠实性最优方法均依赖高步数/高采样聚合，说明精细忠实性普遍以计算成本为代价 |
| **稳定性优先** | **Grad-CAM**：高层卷积特征图天然平滑，抗噪 Spearman 达 0.991、Top-10% IoU 达 0.959 | **RISE**：双线性上采样随机软掩码平滑局部波动，Max-Sens 最低、跨种子 Spearman 达 0.993 | 空间平滑机制大幅提升稳定性，但都会在一定程度上牺牲细粒度 Deletion 忠实性 |
| **运行效率** | **Grad-CAM**（0.015 s/图）与 **Saliency**（0.025 s/图）仅需单次前向+反向，比积分方法快 2 个数量级 | **Ablation**（0.248 s/图，87 次前向）最快；LIME/KernelSHAP（~3.4 s/图）与 RISE（5.7 s/图）需 1024 次前向 | 白盒梯度类在单图推理速度上具有天然结构优势（快约 1–2 个数量级）；黑盒方法则无需访问模型内部结构与梯度 |
| **可迁移性与适用边界** | Grad-CAM 依赖卷积层选择（迁移至 ViT 需改动）；Saliency/IG/EG 需可微白盒接口 | 全模型无关（Model-Agnostic），已直接迁移至 ResNet-50、VGG13 与 ViT-B/16 | 当模型为黑盒或需统一跨架构（CNN/ViT）接口时，扰动/博弈论类更具通用性；在白盒 CNN 实时部署中 Grad-CAM 性价比最高 |

---

## 3. 遮蔽与参考基线敏感性的跨范式呼应

两组均独立开展了基线敏感性控制实验，结论高度互补：

### 3.1 梯度类（XAI01-01：ImageNet 100 图 IG 四基线）

| IG 基线 | Insertion 概率 AUC ↑ | Deletion 概率 AUC ↓ | Spearman ↑ | Top-10% IoU ↑ |
|---|---:|---:|---:|---:|
| black | 0.2792 | 0.1426 | 0.7825 | 0.4950 |
| **blur** | **0.3689** | **0.0496** | 0.8278 | 0.5260 |
| mean | 0.3107 | 0.1045 | 0.7958 | 0.4859 |
| **random** | 0.3292 | 0.0733 | **0.9174** | **0.7437** |

- 规律：在固定方法（IG）内部，**blur 基线**忠实性显著最优，**random 基线**稳定性显著最优，基线本身即在忠实性与稳定性之间引入权衡。

### 3.2 扰动与博弈论类（XAI01-02：调参集 5 种解释基线）

本组在保持评价基线固定为 `normalized_zero` 的条件下，扫描了 `normalized_zero`、`dataset_mean_image`、`dataset_mean_color`、`gaussian_blur_sigma10` 与 `gaussian_noise_dataset_stats` 五种解释基线：

- **ImageNet（75 图）**：ResNet-50 在全部 5 种基线下，Insertion 与 Deletion 排序均稳定保持 `LIME > KernelSHAP > (Ablation/RISE)`；VGG13 仅在 `gaussian_noise` 的 Insertion 上出现 Ablation 反超，其余 4 种基线排序完全不变。
- **CHNCXR（86 图）**：基线敏感性显著放大——在 `normalized_zero` 下两模型均为 `LIME > KernelSHAP > Ablation > RISE`，但切换至 `dataset_mean_image` 或 `gaussian_blur_sigma10` 后，ResNet-50 与 VGG13 上均出现 **KernelSHAP 或 Ablation 反超 LIME** 的排序翻转。

**跨范式共性**：无论路径积分梯度还是黑盒区域遮蔽，“缺失特征的参考基线”都不是中性超参数——它会同时影响单个方法的忠实性/稳定性权衡（XAI01-01），也会在分布结构特殊的医学影像域改写不同归因方法的相对排序（XAI01-02）。

---

## 4. CHNCXR 医学影像进展与数据对齐现状

| 项目 | XAI01-01（梯度类） | XAI01-02（扰动 / 博弈论类） |
|---|---|---|
| 胸片分类模型 | 自训练 ImageNet 预训练 ResNet-50 微调（463/98/101 划分，测试集 Acc=90.10%、ROC-AUC=94.53%） | 课程统一提供的 ResNet-50 与 VGG13 二分类权重（全集 662 张推理筛选共同正确样本） |
| 胸片预处理 | RGB、Resize(256)、CenterCrop(224)、ImageNet 均值/标准差归一化 | Resize(224) 不裁剪、$\text{mean}=\text{std}=0.5$ 归一化（保持与病灶掩码空间严格对齐） |
| 归因评价状态 | 尚未完成四方法集中核验的 CHNCXR 正式汇总表 | 已完成调参集（86 张）、最终评价集（339 张，含 198 张病灶并集掩码阳性图）、基线敏感性与定位失败定量诊断 |
| 定位标注说明 | 暂无正式定位指标（提示肺区 Mask 不等同于病灶 Mask） | 使用 330 张阳性胸片的逐病灶并集掩码；诊断显示病灶面积中位仅占全图 1.57%，显著拉低定位能量与 Pointing Game |

当前两组的 CHNCXR 分类权重、测试划分与预处理均未统一，且梯度类组的胸片归因汇总尚未最终核验，因此 CHNCXR 域暂不进行跨组指标并排比较，医学域结论以本组完整实验与定位诊断结果为主。

---

## 5. 面向方法选型的跨组总结

结合两组实验结果，在自然图像分类（ResNet-50）场景下可形成清晰的方法选型象限：

1. **白盒实时与稳健解释首选 Grad-CAM**：耗时最短（~0.015 s/图）、抗噪最稳，且 Insertion 忠实性与 IG-blur 无显著差异，适合在线监控与快速诊断。
2. **白盒细粒度忠实性优先选 IG-blur**：像素级 Deletion 忠实性显著最优，但耗时比 Grad-CAM 高两个数量级（~2.34 s/图），且强依赖模糊基线选择。
3. **黑盒 / 跨架构忠实性与区域定位优先选 LIME 或 KernelSHAP**：两者无需梯度即可在 CNN 与 ViT 上取得最优的区域级 Insertion/Deletion 与目标定位能量；其中 LIME 略占忠实性与跨种子稳定性优势，KernelSHAP 额外提供精确的加性效率公理保证。
4. **黑盒高稳定性优先选 RISE，快速初筛选 Ablation**：RISE 具备接近确定性的跨种子与抗噪稳定性但计算开销最大；Ablation 以约 1/11 的前向预算提供高性价比的低成本参照。
