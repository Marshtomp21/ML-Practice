# M1 候选集评价准备度分析

来源：ImageNet XML 类别标注、CHNCXR 文件 ID / 病灶标注文件及前届样例清单。

方法：完整候选池枚举与 ID 一致性检查；统计标注文件数量，不以连通域推断临床病灶数。
尚未运行模型共同正确筛选，以下数量是筛选前覆盖，不是最终实验规模。

元数据指纹：83e57faf75942f152eac29d3593f52702f66ad40292832ac9adb2b162aca8350。逐样本记录见同目录 JSON。

## 统计结果

{
  "imagenet_n": 500,
  "imagenet_classes": 284,
  "class_size_frequency": {
    "1": 165,
    "2": 68,
    "3": 22,
    "4": 17,
    "5": 9,
    "6": 1,
    "7": 2
  },
  "classes_singleton": 165,
  "images_in_classes_le2": 301,
  "max_class_size": 7,
  "chncxr_n": 662,
  "positive_n": 336,
  "negative_n": 326,
  "localization_n": 330,
  "unannotated_positive_ids": [
    "CHNCXR_0467_1",
    "CHNCXR_0484_1",
    "CHNCXR_0606_1",
    "CHNCXR_0609_1",
    "CHNCXR_0612_1",
    "CHNCXR_0624_1"
  ],
  "prior_n": 51,
  "prior_positive_n": 51,
  "prior_localization_n": 50,
  "lesion_annotation_n": 1153,
  "single_annotation_n": 55,
  "multi_annotation_n": 275,
  "lesion_count_median": 3.0,
  "lesion_count_q25": 2.0,
  "lesion_count_q75": 4.0,
  "lesion_count_max": 14,
  "lesion_count_frequency": {
    "1": 55,
    "2": 87,
    "3": 62,
    "4": 44,
    "5": 36,
    "6": 12,
    "7": 8,
    "8": 9,
    "9": 5,
    "10": 3,
    "11": 4,
    "12": 3,
    "13": 1,
    "14": 1
  },
  "metadata_sha256": "83e57faf75942f152eac29d3593f52702f66ad40292832ac9adb2b162aca8350"
}

## 对实验设计的影响

- 分类标签与定位掩码分别控制指标适用范围；缺失标注不等于负样本。
- 前届全阳性队列不能替代完整候选池。
- 类别样本稀疏时不支持逐类别显著性排名，按图像配对并审计筛选后覆盖。
- 单/多病灶标注分层；并集定位结果不能证明所有病灶均被覆盖。

复现：在 code 目录运行 python scripts/m1/fig2_evaluation_readiness.py。
