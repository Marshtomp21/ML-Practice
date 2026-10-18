# M4 最终评价集结果

范围：`final_evaluation`（ImageNet 299 张，CHNCXR 339 张），配置由 `results/metrics/m4_config_analysis/final_selection.json` 在调参集上冻结。图像为分析单位，随机方法先在图内对 5 个种子取均值。
区间为图像级 bootstrap 中位数 95% CI（10000 次，基种子 20261009）。
`common_*` 为在统一 SLIC=100 划分上重算的 Insertion/Deletion。

## imagenet / resnet50

| 配置 | 图像 | Insertion | Deletion | 统一划分 Ins | 统一划分 Del | Max-Sens | Cosine | Spearman | 定位能量 | 秒/图（独占 GPU） | 前向数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ablation_default | 299 | 6.2852 | 5.5035 | 6.2852 | 5.5035 | 0.4161 | 0.9452 | — | 0.4736 | 0.248 | 87 |
| kernel_shap_budget_1024 | 299 | 6.7025 | 2.3796 | 6.7025 | 2.3796 | 0.1156 | 0.9953 | 0.3871 | 0.6260 | 3.379 | 1024 |
| lime_budget_1024 | 299 | 6.8032 | 2.3137 | 6.8032 | 2.3137 | 0.1421 | 0.9940 | 0.5652 | 0.6829 | 3.421 | 1024 |
| rise_budget_1024 | 299 | 6.1726 | 4.5785 | 6.1726 | 4.5785 | 0.0091 | 1.0000 | 0.9929 | 0.3176 | 5.741 | 1024 |
| rise_mask_0.75 | 299 | 6.3035 | 4.1341 | 6.3035 | 4.1341 | 0.0155 | 1.0000 | 0.9826 | 0.3300 | 5.750 | 1024 |

## imagenet / vgg13

| 配置 | 图像 | Insertion | Deletion | 统一划分 Ins | 统一划分 Del | Max-Sens | Cosine | Spearman | 定位能量 | 秒/图（独占 GPU） | 前向数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ablation_default | 299 | 16.9504 | 4.4446 | 16.9504 | 4.4446 | 0.1503 | 0.9930 | — | 0.5357 | 0.251 | 87 |
| kernel_shap_budget_1024 | 299 | 18.3144 | 2.0544 | 18.3144 | 2.0544 | 0.0734 | 0.9984 | 0.6000 | 0.6550 | 3.415 | 1024 |
| lime_budget_1024 | 299 | 18.7383 | 1.9789 | 18.7383 | 1.9789 | 0.0653 | 0.9989 | 0.7191 | 0.6955 | 3.454 | 1024 |
| rise_budget_1024 | 299 | 11.9664 | 5.4991 | 11.9664 | 5.4991 | 0.0236 | 1.0000 | 0.9930 | 0.3212 | 5.728 | 1024 |
| rise_mask_0.75 | 299 | 13.2810 | 4.8218 | 13.2810 | 4.8218 | 0.0228 | 1.0000 | 0.9827 | 0.3442 | 5.722 | 1024 |

## chncxr / resnet50

| 配置 | 图像 | Insertion | Deletion | 统一划分 Ins | 统一划分 Del | Max-Sens | Cosine | Spearman | 定位能量 | 秒/图（独占 GPU） | 前向数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ablation_default | 339 | 6.7637 | -4.6105 | 6.7637 | -4.6105 | 0.0430 | 0.9994 | — | 0.0126 | 0.267 | 92 |
| kernel_shap_budget_1024 | 339 | 7.9906 | -6.9595 | 7.9906 | -6.9595 | 0.0257 | 0.9998 | 0.7865 | 0.0109 | 3.453 | 1024 |
| lime_budget_1024 | 339 | 8.5232 | -7.2474 | 8.5232 | -7.2474 | 0.0214 | 0.9998 | 0.8783 | 0.0108 | 3.499 | 1024 |
| lime_slic_300 | 339 | 11.7982 | -10.3228 | 6.1401 | -4.5174 | 0.0156 | 0.9999 | 0.6435 | 0.0140 | 3.610 | 1024 |
| rise_budget_1024 | 339 | 0.5015 | 1.1567 | 0.5015 | 1.1567 | 0.0076 | 1.0000 | 0.9922 | 0.0000 | 5.869 | 1024 |
| rise_mask_0.25 | 339 | 2.3264 | -0.0934 | 2.3264 | -0.0934 | 0.0285 | 1.0000 | 0.9957 | 0.0000 | 5.833 | 1024 |
| rise_mask_0.75 | 339 | 0.1453 | 1.3158 | 0.1453 | 1.3158 | 0.0031 | 1.0000 | 0.9829 | 0.0000 | 5.842 | 1024 |

## chncxr / vgg13

| 配置 | 图像 | Insertion | Deletion | 统一划分 Ins | 统一划分 Del | Max-Sens | Cosine | Spearman | 定位能量 | 秒/图（独占 GPU） | 前向数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ablation_default | 339 | 10.4159 | -2.3447 | 10.4159 | -2.3447 | 0.0136 | 1.0000 | — | 0.0035 | 0.269 | 92 |
| kernel_shap_budget_1024 | 339 | 15.7089 | -3.8459 | 15.7089 | -3.8459 | 0.0129 | 1.0000 | 0.7382 | 0.0075 | 3.518 | 1024 |
| lime_budget_1024 | 339 | 16.0858 | -4.3635 | 16.0858 | -4.3635 | 0.0125 | 1.0000 | 0.7834 | 0.0064 | 3.574 | 1024 |
| lime_slic_300 | 339 | 19.1358 | -6.3223 | 11.4995 | -3.3021 | 0.0142 | 0.9999 | 0.4827 | 0.0118 | 3.736 | 1024 |
| rise_budget_1024 | 339 | 3.0057 | 0.2457 | 3.0057 | 0.2457 | 0.0047 | 1.0000 | 0.9909 | 0.0157 | 6.074 | 1024 |
| rise_grid_10 | 339 | 3.1158 | 0.2505 | 3.1158 | 0.2505 | 0.0040 | 1.0000 | 0.9926 | 0.0159 | 6.023 | 1024 |
| rise_mask_0.75 | 339 | 3.4691 | 0.1881 | 3.4691 | 0.1881 | 0.0019 | 1.0000 | 0.9796 | 0.0156 | 6.054 | 1024 |

## 确认性检验（默认设置，Holm 校正）

差值为右方法减左方法；r 为配对秩二列相关。

| 域 | 模型 | 指标 | 左 | 右 | 差值中位数 [95% CI] | r | p_holm |
|---|---|---|---|---|---:|---:|---:|
| imagenet | resnet50 | insertion_auc | LIME | RISE | -0.6707 [-0.7617, -0.5732] | -0.959 | 4.75e-46 * |
| imagenet | resnet50 | insertion_auc | LIME | Ablation | -0.5131 [-0.6749, -0.4503] | -0.779 | 7.39e-31 * |
| imagenet | resnet50 | insertion_auc | LIME | KernelSHAP | -0.0815 [-0.1005, -0.0544] | -0.558 | 1.18e-16 * |
| imagenet | resnet50 | insertion_auc | RISE | Ablation | 0.1124 [0.0134, 0.1781] | 0.017 | 7.96e-01 |
| imagenet | resnet50 | insertion_auc | RISE | KernelSHAP | 0.5169 [0.4414, 0.6293] | 0.931 | 1.64e-43 * |
| imagenet | resnet50 | insertion_auc | Ablation | KernelSHAP | 0.3157 [0.2170, 0.4206] | 0.637 | 4.08e-21 * |
| imagenet | resnet50 | deletion_auc | LIME | RISE | 1.8082 [1.6007, 1.9978] | 0.952 | 1.04e-45 * |
| imagenet | resnet50 | deletion_auc | LIME | Ablation | 2.5284 [2.3307, 2.7205] | 0.986 | 8.59e-49 * |
| imagenet | resnet50 | deletion_auc | LIME | KernelSHAP | 0.0649 [-0.0006, 0.1290] | 0.083 | 2.15e-01 |
| imagenet | resnet50 | deletion_auc | RISE | Ablation | 0.7260 [0.5067, 0.8981] | 0.546 | 5.78e-16 * |
| imagenet | resnet50 | deletion_auc | RISE | KernelSHAP | -1.6995 [-1.8692, -1.6022] | -0.991 | 3.67e-49 * |
| imagenet | resnet50 | deletion_auc | Ablation | KernelSHAP | -2.5405 [-2.7327, -2.2381] | -0.997 | 1.12e-49 * |
| imagenet | resnet50 | max_sensitivity | LIME | RISE | -0.1279 [-0.1395, -0.1181] | -1.000 | 5.32e-50 * |
| imagenet | resnet50 | max_sensitivity | LIME | Ablation | 0.2527 [0.2347, 0.2732] | 0.974 | 6.15e-48 * |
| imagenet | resnet50 | max_sensitivity | LIME | KernelSHAP | -0.0154 [-0.0228, -0.0057] | -0.487 | 2.80e-13 * |
| imagenet | resnet50 | max_sensitivity | RISE | Ablation | 0.4036 [0.3875, 0.4250] | 1.000 | 5.32e-50 * |
| imagenet | resnet50 | max_sensitivity | RISE | KernelSHAP | 0.1035 [0.1009, 0.1073] | 1.000 | 5.32e-50 * |
| imagenet | resnet50 | max_sensitivity | Ablation | KernelSHAP | -0.2937 [-0.3113, -0.2807] | -1.000 | 5.32e-50 * |
| imagenet | resnet50 | cosine_similarity | LIME | RISE | 0.0060 [0.0051, 0.0072] | 1.000 | 5.32e-50 * |
| imagenet | resnet50 | cosine_similarity | LIME | Ablation | -0.0439 [-0.0504, -0.0393] | -0.979 | 1.73e-48 * |
| imagenet | resnet50 | cosine_similarity | LIME | KernelSHAP | 0.0001 [-0.0002, 0.0010] | 0.269 | 5.47e-05 * |
| imagenet | resnet50 | cosine_similarity | RISE | Ablation | -0.0548 [-0.0611, -0.0498] | -1.000 | 5.32e-50 * |
| imagenet | resnet50 | cosine_similarity | RISE | KernelSHAP | -0.0047 [-0.0053, -0.0042] | -1.000 | 5.32e-50 * |
| imagenet | resnet50 | cosine_similarity | Ablation | KernelSHAP | 0.0500 [0.0441, 0.0547] | 1.000 | 5.32e-50 * |
| imagenet | resnet50 | spearman | LIME | RISE | 0.4259 [0.4131, 0.4502] | 1.000 | 2.66e-50 * |
| imagenet | resnet50 | spearman | LIME | KernelSHAP | -0.1601 [-0.1746, -0.1433] | -0.973 | 3.29e-48 * |
| imagenet | resnet50 | spearman | RISE | KernelSHAP | -0.6049 [-0.6219, -0.5866] | -1.000 | 2.66e-50 * |
| imagenet | resnet50 | top20_jaccard | LIME | RISE | 0.3920 [0.3731, 0.4154] | 1.000 | 2.66e-50 * |
| imagenet | resnet50 | top20_jaccard | LIME | KernelSHAP | -0.1291 [-0.1401, -0.1079] | -0.912 | 1.60e-42 * |
| imagenet | resnet50 | top20_jaccard | RISE | KernelSHAP | -0.5228 [-0.5423, -0.5011] | -1.000 | 2.66e-50 * |
| imagenet | vgg13 | insertion_auc | LIME | RISE | -6.1145 [-6.4667, -5.7069] | -1.000 | 5.71e-50 * |
| imagenet | vgg13 | insertion_auc | LIME | Ablation | -1.3276 [-1.5496, -1.1053] | -0.696 | 5.70e-25 * |
| imagenet | vgg13 | insertion_auc | LIME | KernelSHAP | -0.2376 [-0.3559, -0.1445] | -0.385 | 8.23e-09 * |
| imagenet | vgg13 | insertion_auc | RISE | Ablation | 4.6625 [4.2712, 5.2353] | 0.929 | 1.91e-43 * |
| imagenet | vgg13 | insertion_auc | RISE | KernelSHAP | 5.8796 [5.3585, 6.3558] | 0.998 | 7.33e-50 * |
| imagenet | vgg13 | insertion_auc | Ablation | KernelSHAP | 0.9297 [0.5755, 1.2963] | 0.572 | 1.97e-17 * |
| imagenet | vgg13 | deletion_auc | LIME | RISE | 3.4475 [3.2497, 3.7238] | 1.000 | 5.32e-50 * |
| imagenet | vgg13 | deletion_auc | LIME | Ablation | 2.0740 [1.8859, 2.2894] | 1.000 | 5.32e-50 * |
| imagenet | vgg13 | deletion_auc | LIME | KernelSHAP | 0.1753 [0.1103, 0.2417] | 0.516 | 1.32e-14 * |
| imagenet | vgg13 | deletion_auc | RISE | Ablation | -1.0580 [-1.3878, -0.8445] | -0.520 | 1.32e-14 * |
| imagenet | vgg13 | deletion_auc | RISE | KernelSHAP | -3.3496 [-3.5869, -3.0638] | -1.000 | 5.32e-50 * |
| imagenet | vgg13 | deletion_auc | Ablation | KernelSHAP | -1.8943 [-2.0319, -1.6148] | -0.998 | 5.32e-50 * |
| imagenet | vgg13 | max_sensitivity | LIME | RISE | -0.0370 [-0.0409, -0.0337] | -0.937 | 1.59e-44 * |
| imagenet | vgg13 | max_sensitivity | LIME | Ablation | 0.0822 [0.0760, 0.0873] | 1.000 | 5.32e-50 * |
| imagenet | vgg13 | max_sensitivity | LIME | KernelSHAP | 0.0064 [0.0051, 0.0084] | 0.571 | 1.15e-17 * |
| imagenet | vgg13 | max_sensitivity | RISE | Ablation | 0.1209 [0.1141, 0.1270] | 1.000 | 5.32e-50 * |
| imagenet | vgg13 | max_sensitivity | RISE | KernelSHAP | 0.0434 [0.0406, 0.0465] | 0.964 | 7.36e-47 * |
| imagenet | vgg13 | max_sensitivity | Ablation | KernelSHAP | -0.0752 [-0.0800, -0.0700] | -1.000 | 5.32e-50 * |
| imagenet | vgg13 | cosine_similarity | LIME | RISE | 0.0011 [0.0010, 0.0012] | 1.000 | 5.32e-50 * |
| imagenet | vgg13 | cosine_similarity | LIME | Ablation | -0.0058 [-0.0062, -0.0049] | -1.000 | 5.32e-50 * |
| imagenet | vgg13 | cosine_similarity | LIME | KernelSHAP | -0.0004 [-0.0004, -0.0003] | -0.774 | 4.01e-31 * |
| imagenet | vgg13 | cosine_similarity | RISE | Ablation | -0.0070 [-0.0077, -0.0064] | -1.000 | 5.32e-50 * |
| imagenet | vgg13 | cosine_similarity | RISE | KernelSHAP | -0.0016 [-0.0018, -0.0015] | -1.000 | 5.32e-50 * |
| imagenet | vgg13 | cosine_similarity | Ablation | KernelSHAP | 0.0053 [0.0044, 0.0059] | 1.000 | 5.32e-50 * |
| imagenet | vgg13 | spearman | LIME | RISE | 0.2717 [0.2555, 0.2851] | 1.000 | 2.66e-50 * |
| imagenet | vgg13 | spearman | LIME | KernelSHAP | -0.1108 [-0.1274, -0.0866] | -0.875 | 2.48e-39 * |
| imagenet | vgg13 | spearman | RISE | KernelSHAP | -0.3905 [-0.4091, -0.3721] | -1.000 | 2.66e-50 * |
| imagenet | vgg13 | top20_jaccard | LIME | RISE | 0.2466 [0.2296, 0.2649] | 0.996 | 6.66e-50 * |
| imagenet | vgg13 | top20_jaccard | LIME | KernelSHAP | -0.1086 [-0.1308, -0.0908] | -0.841 | 2.68e-36 * |
| imagenet | vgg13 | top20_jaccard | RISE | KernelSHAP | -0.3552 [-0.3872, -0.3295] | -1.000 | 2.71e-50 * |
| chncxr | resnet50 | insertion_auc | LIME | RISE | -7.5089 [-7.7185, -7.3680] | -1.000 | 1.53e-56 * |
| chncxr | resnet50 | insertion_auc | LIME | Ablation | -1.7391 [-1.8631, -1.6112] | -0.983 | 5.26e-55 * |
| chncxr | resnet50 | insertion_auc | LIME | KernelSHAP | -0.3649 [-0.4231, -0.3036] | -0.844 | 2.18e-41 * |
| chncxr | resnet50 | insertion_auc | RISE | Ablation | 5.8730 [5.5938, 6.0560] | 1.000 | 1.53e-56 * |
| chncxr | resnet50 | insertion_auc | RISE | KernelSHAP | 7.0665 [6.8879, 7.3634] | 1.000 | 1.53e-56 * |
| chncxr | resnet50 | insertion_auc | Ablation | KernelSHAP | 1.3090 [1.1840, 1.4255] | 0.944 | 5.38e-51 * |
| chncxr | resnet50 | deletion_auc | LIME | RISE | 8.1961 [7.9422, 8.5057] | 1.000 | 1.53e-56 * |
| chncxr | resnet50 | deletion_auc | LIME | Ablation | 2.4619 [2.2246, 2.5981] | 0.999 | 1.53e-56 * |
| chncxr | resnet50 | deletion_auc | LIME | KernelSHAP | 0.3213 [0.2705, 0.3875] | 0.792 | 1.39e-36 * |
| chncxr | resnet50 | deletion_auc | RISE | Ablation | -5.6553 [-5.9136, -5.3630] | -1.000 | 1.53e-56 * |
| chncxr | resnet50 | deletion_auc | RISE | KernelSHAP | -7.7238 [-8.0058, -7.4949] | -1.000 | 1.53e-56 * |
| chncxr | resnet50 | deletion_auc | Ablation | KernelSHAP | -2.0201 [-2.1844, -1.8338] | -0.998 | 1.53e-56 * |
| chncxr | resnet50 | max_sensitivity | LIME | RISE | -0.0135 [-0.0142, -0.0131] | -1.000 | 1.53e-56 * |
| chncxr | resnet50 | max_sensitivity | LIME | Ablation | 0.0211 [0.0200, 0.0220] | 1.000 | 1.53e-56 * |
| chncxr | resnet50 | max_sensitivity | LIME | KernelSHAP | 0.0037 [0.0034, 0.0042] | 0.984 | 1.37e-55 * |
| chncxr | resnet50 | max_sensitivity | RISE | Ablation | 0.0348 [0.0336, 0.0361] | 1.000 | 1.53e-56 * |
| chncxr | resnet50 | max_sensitivity | RISE | KernelSHAP | 0.0178 [0.0168, 0.0184] | 1.000 | 1.53e-56 * |
| chncxr | resnet50 | max_sensitivity | Ablation | KernelSHAP | -0.0176 [-0.0185, -0.0166] | -1.000 | 1.53e-56 * |
| chncxr | resnet50 | cosine_similarity | LIME | RISE | 0.0002 [0.0001, 0.0002] | 1.000 | 1.53e-56 * |
| chncxr | resnet50 | cosine_similarity | LIME | Ablation | -0.0004 [-0.0004, -0.0004] | -1.000 | 1.53e-56 * |
| chncxr | resnet50 | cosine_similarity | LIME | KernelSHAP | -0.0001 [-0.0001, -0.0001] | -0.997 | 1.53e-56 * |
| chncxr | resnet50 | cosine_similarity | RISE | Ablation | -0.0006 [-0.0006, -0.0005] | -1.000 | 1.53e-56 * |
| chncxr | resnet50 | cosine_similarity | RISE | KernelSHAP | -0.0002 [-0.0002, -0.0002] | -1.000 | 1.53e-56 * |
| chncxr | resnet50 | cosine_similarity | Ablation | KernelSHAP | 0.0003 [0.0003, 0.0004] | 1.000 | 1.53e-56 * |
| chncxr | resnet50 | spearman | LIME | RISE | 0.1132 [0.1088, 0.1162] | 1.000 | 7.65e-57 * |
| chncxr | resnet50 | spearman | LIME | KernelSHAP | -0.0908 [-0.0977, -0.0833] | -0.998 | 7.65e-57 * |
| chncxr | resnet50 | spearman | RISE | KernelSHAP | -0.2055 [-0.2194, -0.1957] | -1.000 | 7.65e-57 * |
| chncxr | resnet50 | top20_jaccard | LIME | RISE | 0.2001 [0.1913, 0.2068] | 0.999 | 7.65e-57 * |
| chncxr | resnet50 | top20_jaccard | LIME | KernelSHAP | -0.1175 [-0.1273, -0.1063] | -0.935 | 3.22e-50 * |
| chncxr | resnet50 | top20_jaccard | RISE | KernelSHAP | -0.3151 [-0.3292, -0.2981] | -1.000 | 7.65e-57 * |
| chncxr | vgg13 | insertion_auc | LIME | RISE | -11.8597 [-13.2734, -10.5230] | -0.999 | 1.67e-56 * |
| chncxr | vgg13 | insertion_auc | LIME | Ablation | -3.6728 [-4.6735, -3.0051] | -0.862 | 1.31e-42 * |
| chncxr | vgg13 | insertion_auc | LIME | KernelSHAP | -0.2710 [-0.3713, -0.1926] | -0.506 | 7.05e-16 * |
| chncxr | vgg13 | insertion_auc | RISE | Ablation | 6.3297 [5.5878, 7.0810] | 0.994 | 4.31e-56 * |
| chncxr | vgg13 | insertion_auc | RISE | KernelSHAP | 11.7175 [10.5592, 12.5563] | 1.000 | 1.67e-56 * |
| chncxr | vgg13 | insertion_auc | Ablation | KernelSHAP | 3.5200 [3.0031, 4.3327] | 0.825 | 2.92e-39 * |
| chncxr | vgg13 | deletion_auc | LIME | RISE | 7.0806 [6.2535, 7.6859] | 1.000 | 1.53e-56 * |
| chncxr | vgg13 | deletion_auc | LIME | Ablation | 1.8713 [1.4053, 2.9765] | 0.880 | 2.34e-44 * |
| chncxr | vgg13 | deletion_auc | LIME | KernelSHAP | 0.2584 [0.1926, 0.3627] | 0.605 | 4.89e-22 * |
| chncxr | vgg13 | deletion_auc | RISE | Ablation | -3.3368 [-3.5937, -2.9752] | -0.919 | 4.57e-48 * |
| chncxr | vgg13 | deletion_auc | RISE | KernelSHAP | -6.7658 [-7.3978, -5.8938] | -1.000 | 1.53e-56 * |
| chncxr | vgg13 | deletion_auc | Ablation | KernelSHAP | -1.6561 [-2.5762, -1.2291] | -0.846 | 2.91e-41 * |
| chncxr | vgg13 | max_sensitivity | LIME | RISE | -0.0059 [-0.0066, -0.0052] | -0.465 | 2.32e-13 * |
| chncxr | vgg13 | max_sensitivity | LIME | Ablation | 0.0014 [0.0011, 0.0017] | 0.526 | 1.77e-16 * |
| chncxr | vgg13 | max_sensitivity | LIME | KernelSHAP | 0.0005 [0.0003, 0.0005] | 0.580 | 1.26e-19 * |
| chncxr | vgg13 | max_sensitivity | RISE | Ablation | 0.0073 [0.0067, 0.0080] | 0.570 | 4.84e-19 * |
| chncxr | vgg13 | max_sensitivity | RISE | KernelSHAP | 0.0064 [0.0057, 0.0071] | 0.486 | 2.54e-14 * |
| chncxr | vgg13 | max_sensitivity | Ablation | KernelSHAP | -0.0010 [-0.0012, -0.0006] | -0.404 | 1.09e-10 * |
| chncxr | vgg13 | cosine_similarity | LIME | RISE | 0.0000 [0.0000, 0.0000] | 0.745 | 4.51e-32 * |
| chncxr | vgg13 | cosine_similarity | LIME | Ablation | -0.0000 [-0.0000, -0.0000] | -0.591 | 7.77e-21 * |
| chncxr | vgg13 | cosine_similarity | LIME | KernelSHAP | -0.0000 [-0.0000, -0.0000] | -0.828 | 4.70e-39 * |
| chncxr | vgg13 | cosine_similarity | RISE | Ablation | -0.0000 [-0.0000, -0.0000] | -0.768 | 7.85e-34 * |
| chncxr | vgg13 | cosine_similarity | RISE | KernelSHAP | -0.0000 [-0.0000, -0.0000] | -0.755 | 7.75e-33 * |
| chncxr | vgg13 | cosine_similarity | Ablation | KernelSHAP | 0.0000 [0.0000, 0.0000] | 0.389 | 5.15e-10 * |
| chncxr | vgg13 | spearman | LIME | RISE | 0.1991 [0.1912, 0.2052] | 1.000 | 7.65e-57 * |
| chncxr | vgg13 | spearman | LIME | KernelSHAP | -0.0479 [-0.0557, -0.0394] | -0.729 | 2.99e-31 * |
| chncxr | vgg13 | spearman | RISE | KernelSHAP | -0.2521 [-0.2600, -0.2389] | -1.000 | 7.65e-57 * |
| chncxr | vgg13 | top20_jaccard | LIME | RISE | 0.2104 [0.1947, 0.2259] | 0.990 | 6.48e-56 * |
| chncxr | vgg13 | top20_jaccard | LIME | KernelSHAP | -0.0558 [-0.0680, -0.0398] | -0.571 | 7.72e-20 * |
| chncxr | vgg13 | top20_jaccard | RISE | KernelSHAP | -0.2692 [-0.2777, -0.2514] | -1.000 | 7.71e-57 * |

## 域内调参配置相对默认（探索性）

| 域 | 模型 | 方法 | 调参配置 | ΔIns | ΔDel | Δ统一划分 Ins | Δ统一划分 Del | ΔMax-Sens |
|---|---|---|---|---:|---:|---:|---:|---:|
| imagenet | resnet50 | RISE | `rise_mask_0.75` | 0.0939 [0.0774, 0.1142] | -0.2746 [-0.3264, -0.2301] | 0.0939 [0.0774, 0.1126] | -0.2746 [-0.3264, -0.2298] | 0.0048 [0.0039, 0.0060] |
| imagenet | vgg13 | RISE | `rise_mask_0.75` | 1.0692 [0.9122, 1.1708] | -0.6652 [-0.7352, -0.6027] | 1.0692 [0.9122, 1.1708] | -0.6652 [-0.7352, -0.6027] | -0.0013 [-0.0020, -0.0005] |
| chncxr | resnet50 | LIME | `lime_slic_300` | 3.5155 [3.1238, 3.7627] | -3.0847 [-3.2892, -2.8786] | -2.3527 [-2.4712, -2.2251] | 2.5889 [2.4729, 2.7182] | -0.0058 [-0.0064, -0.0054] |
| chncxr | resnet50 | RISE | `rise_mask_0.25` | 0.1508 [0.0945, 0.2448] | -0.1380 [-0.2023, -0.0908] | 0.1508 [0.1021, 0.2448] | -0.1380 [-0.2023, -0.0908] | 0.0212 [0.0181, 0.0261] |
| chncxr | vgg13 | LIME | `lime_slic_300` | 1.9699 [1.6227, 3.2307] | -1.1607 [-1.5108, -0.7919] | -3.1494 [-3.6048, -2.6051] | 1.4915 [1.0127, 2.0171] | 0.0016 [0.0014, 0.0021] |
| chncxr | vgg13 | RISE | `rise_grid_10` | 0.2179 [0.1480, 0.2506] | -0.0924 [-0.1410, -0.0449] | 0.2179 [0.1480, 0.2506] | -0.0924 [-0.1410, -0.0449] | -0.0003 [-0.0004, -0.0003] |

## 选做实验 7：CHNCXR 域内调参 vs ImageNet 参数直接迁移

有病灶并集掩码的最终评价胸片 198 张；差值为域内调参减直接迁移。

| 模型 | 方法 | 迁移配置 | 调参配置 | 子集 | 图像 | ΔIns | ΔDel | Δ统一划分 Ins | Δ统一划分 Del | Δ定位能量 | ΔPointing |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| resnet50 | LIME | `lime_budget_1024` | `lime_slic_300` | all | 339 | 3.5155 [3.1238, 3.7627] | -3.0847 [-3.2892, -2.8786] | -2.3527 [-2.4712, -2.2251] | 2.5889 [2.4729, 2.7182] | 0.0009 [0.0003, 0.0019] | 0.0000 [0.0000, 0.0000] |
| resnet50 | LIME | `lime_budget_1024` | `lime_slic_300` | lesion_masked | 198 | 4.4842 [4.2173, 4.7124] | -2.2493 [-2.3792, -2.0255] | -2.1735 [-2.3648, -2.0468] | 2.5572 [2.4010, 2.6926] | 0.0009 [0.0003, 0.0019] | 0.0000 [0.0000, 0.0000] |
| resnet50 | LIME | `lime_budget_1024` | `lime_slic_300` | lesion_below_one_region | 75 | 4.6043 [4.1251, 4.7864] | -2.1788 [-2.4267, -1.8654] | -2.0605 [-2.2251, -1.8248] | 2.5597 [2.2172, 2.7945] | 0.0004 [0.0001, 0.0009] | 0.0000 [0.0000, 0.0000] |
| resnet50 | LIME | `lime_budget_1024` | `lime_slic_300` | lesion_at_least_one_region | 123 | 4.4627 [4.1180, 4.7348] | -2.2520 [-2.4377, -1.9477] | -2.3007 [-2.4794, -2.0604] | 2.5556 [2.3836, 2.7448] | 0.0021 [-0.0001, 0.0045] | 0.0000 [0.0000, 0.0000] |
| resnet50 | LIME | `lime_budget_1024` | `lime_slic_300` | single_annotation | 30 | 4.6264 [4.0164, 5.1691] | -2.3194 [-2.5684, -1.8691] | -1.8852 [-2.2723, -1.5870] | 2.4983 [1.8772, 2.7538] | 0.0001 [-0.0000, 0.0010] | 0.0000 [0.0000, 0.0000] |
| resnet50 | LIME | `lime_budget_1024` | `lime_slic_300` | multiple_annotations | 168 | 4.4650 [4.2010, 4.6949] | -2.2052 [-2.3972, -1.9582] | -2.2315 [-2.4333, -2.0605] | 2.5572 [2.3992, 2.7066] | 0.0013 [0.0004, 0.0023] | 0.0000 [0.0000, 0.0000] |
| resnet50 | RISE | `rise_mask_0.75` | `rise_mask_0.25` | all | 339 | 0.2083 [0.1484, 0.3261] | -0.2072 [-0.2964, -0.1357] | 0.2083 [0.1484, 0.3261] | -0.2072 [-0.2964, -0.1357] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| resnet50 | RISE | `rise_mask_0.75` | `rise_mask_0.25` | lesion_masked | 198 | 0.9393 [0.6524, 1.3202] | -0.7150 [-1.0709, -0.5423] | 0.9393 [0.6588, 1.3202] | -0.7150 [-1.0679, -0.5315] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| resnet50 | RISE | `rise_mask_0.75` | `rise_mask_0.25` | lesion_below_one_region | 75 | 0.6588 [0.4104, 1.2266] | -0.6116 [-1.0251, -0.3886] | 0.6588 [0.4489, 1.2266] | -0.6116 [-1.0251, -0.3886] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| resnet50 | RISE | `rise_mask_0.75` | `rise_mask_0.25` | lesion_at_least_one_region | 123 | 1.2034 [0.7229, 1.6933] | -0.7759 [-1.3041, -0.5214] | 1.2034 [0.7229, 1.6933] | -0.7759 [-1.3041, -0.5214] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| resnet50 | RISE | `rise_mask_0.75` | `rise_mask_0.25` | single_annotation | 30 | 1.2059 [0.4646, 1.3685] | -0.6876 [-1.3303, -0.4405] | 1.2059 [0.4646, 1.3685] | -0.6876 [-1.3303, -0.4405] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| resnet50 | RISE | `rise_mask_0.75` | `rise_mask_0.25` | multiple_annotations | 168 | 0.8984 [0.6631, 1.5419] | -0.7150 [-1.1183, -0.5101] | 0.8984 [0.6588, 1.5419] | -0.7150 [-1.1278, -0.5139] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| vgg13 | LIME | `lime_budget_1024` | `lime_slic_300` | all | 339 | 1.9699 [1.6227, 3.2307] | -1.1607 [-1.5108, -0.7919] | -3.1494 [-3.6048, -2.6051] | 1.4915 [1.0127, 2.0171] | 0.0034 [0.0027, 0.0049] | 0.0000 [0.0000, 0.0000] |
| vgg13 | LIME | `lime_budget_1024` | `lime_slic_300` | lesion_masked | 198 | 3.4384 [2.1670, 4.5378] | -0.7695 [-1.1254, -0.4777] | -5.5569 [-6.3164, -5.0102] | 0.4747 [0.3518, 0.5931] | 0.0034 [0.0027, 0.0049] | 0.0000 [0.0000, 0.0000] |
| vgg13 | LIME | `lime_budget_1024` | `lime_slic_300` | lesion_below_one_region | 75 | 3.4512 [1.9053, 5.3572] | -0.8319 [-1.3790, -0.4423] | -6.1675 [-7.1956, -5.3178] | 0.3820 [0.1745, 0.5520] | 0.0010 [0.0005, 0.0015] | 0.0000 [0.0000, 0.0000] |
| vgg13 | LIME | `lime_budget_1024` | `lime_slic_300` | lesion_at_least_one_region | 123 | 3.4255 [1.8746, 4.9252] | -0.6853 [-1.1254, -0.2885] | -5.1230 [-6.1740, -4.4824] | 0.5738 [0.3786, 0.6927] | 0.0080 [0.0060, 0.0102] | 0.0000 [0.0000, 0.0000] |
| vgg13 | LIME | `lime_budget_1024` | `lime_slic_300` | single_annotation | 30 | 3.4847 [0.9192, 6.7531] | -0.6771 [-1.3452, -0.2130] | -6.2073 [-8.2013, -5.0045] | 0.3381 [0.1309, 0.7370] | 0.0013 [0.0003, 0.0026] | 0.0000 [0.0000, 0.0000] |
| vgg13 | LIME | `lime_budget_1024` | `lime_slic_300` | multiple_annotations | 168 | 3.4384 [1.9694, 4.5015] | -0.7944 [-1.1769, -0.4226] | -5.2816 [-6.3164, -4.8581] | 0.5121 [0.3670, 0.6013] | 0.0044 [0.0028, 0.0059] | 0.0000 [0.0000, 0.0000] |
| vgg13 | RISE | `rise_mask_0.75` | `rise_grid_10` | all | 339 | -0.1005 [-0.2279, 0.0073] | 0.0771 [-0.0494, 0.1767] | -0.1005 [-0.2279, 0.0073] | 0.0771 [-0.0494, 0.1767] | 0.0001 [0.0000, 0.0002] | 0.0000 [0.0000, 0.0000] |
| vgg13 | RISE | `rise_mask_0.75` | `rise_grid_10` | lesion_masked | 198 | -0.3780 [-0.4392, -0.3017] | 0.1368 [0.0326, 0.2337] | -0.3780 [-0.4392, -0.2962] | 0.1368 [0.0335, 0.2339] | 0.0001 [0.0000, 0.0002] | 0.0000 [0.0000, 0.0000] |
| vgg13 | RISE | `rise_mask_0.75` | `rise_grid_10` | lesion_below_one_region | 75 | -0.3017 [-0.3881, -0.1953] | 0.2024 [0.0257, 0.2780] | -0.3017 [-0.3881, -0.1953] | 0.2024 [0.0257, 0.2815] | -0.0000 [-0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| vgg13 | RISE | `rise_mask_0.75` | `rise_grid_10` | lesion_at_least_one_region | 123 | -0.4176 [-0.5239, -0.3152] | 0.1295 [-0.0028, 0.2341] | -0.4176 [-0.5239, -0.3152] | 0.1295 [-0.0028, 0.2341] | 0.0005 [0.0003, 0.0007] | 0.0000 [0.0000, 0.0000] |
| vgg13 | RISE | `rise_mask_0.75` | `rise_grid_10` | single_annotation | 30 | -0.3321 [-0.5443, -0.1229] | 0.2023 [-0.1512, 0.3799] | -0.3321 [-0.5443, -0.1229] | 0.2023 [-0.1512, 0.3798] | 0.0000 [-0.0000, 0.0003] | 0.0000 [0.0000, 0.0000] |
| vgg13 | RISE | `rise_mask_0.75` | `rise_grid_10` | multiple_annotations | 168 | -0.3836 [-0.4522, -0.3017] | 0.1306 [0.0240, 0.2341] | -0.3836 [-0.4495, -0.2964] | 0.1306 [0.0240, 0.2341] | 0.0001 [0.0000, 0.0003] | 0.0000 [0.0000, 0.0000] |

实验 7 检验族（每模型 × 指标，LIME 与 RISE 两个比较，Holm）：

| 模型 | 指标 | 方法 | 差值中位数 | p_holm |
|---|---|---|---:|---:|
| resnet50 | insertion_auc | LIME | 3.5155 | 6.25e-57 * |
| resnet50 | insertion_auc | RISE | 0.2083 | 7.96e-27 * |
| resnet50 | deletion_auc | LIME | -3.0847 | 5.52e-57 * |
| resnet50 | deletion_auc | RISE | -0.2072 | 5.33e-22 * |
| resnet50 | common_insertion_auc | LIME | -2.3527 | 5.10e-57 * |
| resnet50 | common_insertion_auc | RISE | 0.2083 | 7.96e-27 * |
| resnet50 | common_deletion_auc | LIME | 2.5889 | 5.10e-57 * |
| resnet50 | common_deletion_auc | RISE | -0.2072 | 5.33e-22 * |
| resnet50 | max_sensitivity | LIME | -0.0058 | 5.10e-57 * |
| resnet50 | max_sensitivity | RISE | 0.0256 | 5.10e-57 * |
| resnet50 | cosine_similarity | LIME | 0.0001 | 1.33e-56 * |
| resnet50 | cosine_similarity | RISE | -0.0000 | 7.72e-46 * |
| resnet50 | localization_energy | LIME | 0.0009 | 1.99e-03 * |
| resnet50 | localization_energy | RISE | 0.0000 | 2.22e-10 * |
| resnet50 | pointing_game | LIME | 0.0000 | 1.00e+00 |
| resnet50 | pointing_game | RISE | 0.0000 | 8.43e-02 |
| vgg13 | insertion_auc | LIME | 1.9699 | 9.35e-24 * |
| vgg13 | insertion_auc | RISE | -0.1005 | 3.69e-01 |
| vgg13 | deletion_auc | LIME | -1.1607 | 1.08e-17 * |
| vgg13 | deletion_auc | RISE | 0.0771 | 9.23e-01 |
| vgg13 | common_insertion_auc | LIME | -3.1494 | 4.74e-50 * |
| vgg13 | common_insertion_auc | RISE | -0.1005 | 3.69e-01 |
| vgg13 | common_deletion_auc | LIME | 1.4915 | 1.75e-45 * |
| vgg13 | common_deletion_auc | RISE | 0.0771 | 9.23e-01 |
| vgg13 | max_sensitivity | LIME | 0.0016 | 1.30e-14 * |
| vgg13 | max_sensitivity | RISE | 0.0014 | 1.05e-30 * |
| vgg13 | cosine_similarity | LIME | -0.0000 | 1.19e-50 * |
| vgg13 | cosine_similarity | RISE | 0.0000 | 2.39e-03 * |
| vgg13 | localization_energy | LIME | 0.0034 | 4.44e-18 * |
| vgg13 | localization_energy | RISE | 0.0001 | 1.88e-05 * |
| vgg13 | pointing_game | LIME | 0.0000 | 8.19e-02 |
| vgg13 | pointing_game | RISE | 0.0000 | 8.19e-02 |

## 方法排序（默认设置，按中位数，1 为最优）

| 域 | 模型 | 指标 | 排序 |
|---|---|---|---|
| imagenet | resnet50 | insertion_auc | LIME > KernelSHAP > Ablation > RISE |
| imagenet | resnet50 | deletion_auc | LIME > KernelSHAP > RISE > Ablation |
| imagenet | resnet50 | max_sensitivity | RISE > KernelSHAP > LIME > Ablation |
| imagenet | resnet50 | cosine_similarity | RISE > KernelSHAP > LIME > Ablation |
| imagenet | resnet50 | attribution_seconds | Ablation > KernelSHAP > LIME > RISE |
| imagenet | resnet50 | localization_energy | LIME > KernelSHAP > Ablation > RISE |
| imagenet | vgg13 | insertion_auc | LIME > KernelSHAP > Ablation > RISE |
| imagenet | vgg13 | deletion_auc | LIME > KernelSHAP > Ablation > RISE |
| imagenet | vgg13 | max_sensitivity | RISE > LIME > KernelSHAP > Ablation |
| imagenet | vgg13 | cosine_similarity | RISE > LIME > KernelSHAP > Ablation |
| imagenet | vgg13 | attribution_seconds | Ablation > KernelSHAP > LIME > RISE |
| imagenet | vgg13 | localization_energy | LIME > KernelSHAP > Ablation > RISE |
| chncxr | resnet50 | insertion_auc | LIME > KernelSHAP > Ablation > RISE |
| chncxr | resnet50 | deletion_auc | LIME > KernelSHAP > Ablation > RISE |
| chncxr | resnet50 | max_sensitivity | RISE > LIME > KernelSHAP > Ablation |
| chncxr | resnet50 | cosine_similarity | RISE > LIME > KernelSHAP > Ablation |
| chncxr | resnet50 | attribution_seconds | Ablation > KernelSHAP > LIME > RISE |
| chncxr | resnet50 | localization_energy | Ablation > KernelSHAP > LIME > RISE |
| chncxr | vgg13 | insertion_auc | LIME > KernelSHAP > Ablation > RISE |
| chncxr | vgg13 | deletion_auc | LIME > KernelSHAP > Ablation > RISE |
| chncxr | vgg13 | max_sensitivity | RISE > LIME > KernelSHAP > Ablation |
| chncxr | vgg13 | cosine_similarity | RISE > LIME > KernelSHAP > Ablation |
| chncxr | vgg13 | attribution_seconds | Ablation > KernelSHAP > LIME > RISE |
| chncxr | vgg13 | localization_energy | RISE > KernelSHAP > LIME > Ablation |

## 遮蔽基线是否改变方法排序（调参集基线敏感性实验）

| 域 | 模型 | 解释基线 | 指标 | 排序 |
|---|---|---|---|---|
| chncxr | resnet50 | dataset_mean_color | normalized_insertion_auc | LIME > KernelSHAP > Ablation > RISE |
| chncxr | resnet50 | dataset_mean_color | normalized_deletion_auc | LIME > KernelSHAP > Ablation > RISE |
| chncxr | resnet50 | dataset_mean_image | normalized_insertion_auc | Ablation > LIME > KernelSHAP > RISE |
| chncxr | resnet50 | dataset_mean_image | normalized_deletion_auc | KernelSHAP > LIME > Ablation > RISE |
| chncxr | resnet50 | gaussian_blur_sigma10 | normalized_insertion_auc | KernelSHAP > LIME > Ablation > RISE |
| chncxr | resnet50 | gaussian_blur_sigma10 | normalized_deletion_auc | KernelSHAP > LIME > Ablation > RISE |
| chncxr | resnet50 | gaussian_noise_dataset_stats | normalized_insertion_auc | LIME > KernelSHAP > Ablation > RISE |
| chncxr | resnet50 | gaussian_noise_dataset_stats | normalized_deletion_auc | LIME > KernelSHAP > Ablation > RISE |
| chncxr | resnet50 | normalized_zero | normalized_insertion_auc | LIME > KernelSHAP > Ablation > RISE |
| chncxr | resnet50 | normalized_zero | normalized_deletion_auc | LIME > KernelSHAP > Ablation > RISE |
| chncxr | vgg13 | dataset_mean_color | normalized_insertion_auc | LIME > Ablation > KernelSHAP > RISE |
| chncxr | vgg13 | dataset_mean_color | normalized_deletion_auc | LIME > KernelSHAP > Ablation > RISE |
| chncxr | vgg13 | dataset_mean_image | normalized_insertion_auc | KernelSHAP > LIME > Ablation > RISE |
| chncxr | vgg13 | dataset_mean_image | normalized_deletion_auc | KernelSHAP > LIME > Ablation > RISE |
| chncxr | vgg13 | gaussian_blur_sigma10 | normalized_insertion_auc | Ablation > KernelSHAP > LIME > RISE |
| chncxr | vgg13 | gaussian_blur_sigma10 | normalized_deletion_auc | LIME > Ablation > KernelSHAP > RISE |
| chncxr | vgg13 | gaussian_noise_dataset_stats | normalized_insertion_auc | KernelSHAP > LIME > Ablation > RISE |
| chncxr | vgg13 | gaussian_noise_dataset_stats | normalized_deletion_auc | KernelSHAP > LIME > Ablation > RISE |
| chncxr | vgg13 | normalized_zero | normalized_insertion_auc | LIME > KernelSHAP > Ablation > RISE |
| chncxr | vgg13 | normalized_zero | normalized_deletion_auc | LIME > KernelSHAP > Ablation > RISE |
| imagenet | resnet50 | dataset_mean_color | normalized_insertion_auc | LIME > KernelSHAP > Ablation > RISE |
| imagenet | resnet50 | dataset_mean_color | normalized_deletion_auc | LIME > KernelSHAP > RISE > Ablation |
| imagenet | resnet50 | dataset_mean_image | normalized_insertion_auc | LIME > KernelSHAP > Ablation > RISE |
| imagenet | resnet50 | dataset_mean_image | normalized_deletion_auc | LIME > KernelSHAP > RISE > Ablation |
| imagenet | resnet50 | gaussian_blur_sigma10 | normalized_insertion_auc | LIME > KernelSHAP > Ablation > RISE |
| imagenet | resnet50 | gaussian_blur_sigma10 | normalized_deletion_auc | LIME > KernelSHAP > RISE > Ablation |
| imagenet | resnet50 | gaussian_noise_dataset_stats | normalized_insertion_auc | LIME > KernelSHAP > RISE > Ablation |
| imagenet | resnet50 | gaussian_noise_dataset_stats | normalized_deletion_auc | LIME > KernelSHAP > RISE > Ablation |
| imagenet | resnet50 | normalized_zero | normalized_insertion_auc | LIME > KernelSHAP > Ablation > RISE |
| imagenet | resnet50 | normalized_zero | normalized_deletion_auc | LIME > KernelSHAP > RISE > Ablation |
| imagenet | vgg13 | dataset_mean_color | normalized_insertion_auc | LIME > KernelSHAP > Ablation > RISE |
| imagenet | vgg13 | dataset_mean_color | normalized_deletion_auc | LIME > KernelSHAP > Ablation > RISE |
| imagenet | vgg13 | dataset_mean_image | normalized_insertion_auc | LIME > KernelSHAP > Ablation > RISE |
| imagenet | vgg13 | dataset_mean_image | normalized_deletion_auc | LIME > KernelSHAP > Ablation > RISE |
| imagenet | vgg13 | gaussian_blur_sigma10 | normalized_insertion_auc | LIME > KernelSHAP > Ablation > RISE |
| imagenet | vgg13 | gaussian_blur_sigma10 | normalized_deletion_auc | LIME > KernelSHAP > Ablation > RISE |
| imagenet | vgg13 | gaussian_noise_dataset_stats | normalized_insertion_auc | Ablation > KernelSHAP > LIME > RISE |
| imagenet | vgg13 | gaussian_noise_dataset_stats | normalized_deletion_auc | LIME > KernelSHAP > Ablation > RISE |
| imagenet | vgg13 | normalized_zero | normalized_insertion_auc | LIME > KernelSHAP > Ablation > RISE |
| imagenet | vgg13 | normalized_zero | normalized_deletion_auc | LIME > KernelSHAP > Ablation > RISE |
