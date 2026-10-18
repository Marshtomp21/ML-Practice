# M4 实验 3：遮蔽基线敏感性

## 预先固定的协议

在 M4 configuration-selection 样本上，对 Ablation、LIME、RISE、KernelSHAP 分别使用同一套默认配置重新生成归因。解释阶段的遮蔽基线为实验自变量；Insertion/Deletion 评价仍固定使用项目主协议的 normalized-zero baseline，因此结果估计的是解释基线改变的影响。

- 样本清单：`data/splits/m4_frozen_20260917.json` 的 `configuration_selection`；图像为配对分析单位。
- 默认参数：SLIC=100、预算=1024、mask rate=0.5、RISE grid=7、随机种子=[0, 1, 2, 3, 4]。
- normalized zero：模型归一化空间中的全零张量（RGB 空间对应数据预处理均值）。
- dataset mean image：configuration-selection 图像逐像素 RGB 均值图。
- dataset mean color：configuration-selection 图像所有像素的逐通道 RGB 均值常量图。
- blur：在 RGB [0,1] 空间对每张原图作 Gaussian blur，sigma=10.0 像素。
- noise：用冻结 configuration-selection 池的逐通道 RGB 均值/标准差采样高斯噪声图，裁剪至 [0,1]；种子由样本 ID 哈希固定。
- 对随机方法跨基线复用同一组种子；保留实际归因前向次数、耗时与定位指标。

## 配对结果

逐基线中位数、四分位数见 `summary.json`；相对 normalized-zero 的逐图配对差值中位数和 bootstrap 95% CI 见 `paired_contrasts.json`。Bootstrap 以图像为单位，重复次数 10000，种子 20261008。

Insertion 越高、Deletion 越低通常越好；定位能量越高越好；Pointing Game 越高越好。
这些基线结果是 M4 的补充控制分析，不改变主实验统一使用 normalized-zero 评价基线的口径。

## 各解释基线的指标中位数

| 域 | 模型 | 方法 | 解释基线 | 图像数 | Normalized insertion | Normalized deletion | 归因秒数 | 前向数 |
|---|---|---|---|---:|---:|---:|---:|---:|
| imagenet | resnet50 | ablation | normalized_zero | 75 | 0.9451 | 0.8268 | 0.1974 | 88.0000 |
| imagenet | resnet50 | ablation | dataset_mean_image | 75 | 0.9413 | 0.8545 | 0.1905 | 88.0000 |
| imagenet | resnet50 | ablation | dataset_mean_color | 75 | 0.9330 | 0.8345 | 0.1857 | 88.0000 |
| imagenet | resnet50 | ablation | gaussian_blur_sigma10 | 75 | 0.9539 | 0.8553 | 0.1856 | 88.0000 |
| imagenet | resnet50 | ablation | gaussian_noise_dataset_stats | 75 | 0.8886 | 0.8491 | 0.1880 | 88.0000 |
| imagenet | resnet50 | lime | normalized_zero | 75 | 1.0097 | 0.3802 | 2.9432 | 1024.0000 |
| imagenet | resnet50 | lime | dataset_mean_image | 75 | 0.9863 | 0.3396 | 2.9124 | 1024.0000 |
| imagenet | resnet50 | lime | dataset_mean_color | 75 | 1.0030 | 0.3637 | 2.9156 | 1024.0000 |
| imagenet | resnet50 | lime | gaussian_blur_sigma10 | 75 | 0.9811 | 0.3746 | 2.9177 | 1024.0000 |
| imagenet | resnet50 | lime | gaussian_noise_dataset_stats | 75 | 0.9810 | 0.3270 | 2.9199 | 1024.0000 |
| imagenet | resnet50 | rise | normalized_zero | 75 | 0.9257 | 0.6997 | 5.2507 | 1024.0000 |
| imagenet | resnet50 | rise | dataset_mean_image | 75 | 0.9260 | 0.6921 | 5.2722 | 1024.0000 |
| imagenet | resnet50 | rise | dataset_mean_color | 75 | 0.9258 | 0.6993 | 5.2194 | 1024.0000 |
| imagenet | resnet50 | rise | gaussian_blur_sigma10 | 75 | 0.9228 | 0.7003 | 5.3327 | 1024.0000 |
| imagenet | resnet50 | rise | gaussian_noise_dataset_stats | 75 | 0.9306 | 0.6764 | 5.2767 | 1024.0000 |
| imagenet | resnet50 | kernel_shap | normalized_zero | 75 | 0.9891 | 0.3908 | 2.9269 | 1024.0000 |
| imagenet | resnet50 | kernel_shap | dataset_mean_image | 75 | 0.9849 | 0.4521 | 2.8864 | 1024.0000 |
| imagenet | resnet50 | kernel_shap | dataset_mean_color | 75 | 0.9878 | 0.3795 | 2.8980 | 1024.0000 |
| imagenet | resnet50 | kernel_shap | gaussian_blur_sigma10 | 75 | 0.9725 | 0.5087 | 2.9355 | 1024.0000 |
| imagenet | resnet50 | kernel_shap | gaussian_noise_dataset_stats | 75 | 0.9756 | 0.4678 | 2.9108 | 1024.0000 |
| imagenet | vgg13 | ablation | normalized_zero | 75 | 0.7998 | 0.2288 | 0.2091 | 88.0000 |
| imagenet | vgg13 | ablation | dataset_mean_image | 75 | 0.7684 | 0.2344 | 0.2033 | 88.0000 |
| imagenet | vgg13 | ablation | dataset_mean_color | 75 | 0.8002 | 0.2134 | 0.2129 | 88.0000 |
| imagenet | vgg13 | ablation | gaussian_blur_sigma10 | 75 | 0.7461 | 0.2308 | 0.2045 | 88.0000 |
| imagenet | vgg13 | ablation | gaussian_noise_dataset_stats | 75 | 0.7914 | 0.2108 | 0.2020 | 88.0000 |
| imagenet | vgg13 | lime | normalized_zero | 75 | 0.8856 | 0.0902 | 3.0287 | 1024.0000 |
| imagenet | vgg13 | lime | dataset_mean_image | 75 | 0.8594 | 0.0962 | 2.9898 | 1024.0000 |
| imagenet | vgg13 | lime | dataset_mean_color | 75 | 0.8828 | 0.0909 | 3.0531 | 1024.0000 |
| imagenet | vgg13 | lime | gaussian_blur_sigma10 | 75 | 0.8363 | 0.1148 | 3.0397 | 1024.0000 |
| imagenet | vgg13 | lime | gaussian_noise_dataset_stats | 75 | 0.7569 | 0.1238 | 3.0506 | 1024.0000 |
| imagenet | vgg13 | rise | normalized_zero | 75 | 0.5736 | 0.2909 | 5.3096 | 1024.0000 |
| imagenet | vgg13 | rise | dataset_mean_image | 75 | 0.5725 | 0.2951 | 5.2793 | 1024.0000 |
| imagenet | vgg13 | rise | dataset_mean_color | 75 | 0.5760 | 0.2909 | 5.3402 | 1024.0000 |
| imagenet | vgg13 | rise | gaussian_blur_sigma10 | 75 | 0.5748 | 0.2956 | 5.3262 | 1024.0000 |
| imagenet | vgg13 | rise | gaussian_noise_dataset_stats | 75 | 0.6049 | 0.2616 | 5.2303 | 1024.0000 |
| imagenet | vgg13 | kernel_shap | normalized_zero | 75 | 0.8757 | 0.1001 | 3.0346 | 1024.0000 |
| imagenet | vgg13 | kernel_shap | dataset_mean_image | 75 | 0.8450 | 0.1067 | 3.0666 | 1024.0000 |
| imagenet | vgg13 | kernel_shap | dataset_mean_color | 75 | 0.8670 | 0.1042 | 3.0733 | 1024.0000 |
| imagenet | vgg13 | kernel_shap | gaussian_blur_sigma10 | 75 | 0.7878 | 0.1185 | 3.0451 | 1024.0000 |
| imagenet | vgg13 | kernel_shap | gaussian_noise_dataset_stats | 75 | 0.7677 | 0.1422 | 3.0944 | 1024.0000 |
| chncxr | resnet50 | ablation | normalized_zero | 86 | 1.3799 | -0.9773 | 0.2061 | 92.0000 |
| chncxr | resnet50 | ablation | dataset_mean_image | 86 | 0.9951 | -0.4903 | 0.2032 | 92.0000 |
| chncxr | resnet50 | ablation | dataset_mean_color | 86 | 1.1660 | -0.6873 | 0.2028 | 92.0000 |
| chncxr | resnet50 | ablation | gaussian_blur_sigma10 | 86 | 0.9125 | -0.4304 | 0.2066 | 92.0000 |
| chncxr | resnet50 | ablation | gaussian_noise_dataset_stats | 86 | 1.1100 | -0.5478 | 0.1938 | 92.0000 |
| chncxr | resnet50 | lime | normalized_zero | 86 | 1.7410 | -1.5802 | 2.9045 | 1024.0000 |
| chncxr | resnet50 | lime | dataset_mean_image | 86 | 0.9641 | -0.6506 | 2.8803 | 1024.0000 |
| chncxr | resnet50 | lime | dataset_mean_color | 86 | 1.4905 | -1.1760 | 2.9179 | 1024.0000 |
| chncxr | resnet50 | lime | gaussian_blur_sigma10 | 86 | 1.0310 | -0.6159 | 2.9346 | 1024.0000 |
| chncxr | resnet50 | lime | gaussian_noise_dataset_stats | 86 | 1.1449 | -0.6401 | 2.8632 | 1024.0000 |
| chncxr | resnet50 | rise | normalized_zero | 86 | 0.1745 | 0.2406 | 5.1271 | 1024.0000 |
| chncxr | resnet50 | rise | dataset_mean_image | 86 | 0.3801 | 0.1900 | 5.1228 | 1024.0000 |
| chncxr | resnet50 | rise | dataset_mean_color | 86 | 0.1506 | 0.2933 | 5.1319 | 1024.0000 |
| chncxr | resnet50 | rise | gaussian_blur_sigma10 | 86 | 0.3637 | 0.1652 | 5.1141 | 1024.0000 |
| chncxr | resnet50 | rise | gaussian_noise_dataset_stats | 86 | 0.1388 | 0.2966 | 5.0931 | 1024.0000 |
| chncxr | resnet50 | kernel_shap | normalized_zero | 86 | 1.7014 | -1.4167 | 2.8283 | 1024.0000 |
| chncxr | resnet50 | kernel_shap | dataset_mean_image | 86 | 0.9172 | -0.6567 | 2.8611 | 1024.0000 |
| chncxr | resnet50 | kernel_shap | dataset_mean_color | 86 | 1.4634 | -1.1312 | 2.8603 | 1024.0000 |
| chncxr | resnet50 | kernel_shap | gaussian_blur_sigma10 | 86 | 1.0662 | -0.6500 | 2.8670 | 1024.0000 |
| chncxr | resnet50 | kernel_shap | gaussian_noise_dataset_stats | 86 | 1.1315 | -0.6144 | 2.8415 | 1024.0000 |
| chncxr | vgg13 | ablation | normalized_zero | 86 | 3.6075 | -1.0448 | 0.2054 | 92.0000 |
| chncxr | vgg13 | ablation | dataset_mean_image | 86 | 1.3300 | -0.0400 | 0.2033 | 92.0000 |
| chncxr | vgg13 | ablation | dataset_mean_color | 86 | 2.3842 | -0.4430 | 0.2075 | 92.0000 |
| chncxr | vgg13 | ablation | gaussian_blur_sigma10 | 86 | 1.4152 | 0.0153 | 0.2027 | 92.0000 |
| chncxr | vgg13 | ablation | gaussian_noise_dataset_stats | 86 | 2.7172 | -0.5899 | 0.2104 | 92.0000 |
| chncxr | vgg13 | lime | normalized_zero | 86 | 4.5414 | -2.7602 | 3.0424 | 1024.0000 |
| chncxr | vgg13 | lime | dataset_mean_image | 86 | 1.4470 | -0.2920 | 3.0317 | 1024.0000 |
| chncxr | vgg13 | lime | dataset_mean_color | 86 | 2.3911 | -0.9341 | 3.0574 | 1024.0000 |
| chncxr | vgg13 | lime | gaussian_blur_sigma10 | 86 | 1.1636 | 0.0049 | 3.0380 | 1024.0000 |
| chncxr | vgg13 | lime | gaussian_noise_dataset_stats | 86 | 3.1644 | -1.5002 | 3.0115 | 1024.0000 |
| chncxr | vgg13 | rise | normalized_zero | 86 | 1.0663 | 0.0836 | 5.2967 | 1024.0000 |
| chncxr | vgg13 | rise | dataset_mean_image | 86 | 0.9157 | 0.2369 | 5.2834 | 1024.0000 |
| chncxr | vgg13 | rise | dataset_mean_color | 86 | 0.7458 | 0.2978 | 5.2704 | 1024.0000 |
| chncxr | vgg13 | rise | gaussian_blur_sigma10 | 86 | 0.8168 | 0.2715 | 5.2565 | 1024.0000 |
| chncxr | vgg13 | rise | gaussian_noise_dataset_stats | 86 | 1.2624 | 0.0233 | 5.2648 | 1024.0000 |
| chncxr | vgg13 | kernel_shap | normalized_zero | 86 | 4.4164 | -2.7571 | 2.9446 | 1024.0000 |
| chncxr | vgg13 | kernel_shap | dataset_mean_image | 86 | 1.5553 | -0.2944 | 2.9808 | 1024.0000 |
| chncxr | vgg13 | kernel_shap | dataset_mean_color | 86 | 2.0778 | -0.8036 | 2.9764 | 1024.0000 |
| chncxr | vgg13 | kernel_shap | gaussian_blur_sigma10 | 86 | 1.1755 | 0.0301 | 2.9995 | 1024.0000 |
| chncxr | vgg13 | kernel_shap | gaussian_noise_dataset_stats | 86 | 3.2496 | -1.5224 | 2.9796 | 1024.0000 |
