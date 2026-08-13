# Cross View Consistency

这是放在 `JSWprobe/cross_view_consistency/` 下的独立实验项目，用来评估不同 dense patch feature 在 ScanNet 跨视角匹配中的几何一致性。项目不修改 `lingbot-map`、官方 `dinov2` 或旧的 `Lingbot_map_pair_match` 代码。

## 实验目标

给定同一场景中的两帧图像，先用源帧 patch feature 在目标帧 patch feature 中做最近邻查找，再用 ScanNet 的 depth、pose 和 intrinsic 投影得到 GT 对应点，评估预测 patch 与 GT 的像素误差、token 误差和 top-k rank。

所有 backend 共用同一批帧、pair、query token、GT 对应关系和指标阈值，因此结果可以直接横向比较。

## 当前主要配置

最新主要实验使用 `configs/scene0002_16f_ropeoff.yaml`：

- 数据集：ScanNet `scene0002_00`
- 选帧：16 帧，`start_frame=0`，`stride=10`，实际覆盖 `0,10,...,150`
- 输入尺寸：RGB resize 到 `392 x 518`
- patch size：`14`
- token grid：`28 x 37`，每帧 `1036` 个 patch tokens
- 设备：CUDA
- 随机种子：`42`
- pair 采样：small/medium/large 每类最多 12 对
- 最新 16 帧结果实际得到：small 12 对、medium 12 对、large 3 对，共 27 对
- query：每 pair 最多 512 个 valid tokens，所有 backend 共享
- matching：L2 normalize 后做 cosine nearest neighbor，同时保留 `random` 和 `same_coord` baseline

## 路径

当前默认路径：

```yaml
scannet_root: /home/data1/3dsm/ScanNet/scans_extracted
lingbot_repo: /home/3dsm/Desktop/JSWprobe/lingbot-map
lingbot_checkpoint: /home/3dsm/Desktop/JSWprobe/lingbot-map/checkpoints/lingbot-map.pt
dinov2_repo: /home/3dsm/Desktop/JSWprobe/dinov2
dinov2_checkpoint: /home/3dsm/Desktop/JSWprobe/weights/dinov2/dinov2_vitl14_reg4_pretrain.pth
```

如果换机器或路径，优先修改 `configs/*.yaml` 中的 `paths` 字段。

## Feature Backends

项目目前支持四类表征：

- `public_dinov2_vitl14`：原始 public DINOv2 baseline，当前默认使用 ViT-L/14 register checkpoint。
- `lingbot_backbone_pre_agg_rope_off`：Lingbot-map checkpoint 中的 DINOv2 初始化 backbone，位于重建聚合器之前。
- `lingbot_single_frame_recon_rope_off`：每帧独立通过 Lingbot-map reconstruction blocks，清空 KV cache，不允许帧间信息交互。
- `lingbot_offline_global_32f_rope_off`：把当前选中的所有帧一起作为 scale frames 输入 `GCTStream._aggregate_features`，用来近似 offline/global 多帧 attention 路径。

Lingbot 相关 backend 构造模型时关闭 3D/video RoPE：`enable_3d_rope=False`，并关闭 global RoPE。这里没有移除 2D positional encoding 或 2D position bias。

## Global Attention 前后对比

`lingbot_offline_global_32f_rope_off` 会对 `stage0..stage3` 都输出三种 feature：

- `stage*`：Lingbot 原始 concat 输出，维度 `2048`
- `stage*_frame_pre_global`：global attention 前的 frame branch，维度 `1024`
- `stage*_global_post`：global attention 后的 global branch，维度 `1024`

因此单独研究 global attention 前后时，一共比较 `4 stages x 3 branches = 12` 种表征。backend 名字里保留了历史上的 `32f`，但实际帧数由 config 决定；在 16 帧配置下就是 16 帧一起进入该路径。

## 运行命令

使用 `lingbot-map` conda 环境。

只运行最新的 12 种 global pre/post 表征：

```bash
cd /home/3dsm/Desktop/JSWprobe/cross_view_consistency

/home/3dsm/miniconda3/envs/lingbot-map/bin/python scripts/run_probe.py \
  --config configs/scene0002_16f_ropeoff.yaml \
  --tag global_prepost_16f_stride10 \
  --backends lingbot_offline_global_32f_rope_off
```

运行四类 backend 的 16 帧对照实验：

```bash
cd /home/3dsm/Desktop/JSWprobe/cross_view_consistency

/home/3dsm/miniconda3/envs/lingbot-map/bin/python scripts/run_probe.py \
  --config configs/scene0002_16f_ropeoff.yaml \
  --tag full_16f_stride10_all4_ropeoff
```

查看某次输出：

```bash
/home/3dsm/miniconda3/envs/lingbot-map/bin/python scripts/inspect_outputs.py outputs/<run_id>
```

## 输出结构

每次实验会在 `outputs/<run_id>/` 下生成：

- `config_used.yaml`：本次实际使用的配置
- `selected_frames.csv`：选中的帧
- `pairs.csv`：采样到的 frame pairs
- `query_sets/`：每个 pair 共用的 query token
- `features/`：每个 backend 的 feature tensor
- `metrics_by_backend.csv`：按 bucket/backend/method 聚合的指标
- `metrics_by_pair.csv`：按 pair/backend/method 聚合的指标
- `query_matches.csv`：query 级别最近邻预测与 GT 对比
- `visualizations/`：pair overview、GT projection、match plot、heatmap 和 summary plot

如果运行被外部 watchdog 在可视化阶段中断，只要 `features/`、`metrics_by_backend.csv`、`metrics_by_pair.csv` 和 `query_matches.csv` 已经写完，核心能力对比仍然可用。

## 指标

- pixel recall：预测点距离 GT 投影点小于 `4/8/10/16/32/64 px` 的比例
- token recall：预测 token 距离 GT token 小于 `1/2/4` 个 grid cell 的比例
- error stats：像素误差和 token 误差的均值/中位数
- rank metrics：MRR、median GT rank、top-1/top-5/top-10 recall

## 已观察到的简要结论

在 `scene0002_00` 的 16 帧实验中：

- stage0 整体最适合 dense nearest-neighbor，尤其 small 视角。
- stage1 的 `global_post` 在 medium/large 视角上更强，是观察 global attention 增益的主要位置。
- stage2 基本进入平台期，pre/post/concat 差异较小。
- stage3 对 patch-level 几何匹配明显退化。

这些结论目前主要来自单个 ScanNet scene，后续如果要形成稳健结论，需要扩展到更多 scene 和更多 large-view pairs。
