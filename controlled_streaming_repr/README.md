# controlled_streaming_repr

本项目用于组织 Lingbot-map 流式三维重建表征的受控实验。主线流程是：从 ScanNet raw scene 生成受控序列 manifest，按 manifest 调用官方 Lingbot-map 路径抽取 tokens 和 geometry，再用统一的 `TokenBundle` schema 进行可插拔分析。

目标不是直接证明某个 token 等价于下游语义任务，而是观察在输入流被严格控制时，表示、attention、memory/cache 状态和几何输出如何变化。

## 项目架构

`configs/` 保存轻量配置，不直接运行模型。`controlled_scannet_v1.yaml` 定义数据源、序列 setting 和 Lingbot-map extraction 参数；`analysis_controlled_v1.yaml` 定义要运行的 analysis methods 及可视化参数。

`sequence_settings/` 只负责生成 manifest。每个 setting 读取 ScanNet frame list，组合出受控输入序列，但不运行模型、不保存 token、不画图。

`data/manifest_utils/` 提供 ScanNet reader、pose 校验、manifest schema 等工具。Raw ScanNet 数据不放进本项目，默认由 `SCANNET_RAW_ROOT` 或配置里的 `scannet_root` 指向外部路径。

`adapters/` 把模型输出适配成统一 `TokenBundle`。当前可执行主线是 `LingbotAdapter`，它复用官方 Lingbot-map demo 路径抽取中间层 tokens。

`scripts/` 是命令行入口。核心 pipeline 是：

- `build_sequence_manifests.py`：生成 controlled sequence manifests。
- `run_controlled_extraction.py`：按 manifest 抽取 tokens 并保存 `token_bundle.json`。
- `run_controlled_geometry.py`：按 manifest 调用官方 Lingbot-map geometry 路径，并把 geometry 输出挂回 bundle。
- `run_controlled_analysis.py`：读取 manifest 和 bundle，运行 analysis methods。
- `run_controlled_pipeline.py`：build/extraction/analysis 的便捷包装。

`analysis_methods/` 是可插拔分析方法。每个方法读取 `TokenBundle + manifest`，输出 metrics JSON 和 figures PNG。

`outputs/` 保存所有实验产物，包括 manifests、controlled image symlinks、tokens、geometry、metrics、figures 和 logs。

## 命名规范

项目名来自配置：

```text
project.name = controlled_scannet_v1
```

setting 名使用 registry 中的 lower snake case：

```text
repeated_frame_stability
two_state_alternation
order_perturbation
```

condition id 由 setting 生成，要求稳定、可排序、可读：

```text
repeat_A_seq{seq_len}_anchor{anchor_number:03d}
abab_near_seq{seq_len}_pair{pair_number:03d}
azaz_far_seq{seq_len}_pair{pair_number:03d}
order_{normal|reverse|shuffle}_seq{seq_len}_seg{segment_number:03d}
```

输出路径统一使用：

```text
outputs/<artifact>/<project>/<model>/<scene_id>/<setting_name>/<condition_id>/
```

manifest 和 controlled image sequence 不带 model：

```text
outputs/manifests/<project>/<scene_id>/<setting_name>/<condition_id>.json
outputs/controlled_sequences/<project>/<scene_id>/<setting_name>/<condition_id>/images/
```

token layer 命名为：

```text
layer_04/patch_tokens
layer_11/patch_tokens
layer_17/patch_tokens
layer_23/patch_tokens
```

analysis 输出文件以 method name 为前缀，例如：

```text
token_drift.json
shared_pca_patchmap_layer_11_patch_tokens_t000.png
geometry_metrics.json
```

## 已实现的特殊序列配置

特殊序列只由 `configs/controlled_scannet_v1.yaml` 的 `sequence.settings` 控制。`build_sequence_manifests.py` 会扫描 enabled setting，为每个 scene 和 seq_len 生成 manifest。这个步骤只写 JSON manifest，不运行 Lingbot-map，也不生成 token 或 geometry。

当前配置示例：

```yaml
sequence:
  seq_lens: [32]
  settings:
    repeated_frame_stability:
      enabled: true
    two_state_alternation:
      enabled: false
    order_perturbation:
      enabled: false
```

命令行 `--seq-lens 16` 会临时覆盖 YAML 里的 `sequence.seq_lens`，所以 `repeat_A_seq16_anchor000/001/002` 是由命令行覆盖生成的。

### repeated_frame_stability

生成 `A, A, A, ...`。它重复同一 anchor frame，用来测试当前图像完全不变时，tokens、attention、memory/cache 和 geometry 是否仍会随流式状态演化。

配置入口：

```yaml
sequence:
  settings:
    repeated_frame_stability:
      enabled: true
      num_anchors_per_scene: 3
      anchor_strategy: "uniform_valid"
      avoid_boundary_ratio: 0.1
      require_valid_pose: true
```

逻辑：

- 从有效帧中均匀选择 `num_anchors_per_scene` 个 anchor。
- 每个 anchor 生成 `A, A, A, ...`。
- 每个时间步都指向同一张 RGB/depth/pose/intrinsic。
- condition id 形如 `repeat_A_seq16_anchor000`。

构造命令：

```bash
python scripts/build_sequence_manifests.py \
  --config configs/controlled_scannet_v1.yaml \
  --limit-scenes 1 \
  --seq-lens 16 \
  --overwrite
```

### two_state_alternation

生成 `A, B, A, B, ...` 和 `A, Z, A, Z, ...`。near pair 使用 `B`，far pair 使用 `Z`，用于观察表示更绑定当前图像内容、局部 pose 窗口，还是历史状态。

启用配置：

```yaml
sequence:
  settings:
    repeated_frame_stability:
      enabled: false
    two_state_alternation:
      enabled: true
      num_pairs_per_scene: 3
      near_gap_candidates: [5, 10, 15]
      far_min_index_gap: 80
      prefer_pose_distance_for_far: true
      far_min_translation: 0.8
      if_pose_distance_unavailable: "fallback_to_index_gap"
      require_valid_pose: true
      manual_pairs: []
```

逻辑：

- near pair 生成 `A, B, A, B, ...`，condition id 形如 `abab_near_seq16_pair000`。
- far pair 生成 `A, Z, A, Z, ...`，condition id 形如 `azaz_far_seq16_pair000`。
- near pair 优先使用 `near_gap_candidates` 中的帧间隔。
- far pair 优先使用 pose translation 距离；如果 pose 不可用，按配置 fallback 到 frame index gap。
- 如果 `manual_pairs` 非空，则跳过自动 pair 选择。每个条目可写 `A_frame_id` 和 `B_frame_id` 来生成手动 ABAB；也可额外写 `Z_frame_id` 生成手动 AZAZ。例如：

```yaml
two_state_alternation:
  enabled: true
  manual_pairs:
    - label: overlap_candidate
      A_frame_id: "0000"
      B_frame_id: "0080"
      emit_pair_types: ["near"]
```

构造命令同样使用：

```bash
python scripts/build_sequence_manifests.py \
  --config configs/controlled_scannet_v1.yaml \
  --limit-scenes 1 \
  --seq-lens 16 \
  --overwrite
```

### order_perturbation

从同一帧段生成 `normal`、`reverse` 和 deterministic `shuffle` 三种顺序。它用于测试相同帧集合在不同输入顺序下，tokens 和 geometry 输出是否变化。

启用配置：

```yaml
sequence:
  settings:
    repeated_frame_stability:
      enabled: false
    order_perturbation:
      enabled: true
      num_segments_per_scene: 3
      base_stride_candidates: [5, 10, 15]
      shuffle_seed: 2026
      shuffle_mode: "deterministic_interleave"
      require_valid_pose: true
```

逻辑：

- 从同一个原始 frame segment 构造三种顺序：`normal`、`reverse`、`shuffle`。
- 三个 condition 使用同一组 source frames，只改变输入顺序。
- condition id 形如 `order_normal_seq16_seg000`、`order_reverse_seq16_seg000`、`order_shuffle_seq16_seg000`。
- `shuffle` 当前是 deterministic interleave，不依赖随机运行状态。

构造命令：

```bash
python scripts/build_sequence_manifests.py \
  --config configs/controlled_scannet_v1.yaml \
  --limit-scenes 1 \
  --seq-lens 16 \
  --overwrite
```

如果多个 setting 同时 `enabled: true`，同一次 build 会生成所有 enabled setting 的 manifests。后续 extraction、geometry 和 analysis 再用 `--setting` 过滤。

## 已实现的分析方法

`frame_gram`：把每帧 token mean-pool 成 frame embedding，计算帧间余弦相似度矩阵。适合看全局表示稳定性。

`token_drift`：计算每帧 frame embedding 到第一帧和前一帧的 `1 - cosine` 距离。适合看流式状态是否随时间累积漂移。

`shared_pca_patchmap`：对同组条件、同一 layer 的 patch tokens 拟合共享 PCA，再把每帧 patch token 投影成 RGB 图。默认输出 pixel-level 上采样图；`upsample_mode` 可选 `nearest` 或 `bilinear`。

`patch_affinity`：选取少量帧对，计算 patch-to-patch 余弦 affinity 矩阵。适合看局部 patch 表示是否保持对应关系。

`attention_mass`：读取 bundle 中已保存的 attention summary，汇总并画 `mass_*`、`entropy` 或 `attention_entropy` 曲线。当前 extraction 脚本还需要额外 hook 才会产生 attention summary。

`geometry_stability`：读取 bundle 的 `geometry_outputs`，统计 camera translation step、depth variance、pointmap variance。需要先运行 `run_controlled_geometry.py` 或手动把 geometry 输出挂回 bundle。

## 分析配置逻辑

分析方法由 `configs/analysis_controlled_v1.yaml` 控制：

```yaml
analysis:
  methods:
    - "frame_gram"
    - "token_drift"
    - "shared_pca_patchmap"
    - "patch_affinity"
    - "attention_mass"
    - "geometry_stability"
```

运行时可以用 `--methods` 临时选择一部分方法，不需要改 YAML。例如只跑 drift 和 PCA：

```bash
python scripts/run_controlled_analysis.py \
  --experiment-config configs/controlled_scannet_v1.yaml \
  --analysis-config configs/analysis_controlled_v1.yaml \
  --manifest-index outputs/manifests/controlled_scannet_v1/index.json \
  --models lingbot-map \
  --methods token_drift shared_pca_patchmap \
  --setting repeated_frame_stability \
  --seq-len 16
```

不同 analysis 对输入字段的要求不同：

```text
frame_gram / token_drift / shared_pca_patchmap / patch_affinity
  需要 token_bundle.json 中的 layer_tokens。

geometry_stability
  需要 token_bundle.json 中的 geometry_outputs。
  通常先运行 run_controlled_geometry.py。

attention_mass
  需要 token_bundle.json 中的 attention summary。
  当前 Lingbot token extraction 还没有自动保存该字段。
```

对 `order_perturbation`，如果要让 `token_drift` 和 `patch_affinity` 计算 normal/reverse/shuffle 的 cross-condition 对照，不要加 `--condition-id`，要让同一个 segment 的多个 order variants 一起进入 group run：

```bash
python scripts/run_controlled_analysis.py \
  --experiment-config configs/controlled_scannet_v1.yaml \
  --analysis-config configs/analysis_controlled_v1.yaml \
  --manifest-index outputs/manifests/controlled_scannet_v1/index.json \
  --models lingbot-map \
  --methods token_drift patch_affinity shared_pca_patchmap \
  --setting order_perturbation \
  --seq-len 16
```

如果只想分析单个 condition，可以加 `--condition-id`；但这种情况下 order cross-condition 指标不会产生。

## 标准运行流程

以下命令从项目目录运行：

```bash
cd /home/3dsm/Desktop/JSWprobe/controlled_streaming_repr
```

### 1. 检查路径

```bash
python scripts/check_project_paths.py
python scripts/check_data_paths.py
```

### 2. 生成受控序列 Manifest

预览将要生成的 manifest：

```bash
python scripts/build_sequence_manifests.py \
  --config configs/controlled_scannet_v1.yaml \
  --limit-scenes 1 \
  --seq-lens 16 \
  --dry-run
```

写入 manifest：

```bash
python scripts/build_sequence_manifests.py \
  --config configs/controlled_scannet_v1.yaml \
  --limit-scenes 1 \
  --seq-lens 16 \
  --overwrite
```

输出位置：

```text
outputs/manifests/controlled_scannet_v1/index.json
outputs/manifests/controlled_scannet_v1/<scene>/<setting>/<condition>.json
```

### 3. 抽取 Tokens

抽取单个 condition：

```bash
python scripts/run_controlled_extraction.py \
  --config configs/controlled_scannet_v1.yaml \
  --manifest-index outputs/manifests/controlled_scannet_v1/index.json \
  --models lingbot-map \
  --setting repeated_frame_stability \
  --seq-len 16 \
  --condition-id repeat_A_seq16_anchor000
```

抽取该 setting 下所有 matching conditions：

```bash
python scripts/run_controlled_extraction.py \
  --config configs/controlled_scannet_v1.yaml \
  --manifest-index outputs/manifests/controlled_scannet_v1/index.json \
  --models lingbot-map \
  --setting repeated_frame_stability \
  --seq-len 16
```

输出位置：

```text
outputs/tokens/controlled_scannet_v1/lingbot-map/<scene>/<setting>/<condition>/
```

关键文件：

```text
metadata.json
layer_04_patch_tokens.npy
layer_11_patch_tokens.npy
layer_17_patch_tokens.npy
layer_23_patch_tokens.npy
token_bundle.json
token_summary.json
```

### 4. 抽取 Geometry

运行官方 Lingbot-map geometry 路径，并把结果挂回 `token_bundle.json`：

```bash
python scripts/run_controlled_geometry.py \
  --config configs/controlled_scannet_v1.yaml \
  --manifest-index outputs/manifests/controlled_scannet_v1/index.json \
  --models lingbot-map \
  --setting repeated_frame_stability \
  --seq-len 16 \
  --condition-id repeat_A_seq16_anchor000
```

如果需要 pointmap variance，额外保存 dense world points：

```bash
python scripts/run_controlled_geometry.py \
  --config configs/controlled_scannet_v1.yaml \
  --manifest-index outputs/manifests/controlled_scannet_v1/index.json \
  --models lingbot-map \
  --setting repeated_frame_stability \
  --seq-len 16 \
  --condition-id repeat_A_seq16_anchor000 \
  --save-dense-world-points
```

输出位置：

```text
outputs/reconstruction_official/controlled_scannet_v1/lingbot-map/<scene>/<setting>/<condition>/geometry/
```

关键文件：

```text
metadata.json
extrinsic.npy
intrinsic.npy
pose_enc.npy
depth.npy
depth_conf.npy
points_depth_camera_sample.ply
world_points_from_depth.npy  # only with --save-dense-world-points
```

### 5. 运行 Analysis

运行表示相关分析：

```bash
python scripts/run_controlled_analysis.py \
  --experiment-config configs/controlled_scannet_v1.yaml \
  --analysis-config configs/analysis_controlled_v1.yaml \
  --manifest-index outputs/manifests/controlled_scannet_v1/index.json \
  --models lingbot-map \
  --methods frame_gram token_drift shared_pca_patchmap patch_affinity \
  --setting repeated_frame_stability \
  --seq-len 16
```

运行 geometry 分析：

```bash
python scripts/run_controlled_analysis.py \
  --experiment-config configs/controlled_scannet_v1.yaml \
  --analysis-config configs/analysis_controlled_v1.yaml \
  --manifest-index outputs/manifests/controlled_scannet_v1/index.json \
  --models lingbot-map \
  --methods geometry_stability \
  --setting repeated_frame_stability \
  --seq-len 16
```

输出位置：

```text
outputs/metrics/controlled_scannet_v1/lingbot-map/<scene>/<setting>/<condition>/
outputs/figures/controlled_scannet_v1/lingbot-map/<scene>/<setting>/<condition>/
```

### 6. 便捷 Pipeline

如果只需要 build/extraction/analysis 三步，可以用：

```bash
python scripts/run_controlled_pipeline.py \
  --config configs/controlled_scannet_v1.yaml \
  --analysis-config configs/analysis_controlled_v1.yaml \
  --manifest-index outputs/manifests/controlled_scannet_v1/index.json \
  --models lingbot-map \
  --methods frame_gram token_drift shared_pca_patchmap patch_affinity \
  --setting repeated_frame_stability \
  --seq-len 16 \
  --overwrite-manifests
```

geometry 当前由 `run_controlled_geometry.py` 单独运行。

## 底层官方入口

`run_lingbot_official_geometry.py` 和 `run_lingbot_official_tokens.py` 是更底层的官方路径封装。通常优先使用 `run_controlled_extraction.py` 和 `run_controlled_geometry.py`；只有在调试单个 image folder 时才直接调用 official 脚本。

## 环境变量

Lingbot Python 环境可由配置或环境变量指定：

```bash
export LINGBOT_PYTHON=/home/3dsm/miniconda3/envs/lingbot-map/bin/python
```

ScanNet raw 数据路径可由环境变量覆盖：

```bash
export SCANNET_RAW_ROOT=/home/data1/3dsm/ScanNet/scans_extracted
```
