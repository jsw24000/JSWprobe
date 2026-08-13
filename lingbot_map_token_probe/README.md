# Lingbot-map Token Probe

这个目录是 Lingbot-map 的并列实验项目，不修改原仓库代码。当前检测到的 Lingbot-map 源码路径是：

```bash
/home/3dsm/Desktop/JSWprobe/lingbot-map
```

默认 checkpoint 使用：

```bash
../lingbot-map/checkpoints/lingbot-map.pt
```

这台主机可见的数据盘挂载在 `/home/data1`。脚本会把用户常用的 `/disk1/...` 和 `/data1/...` 自动映射到 `/home/data1/...`。当前可见：

- `/home/data1/3dsm/S2VGGT/processed_scannetpp_v2/processed_scannetpp_v2.combined.zip`
- `/home/data1/ScanNet/scans/scene0000_00` 等 ScanNet scene

`/home/data1/ScanNet/scans` 里的 scene 目前没有展开的 `color/` 或 `rgb/` 目录，但有 `.sens` 文件；`extract_tokens.py` 支持直接从 `.sens` 中按采样 index 抽取 RGB 帧。

## 实验目的

这是一个 training-free qualitative probing 实验。它用 Lingbot-map checkpoint 对 ScanNet 图像序列跑前向到 aggregator token 输出，抽取 image patch tokens，然后做可视化：

- PCA RGB token map，看 token 主方差方向中是否有物体轮廓、区域边界、纹理/几何边界。
- DINO backbone patch token map，作为进入第一层 aggregator 前的对照。
- 不同 stage 的 token map 差异。
- 同一 scene 不同帧的 PCA 颜色是否有跨帧一致性。
- 用 token cosine similarity 做简单跨帧对应热力图。

脚本只保存空间 patch tokens。Lingbot-map 的 camera/register/scale special tokens 会通过 `patch_start_idx` 被切掉，不混进 PCA 主图。

## 为什么用 PCA RGB

每个 patch token 是高维向量，PCA 把一个 scene 内同一 stage 的所有帧 token 合在一起 fit 到 3 维，再映射成 RGB。这样可以把高维特征里最大的几个变化方向投到图像空间，方便肉眼观察空间结构。

默认不能每帧单独 fit PCA。每帧单独 fit 会让每张图的颜色坐标系都不同，跨帧颜色没有可比性。本实验默认对同一 scene、同一 stage 的所有选中帧共同 fit PCA，再把同一个 projection 用到每一帧。

## 输出怎么看

一次运行会生成：

```bash
outputs/<scene>/
  frames/                         # 抽样 RGB 帧
  tokens/backbone_tokens.pt        # DINO patch tokens before aggregator blocks
  tokens/stage_XX_tokens.pt        # [T, H_token, W_token, C]
  pca_vis/backbone/                # DINO backbone PCA RGB
  pca_vis/stage_XX/                # 每个 stage 的 PCA RGB
  pca_vis/comparison/              # RGB | backbone | stage_00 | stage_01 | ...
  pca_vis_controls/shuffled_tokens/
  pca_vis_controls/random_projection/
  correspondence_vis/
  metadata.json
  summary.md
```

`pca_vis/comparison/*_all_stages.png` 是最方便看的主图。`shuffled_tokens` control 会打乱每帧 token 的空间位置，如果空间结构消失，说明原始 token map 的结构不是可视化流程凭空制造的。`random_projection` control 用固定随机矩阵降到 3 维，如果它也能看出轮廓，说明 token 本身空间结构很强；如果主要 PCA 明显，说明结构更集中在主成分方向。

`correspondence_vis` 中每张图左边是 source RGB 和 query token，右边是 target frame 上所有 token 对该 query 的 cosine similarity heatmap，中间会标出 target 中 similarity 最大的位置。

## 能说明什么

如果 PCA RGB 图能看到物体轮廓或区域边界，只能说明 Lingbot-map image tokens 的主方差方向中存在空间结构。跨帧颜色一致性是一种 qualitative clue，提示某些特征方向可能在不同视角中保持相似。

## 不能说明什么

这个实验不能直接说明模型具备实例分割能力，也不能给出严格的语义、几何或 correspondence 定量结论。更严格的实验需要继续做 token affinity、nearest-neighbor correspondence、linear probe、边界预测或 segmentation evaluation。

## 运行

建议用 Lingbot-map 环境：

```bash
cd /home/3dsm/Desktop/JSWprobe/lingbot_map_token_probe
conda activate lingbot-map
./run_probe.sh
```

也可以显式指定路径：

```bash
cd /home/3dsm/Desktop/JSWprobe/lingbot_map_token_probe
PYTHON=/home/3dsm/miniconda3/envs/lingbot-map/bin/python \
LINGBOT_ROOT=../lingbot-map \
DATA_ROOT=/disk1/3dsm/S2VGGT \
SCENE=scene0000_00 \
NUM_FRAMES=12 \
FRAME_STRIDE=10 \
./run_probe.sh
```

单步运行：

```bash
python scripts/find_scannet_scenes.py --data-root /disk1/3dsm/S2VGGT

python scripts/extract_tokens.py \
  --lingbot-root ../lingbot-map \
  --data-root /disk1/3dsm/S2VGGT \
  --scene scene0000_00 \
  --num-frames 12 \
  --frame-stride 10 \
  --output outputs/scene0000_00

python scripts/visualize_pca_tokens.py \
  --input outputs/scene0000_00 \
  --output outputs/scene0000_00/pca_vis \
  --fit-scope scene

python scripts/visualize_correspondence.py \
  --input outputs/scene0000_00 \
  --stage auto \
  --src-frame 0 \
  --tgt-frame 5 \
  --output outputs/scene0000_00/correspondence_vis
```

`--fit-scope global` 也支持：把 `--input` 指向包含多个 scene 输出的父目录，会对多个 scene 的同名 stage 共同 fit PCA，再分别输出可视化。默认还是 scene-level PCA。

## Noise-Slot SVD Debias

新增的 `noise_slot_svd` 实验会估计并移除 token feature 中与二维位置相关的低维子空间。它不会覆盖旧的 PCA、control 或 raw correspondence 输出；所有结果都写入：

```bash
outputs/<scene>/05_noise_slot_svd_debias/
```

默认配置在：

```bash
configs/noise_slot_svd_debias.yaml
```

默认快速设置使用 6 个 slot、5 种 low-semantic replacement、2 个 seed，也就是 60 次 Lingbot-map aggregator forward。运行：

```bash
cd /home/3dsm/Desktop/JSWprobe/lingbot_map_token_probe
conda activate lingbot-map
bash run_noise_slot_svd_debias.sh
```

或指定 Python：

```bash
PYTHON=/home/3dsm/miniconda3/envs/lingbot-map/bin/python \
bash run_noise_slot_svd_debias.sh
```

pipeline 步骤：

1. `generate_noise_slot_samples.py`: 每次只替换一个真实时间步为 gaussian / smooth_noise / gray / hgrad / vgrad，重新跑 Lingbot-map，只保存被替换 slot 的 tokens。
2. `estimate_noise_slot_subspace.py`: 对 noise-slot tokens 按空间位置平均，做 SVD，保存 K=1/2/4/8/16 的 feature-space basis。
3. `visualize_noise_slot_basis.py`: 可视化 basis 在 mean noise-slot field 和真实帧上的 score maps。
4. `visualize_correspondence_noise_debiased.py`: 在真实 tokens 上比较 raw vs orthogonal-complement debiased cosine correspondence。
5. `evaluate_noise_debiased_correspondence.py`: 输出无 GT 的 position-bias diagnostics。
6. `make_noise_slot_debias_summary.py`: 写最终 summary。

最终报告在：

```bash
outputs/<scene>/05_noise_slot_svd_debias/06_summary/debias_summary.md
```

If the machine feels unstable, run smaller configs first:

```bash
# 1 forward, one stage, smoke test only
CONFIG=configs/noise_slot_svd_debias_tiny.yaml bash run_noise_slot_svd_debias.sh

# 9 forwards, stage_02/stage_03 only, useful first pass
CONFIG=configs/noise_slot_svd_debias_light.yaml bash run_noise_slot_svd_debias.sh

# 60 forwards, all stages, larger output
CONFIG=configs/noise_slot_svd_debias.yaml bash run_noise_slot_svd_debias.sh
```

The noise-slot token sets are stored as `float16` by default to reduce RAM and disk usage; SVD estimation converts them back to `float32`.
