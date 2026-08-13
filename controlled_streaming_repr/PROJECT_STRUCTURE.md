# 项目结构说明

本项目把可靠重建、token 抽取、token 适配和后续 controlled/probe 分析分开组织。当前可执行主线以 Lingbot-map 官方 demo 路径为准。

## 根目录文件

- `README.md`：当前实验主线、运行命令和 token 抽取说明。
- `PROJECT_STRUCTURE.md`：解释目录职责和保留边界。

## `configs/`

轻量配置文件，不直接触发大模型加载。

- `paths.yaml`：项目级路径配置。
- `model_lingbot.yaml`：Lingbot-map repo、checkpoint 和 token 输出配置。
- `model_vggt.yaml`：保留为后续非流式对照模型配置，不是当前主线。
- `experiment_template.yaml`：新 controlled 实验配置模板。
- `controlled_scannet_v1.yaml`：第一组 ScanNet controlled sequence 和 extraction 配置。
- `analysis_controlled_v1.yaml`：可插拔 analysis method 配置。

## `third_party/`

通过软链接指向主实验目录下已有源码仓库：

- `third_party/lingbot-map -> ../../lingbot-map`
- `third_party/vggt -> ../../vggt`

这里不复制源码，也不移动 checkpoint。

## `data/`

保存受控实验的输入帧池和轻量数据工具。Raw ScanNet 数据不放进本项目，生成的实验 manifest 也不写进 `data/`。

### `data/controlled_sequences/`

输入数据池，保存已经准备好的 scene/frame 文件或软链接，例如固定 stride 抽帧序列。这里可以放图片、depth、pose、intrinsic 的小规模派生输入或软链接，但不保存由实验 setting 生成的 JSON manifest。

当前保留的默认输入序列：

```text
data/controlled_sequences/scene0000_00_lingbot_stream512_stride4/images
```

### `data/manifest_utils/`

manifest schema、ScanNet reader、pose utils 等工具代码：

- `manifest_schema.py`：统一 controlled manifest 和 index schema。
- `scannet_reader.py`：扫描 raw ScanNet scene，按数字 frame id 排序并跳过无效 pose。
- `pose_utils.py`：读取/校验 pose，并计算 translation/rotation distance。

## `sequence_settings/`

受控序列生成插件，只负责把 frame list 组合成 repeated/two-state/order 等 manifest object：

- `repeated_frame.py`：生成 `A, A, A, ...`。
- `two_state_alternation.py`：生成 `A, B, A, B, ...` 和 `A, Z, A, Z, ...`。
- `order_perturbation.py`：生成 normal/reverse/shuffle。

这里不运行模型、不保存 token、不画图、不做 PCA/Gram/attention 分析。

## `adapters/`

统一模型输出到 `TokenBundle`。

- `token_schema.py`：统一 token 数据结构。
- `base_adapter.py`：通用 adapter 抽象接口。
- `lingbot_adapter.py`：把官方 Lingbot token 输出适配成 `TokenBundle`，也能构造/运行官方 token 抽取命令。
- `vggt_adapter.py`：保留为后续 VGGT 对照接口。

## `analysis/`

底层数学工具函数，例如 PCA、Gram、RSA/CKA、token displacement 和 IO 工具。这里应只依赖数组形状和数值含义，不依赖 Lingbot-map 内部 forward 细节，也不负责遍历实验条件。

## `analysis_methods/`

可插拔分析方法，读取 `token_bundle + manifest` 后生成 metrics 和 figures：

- `frame_gram`
- `token_drift`
- `shared_pca_patchmap`
- `patch_affinity`
- `attention_mass`
- `geometry_stability`

## `experiments/`

实验说明、科学问题和推荐运行配置，不放核心代码。按照科学问题组织实验接口，而不是按照某个分析工具组织。

- `controlled_scannet_v1/`
- `exp01_history_conditioning/`
- `exp02_order_sensitivity/`
- `exp03_geometry_appearance_conflict/`
- `exp_template/`

新增实验可以使用：

```bash
python scripts/make_experiment_folder.py --name exp04_xxx
```

## `scripts/`

当前保留的命令行入口：

- `check_project_paths.py`：检查仓库软链接、输出目录和基础路径。
- `check_data_paths.py`：检查外部 ScanNet raw 数据位置。
- `run_lingbot_official_geometry.py`：复用官方 demo 路径，保存几何结果。
- `run_lingbot_official_tokens.py`：复用官方 demo 路径，捕获中间层 tokens。
- `extract_lingbot_tokens.py`：公共 token 抽取/适配入口，调用官方 token 脚本并生成 `token_bundle.json`。
- `build_sequence_manifests.py`：调用 ScanNet reader 和 sequence setting registry，输出 JSON manifests。
- `run_controlled_extraction.py`：读取 manifest index，调用 model adapters 保存 token bundles。
- `run_controlled_analysis.py`：读取 token bundles 和 manifests，调用 analysis method registry。
- `run_controlled_pipeline.py`：三步流程的方便入口，不承载核心算法逻辑。
- `run_analysis_template.py`：分析方法的轻量调用模板。
- `make_experiment_folder.py`：创建新的 controlled 实验接口目录。

脚本是命令行入口，不放核心算法逻辑。

## `outputs/`

统一保存实验产物：

- `reconstruction_official/`：官方路径输出的几何结果。
- `manifests/`：`build_sequence_manifests.py` 生成的 JSON manifest 输出目录，例如 `outputs/manifests/controlled_scannet_v1/`。
- `tokens/`：token `.npy`、metadata 和 `token_bundle.json`。
- `figures/`：可视化结果。
- `metrics/`：JSON/CSV 指标。
- `logs/`：运行日志。
