# data

本目录只保存受控实验的输入数据池和轻量数据工具，不保存由
`build_sequence_manifests.py` 生成的实验 manifest。

## controlled_sequences

`data/controlled_sequences/` 是输入帧池，可以保存已经准备好的小规模
scene/frame 文件、软链接或输入序列元数据。不要把完整 raw ScanNet 数据复制
到这里。

## manifest_utils

`data/manifest_utils/` 保存 manifest schema、ScanNet reader 和 pose utils 等
工具代码。生成的 JSON manifest 输出到：

```text
outputs/manifests/<project_name>/
```

当前已验证的 raw ScanNet 路径是：

```bash
/home/data1/ScanNet/scans
```

如果换机器或挂载点变化，可以设置：

```bash
export SCANNET_RAW_ROOT=/path/to/ScanNet/scans
```

然后运行：

```bash
python scripts/check_data_paths.py
```
