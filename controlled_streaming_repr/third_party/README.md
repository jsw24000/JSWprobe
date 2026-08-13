# third_party

本目录只保存指向主实验目录中已有源码仓库的软链接，不复制源码、不保存权重。

期望结构：

- `lingbot-map -> ../../lingbot-map`
- `vggt -> ../../vggt`

如果软链接不存在或失效，可以在项目根目录执行：

```bash
ln -s ../../lingbot-map controlled_streaming_repr/third_party/lingbot-map
ln -s ../../vggt controlled_streaming_repr/third_party/vggt
```

