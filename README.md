# Octree-Gsplat
将Gsplat后端融入Octree-GS，实现后端初步优化。

## Tile-based 渲染参数
`train.py` 和所有训练脚本都会自动暴露 `Pipeline Parameters` 分组里的参数，因此可以直接在命令行上控制图块调度：

```bash
python train.py \
    --source_path data/xxx \
    --model_path outputs/xxx \
    --enable_tiling \
    --tile_size 192 \
    --tile_overlap 2
```

- `--enable_tiling`：开启图块调度（默认关闭）。
- `--tile_size`：单个 tile 的像素尺寸，必须与训练时设定的输入分辨率兼容。
- `--tile_overlap`：相邻 tile 之间额外扩展的像素行/列数，用于避免边界裁剪伪影。

如果你通过 `train.sh` 等脚本封装训练，只需把同样的参数原样追加到脚本尾部，它们会被传递到内部的 `python train.py ...` 调用中。
