# Attention visualization

The denoiser can return attention dictionaries with `attn_256` and `attn_512`. Each attention tensor has shape `(batch, heads, points, tokens)`. In the current implementation the returned weights include stop-word reweighting when token IDs and the tokenizer are available.

Render dataset-based samples and attention from the repository root:

```bash
python -m scripts.visualize_dataset \
    --ckpt pretrained/YOUR_CHECKPOINT.pt \
    --dataset_path data/shapenet_tablechair.hdf5 \
    --captions_path data/captions.tablechair.csv \
    --modelid_mapping_path data/modelid_mapping_tablechair.json \
    --categories chair table --sample_num_points 1024 --visualize_attention
```

For custom prompts without reference point clouds:

```bash
python -m scripts.visualize \
    --ckpt pretrained/YOUR_CHECKPOINT.pt \
    --captions_file examples/captions/custom.txt \
    --sample_num_points 1024 --grid_only --save_pointclouds
```

Use `python -m scripts.visualize --help` for view, grid and denoising-process options. Open3D rendering can require a graphical/OpenGL environment.

A visualization is evidence of the attention distribution, not by itself a quantitative test of semantic grounding. Record the prompt, checkpoint, seed, point count and rendering settings alongside any curated figure.
