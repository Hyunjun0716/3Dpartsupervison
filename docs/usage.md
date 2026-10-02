# Usage and path migration

Run module commands from the repository root. Install a GPU-compatible PyTorch build and `requirements.txt` first. Default paths resolve from `utils.paths.PROJECT_ROOT`; explicit command-line paths retain normal working-directory semantics.

## Training

```bash
python -m scripts.train \
    --model flow \
    --dataset_path data/shapenet_tablechair.hdf5 \
    --captions_path data/captions.tablechair.csv \
    --modelid_mapping_path data/modelid_mapping_tablechair.json \
    --categories chair,table \
    --use_text_condition True \
    --train_batch_size 8 --device cuda
```

Use `--max_iters` to bound training, `--resume` to continue from a checkpoint, and `--log_root` to override the log destination. Defaults include latent dimension 256, 100 diffusion steps and 1,024-point training samples.

The optional alignment-loss implementation detaches the predicted cloud before the auxiliary shape encoder, so this branch does not directly backpropagate through that cloud into the denoiser. Consult `models/diffusion.py` before planning loss ablations.

## Generation and evaluation

```bash
python -m scripts.generate \
    --ckpt pretrained/YOUR_CHECKPOINT.pt \
    --dataset_path data/shapenet_tablechair.hdf5 \
    --captions_path data/captions.tablechair.csv \
    --modelid_mapping_path data/modelid_mapping_tablechair.json \
    --categories chair --sample_num_points 1024 --batch_size 8 --device cuda
```

Use `--categories table` for a separate table run. Add `--text_prompt "a chair with armrests"` to override all sampled captions; this evaluator still loads a reference dataset. Use `scripts.visualize` for custom prompts without a reference dataset.

Samples are saved to a timestamped folder under `results/` as `out.npy`. Metrics include MMD-CD, COV-CD, 1-NN-CD and JSD. EMD is not implemented in this path. Match normalization, reference split, sample count and reporting scales before comparing with the paper. Training-time validation is not interchangeable with final test evaluation.

## Visualization

```bash
python -m scripts.visualize \
    --ckpt pretrained/YOUR_CHECKPOINT.pt \
    --captions_file examples/captions/custom.txt \
    --sample_num_points 1024 --grid_only --save_dir results/qualitative

python -m scripts.visualize_dataset \
    --ckpt pretrained/YOUR_CHECKPOINT.pt \
    --categories chair table --visualize_attention --sample_num_points 1024
```

Open3D rendering may require a graphical/OpenGL environment. See the attention guide for rendering options. Curated experimental images should be copied into `assets/images/` with the originating prompt, checkpoint or paper figure recorded in `assets/README.md`.

## Monitoring

```bash
python -m scripts.monitor --log_root logs_gen
```

## Earlier command names

| Previous root file | Current command |
| --- | --- |
| `train_gen.py` | `python -m scripts.train` |
| `test_gen.py` | `python -m scripts.generate` |
| `visualize_random_captions.py` | `python -m scripts.visualize` |
| `visualize_training_samples.py` | `python -m scripts.visualize_dataset` |
| `create_tablechair_hdf5.py` | `python -m scripts.prepare_data` |
| `build_shapenet_hdf5.py` | `python -m scripts.build_shapenet` |
| `check_training_progress.py` | `python -m scripts.monitor` |
| `test_tablechair_dataset.py` | `python -m scripts.check_dataset` |

Legacy autoencoder/generation/comprehensive evaluation scripts are preserved under `archive/scripts/`. Their reports and environment snapshot live in `archive/`; they are not the recommended paper reproduction workflow.

`models`, `utils` and `evaluation` retain their package names. No checkpoints or numerical benchmark reports were regenerated during this organization change.
