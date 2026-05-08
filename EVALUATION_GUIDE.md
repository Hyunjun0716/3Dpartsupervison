# Model Evaluation Guide

## Overview
This guide describes how to use the comprehensive evaluation script for the text-conditioned point cloud generation model.

## Evaluation Script

### Location
- Main script: `evaluate_model.py`
- Supporting metric files:
  - `evaluation/geometric_metrics.py` - CD, EMD, F-score
  - `evaluation/distribution_metrics.py` - FPD, JSD, MMD, Coverage, 1-NN
  - `evaluation/clip_metrics.py` - CLIP-Score, R-Precision

### Usage

Basic usage (evaluates on all test samples):
```bash
python evaluate_model.py
```

With custom checkpoint and output:
```bash
python evaluate_model.py \
    --checkpoint logs_gen/GEN_2025_11_30__01_59_45/ckpt_0.000000_500000.pt \
    --output evaluation_results.txt
```

With limited samples for testing:
```bash
python evaluate_model.py --num_samples 10
```

### Metrics Computed

#### 1. Geometric Fidelity Metrics
- **Chamfer Distance (CD)**: Measures point-to-point distance between generated and ground truth
- **Earth Mover's Distance (EMD)**: Optimal transport distance (currently not implemented due to GPU compatibility)
- **F-score @ thresholds**: Precision-recall harmonic mean at 0.001, 0.002, 0.005 thresholds

#### 2. Distribution Metrics
- **MMD-CD (Minimum Matching Distance)**: Measures how well generated samples match reference distribution
- **Coverage (COV-CD)**: Percentage of reference samples matched by generated samples
- **1-NN Accuracy**: Leave-one-out nearest neighbor classification accuracy (lower is better)
- **JSD (Jensen-Shannon Divergence)**: Statistical distance between distributions

#### 3. CLIP-based Metrics
- **CLIP-Score**: Cosine similarity between point cloud and text embeddings
- **R-Precision@K**: Retrieval accuracy at different ranks (K=1, 5, 10)
- **Median Rank**: Median rank of correct text for each point cloud
- **MRR (Mean Reciprocal Rank)**: Average of reciprocal ranks

## Implementation Details

### Key Features
1. **Memory-efficient**: Processes large datasets in batches to avoid OOM errors
2. **GPU-accelerated**: Uses CUDA for all distance computations
3. **Progress tracking**: Shows progress bars for all long-running operations
4. **Comprehensive logging**: Saves detailed results to file

### Fixed Issues
- ✅ Handles model attributes correctly (`model.args.latent_dim`)
- ✅ Batch processing for F-score computation to avoid OOM
- ✅ Proper tensor device management
- ✅ Dimension projection for CLIP metrics (256D → 512D)
- ✅ Compatible with both chair and table categories

### Expected Runtime
For 925 test samples on GPU:
- Generation: ~1-2 minutes
- Geometric metrics: ~1 minute
- Distribution metrics: ~6-8 minutes (due to pairwise distance computation)
- CLIP metrics: ~1-2 minutes
- **Total**: ~10-13 minutes

## Example Output

```
============================================================
EVALUATION RESULTS SUMMARY
============================================================
CD                            : 0.538155
EMD                           : 0.000000
F-score@0.001                 : 1.000000
F-score@0.002                 : 1.000000
F-score@0.005                 : 1.000000
lgan_mmd-CD                   : 0.426255
lgan_cov-CD                   : 0.300000
lgan_mmd_smp-CD               : 0.352172
1-NN-CD-acc                   : 0.850000
JSD                           : 0.122443
CLIP-Score                    : 0.025319
R-Precision@1                 : 0.100000
R-Precision@5                 : 0.500000
R-Precision@10                : 1.000000
Median-Rank                   : 5.000000
MRR                           : 0.287897
```

## Dataset Information

- **Dataset**: shapenet_tablechair.hdf5
- **Categories**: chair (03001627) + table (04379243)
- **Total test samples**: 925 (76.7% with caption matches)
- **Points per model**: 2,048
- **Normalization**: Shape-unit (per-shape centering + scaling)

## Citation

If you use this evaluation code, please cite:
- Chamfer Distance: [PointFlow]
- MMD/Coverage/1-NN: [Learning Representations and Generative Models for 3D Point Clouds]
- JSD: [Learning Representations and Generative Models for 3D Point Clouds]
- CLIP metrics: [CLIP]

## Troubleshooting

### Out of Memory (OOM) Errors
If you encounter OOM errors:
1. Reduce `--batch_size` (default: 32)
2. Reduce `--num_samples` to evaluate on fewer samples
3. Use a GPU with more memory

### Slow Evaluation
- Distribution metrics can be slow due to pairwise distance computation
- Consider evaluating on a subset first with `--num_samples 100`

### CLIP Metrics Warning
- CLIP metrics project point cloud features (256D) to text features (512D)
- This is expected and handled automatically by the script
