# Implementation Details

## Dataset

### Data Preparation

We use ShapeNet chair and table categories for training our text-conditioned point cloud generation model.

**Dataset Statistics:**
- **Categories**: Chair (03001627), Table (04379243)
- **Total Models**: 12,041
  - Chair: 6,778 models (train: 5,422 / val: 677 / test: 679)
  - Table: 5,263 models (train: 4,210 / val: 526 / test: 527)
- **Points per Model**: 2,048 points
- **Train/Val/Test Split**: 80% / 10% / 10%

**Data Processing Pipeline:**
1. Load point clouds from ShapeNet dataset
   - Chair: `.pts` format from `data/chair_part/03001627/points/`
   - Table: `.txt` format from `data/04379243/` (7-column format, using first 3 columns for xyz coordinates)
2. Resample each point cloud to 2,048 points using random sampling
3. Apply normalization: `scale_mode='shape_unit'`
   - Center each shape: subtract mean coordinate
   - Scale by standard deviation of all points in the shape
4. Store in HDF5 format for efficient loading

**Text Captions:**
- Source: `captions.tablechair.csv` (45,219 captions)
- Coverage: 74.1% of point clouds have matched text descriptions
- Caption-to-model matching via model ID mapping
- Filtered dataset: Only samples with exact caption matches are used (7,142 training samples)

### Data Loading

```python
train_dataset = ShapeNetCoreText(
    path='./data/shapenet_tablechair.hdf5',
    cates=['chair', 'table'],
    split='train',
    scale_mode='shape_unit',
    captions_path='./data/captions.tablechair.csv',
    modelid_mapping_path='./data/modelid_mapping_tablechair.json'
)
```

## Model Architecture

### Overview

Our model consists of three main components:
1. **Encoder**: PointNet-based VAE encoder (Point Cloud → Latent Code)
2. **Flow Model**: Normalizing flow for latent space modeling
3. **Diffusion Model**: Denoising diffusion for point cloud generation

### Text Conditioning

**Text Encoder:**
- Model: CLIP ViT-B/32 (`openai/clip-vit-base-patch32`)
- Output: 512-dimensional text embeddings
- Token sequence length: 77 tokens
- Frozen during training (no fine-tuning)

**Cross-Attention Integration:**
- Two cross-attention layers at different feature dimensions:
  - 256-dimensional features (early layers)
  - 512-dimensional features (middle layers)
- Query: Point cloud features
- Key/Value: CLIP text token embeddings
- Attention heads: 8 (implicit in implementation)

**FiLM Conditioning:**
- Feature-wise Linear Modulation layers
- Applied to 512-dimensional features
- Modulates diffusion features with text information

### Encoder (VAE)

**Architecture: PointNet**
- Input: (B, 2048, 3) point cloud
- Layers:
  ```
  Conv1D(3 → 128) + BN + ReLU
  Conv1D(128 → 128) + BN + ReLU
  Conv1D(128 → 256) + BN + ReLU
  Conv1D(256 → 512) + BN + ReLU
  MaxPool(512 → 512)  # Global feature
  FC(512 → 256 → 128 → 256)  # Mean
  FC(512 → 256 → 128 → 256)  # Log-variance
  ```
- Output: Latent code z ∈ ℝ^256

### Flow Model

**Normalizing Flow:**
- Type: Coupling layers (affine transformations)
- Depth: 14 layers
- Hidden dimension: 256
- Network per layer:
  ```
  FC(128 → 256) + ReLU
  FC(256 → 256) + ReLU
  FC(256 → 256)
  ```
- Purpose: Model complex latent distribution p(z)

### Diffusion Model

**Architecture: PointwiseNet with Cross-Attention**

**Point-wise MLP:**
```
ConcatSquashLinear(3 → 128)     # Input layer
ConcatSquashLinear(128 → 256)   # + Cross-Attention
ConcatSquashLinear(256 → 512)   # + Cross-Attention + FiLM
ConcatSquashLinear(512 → 256)   # + FiLM
ConcatSquashLinear(256 → 128)
ConcatSquashLinear(128 → 3)     # Output layer
```

**Conditional Inputs:**
- Latent code z: 256-dim → project to 256-dim via MLP
- Text embeddings: 512-dim token sequences (77 tokens)
- Timestep t: embedded and concatenated at each layer

**Variance Schedule:**
- Type: Linear schedule
- β₁ = 0.0001
- βₜ = 0.02
- Number of timesteps T = 100

## Training Configuration

### Hyperparameters

**Optimization:**
- Optimizer: Adam
- Learning rate: 0.001 (initial)
- End learning rate: 0.00001
- Learning rate schedule: Linear decay
  - Start epoch: 50,000
  - End epoch: 100,000
- Weight decay: 0
- Gradient clipping: max_norm = 10

**Batch Size:**
- Training: 8
- Validation: 8

**Loss Weights:**
- KL divergence weight: 0.001
- Alignment loss weight: 0.1 (CLIP-based text-shape alignment)

**Training Duration:**
- Maximum iterations: 100,000
- Validation frequency: every 2,000 iterations
- Test frequency: every 10,000 iterations

### Loss Function

**Total Loss:**
```
L_total = L_diffusion + λ_KL * L_KL + λ_align * L_align
```

Where:
- `L_diffusion`: Denoising score matching loss
- `L_KL`: KL divergence between q(z|x) and p(z) (flow prior)
- `L_align`: CLIP-based alignment loss between generated shapes and text

**Denoising Loss:**
```
L_diffusion = E_{t,x₀,ε}[||ε - ε_θ(x_t, t, z, text)||²]
```

### Data Augmentation

- Random point shuffling (implicit in sampling)
- Shape-unit normalization for scale invariance
- No rotation or translation augmentation

## Implementation Details

**Framework:**
- PyTorch 2.x
- CUDA-enabled GPU training

**Reproducibility:**
- Random seed: 2020 (fixed for all experiments)
- Deterministic data splits
- Frozen CLIP weights ensure consistency

**Logging:**
- TensorBoard for training metrics
- Checkpoint saving for model recovery
- Validation samples generated every 2,000 iterations

## Generation

**Sampling Process:**
1. Sample latent code z ~ p(z) from flow prior
2. Initialize x_T ~ N(0, I)
3. Denoise for T steps:
   ```
   x_{t-1} = μ_θ(x_t, t, z, text) + σ_t * ε
   ```
4. Output: Generated point cloud x_0

**Inference Parameters:**
- Number of diffusion steps: 100
- Truncation: σ truncated at 2.0 std
- Flexibility: 0.0 (deterministic sampling)

**Generation Control:**
- Text prompt: Natural language description
- Latent interpolation: Smooth transitions between shapes
- Category control: Implicit via text description

## File Organization

```
data/
├── shapenet_tablechair.hdf5          # Main dataset
├── captions.tablechair.csv            # Text descriptions
├── modelid_mapping_tablechair.json   # Caption-to-model mapping
└── shapenet_tablechair_stats/        # Normalization statistics

logs_gen/                              # Training logs
├── checkpoints/                       # Model checkpoints
└── tensorboard/                       # TensorBoard events

models/
├── vae_flow.py                        # VAE + Flow model
├── diffusion.py                       # Diffusion model
├── attention.py                       # Cross-attention layers
└── clip_encoder.py                    # CLIP text encoder
```

## Key Design Choices

1. **Two-stage generation:** VAE+Flow for latent modeling, then diffusion for point generation
   - Separates global shape from local details
   - More stable training than direct point diffusion

2. **Cross-attention for text conditioning:**
   - Allows fine-grained text-to-point correspondence
   - Better than simple concatenation or FiLM alone
   - Applied at multiple feature scales

3. **CLIP for text encoding:**
   - Pretrained vision-language alignment
   - Rich semantic understanding
   - Frozen weights reduce overfitting

4. **Shape-unit normalization:**
   - Scale-invariant learning
   - Better generalization across sizes
   - Preserves relative proportions

5. **Filtered dataset (caption-matched only):**
   - Ensures high-quality text-shape pairs
   - 74.1% coverage is sufficient for training
   - Avoids noisy pseudo-captions

## Computational Requirements

**Training:**
- GPU: 1x NVIDIA GPU with 16GB+ VRAM (estimated)
- Training time: ~20-30 hours for 100k iterations (estimated)
- Batch size: 8 (can be adjusted based on GPU memory)

**Inference:**
- Generation time: ~1-2 seconds per sample
- Memory: ~2GB VRAM for batch size 1

## Differences from Original Paper (if applicable)

If this is based on existing work, document:
- Architecture modifications
- Hyperparameter changes
- Dataset differences
- Training procedure adjustments
