# Implementation

## Text and shape conditioning

`models/clip_encoder.py` freezes `CLIPTextModel` and returns token features, pooled sentence features and token IDs. The default checkpoint is `openai/clip-vit-base-patch32`, with a maximum sequence length of 77.

`models/diffusion.py` applies cross-attention at 256- and 512-dimensional point-feature stages. Point features provide queries; CLIP tokens provide keys and values. The attention output is added with residual weight 0.5. FiLM uses pooled text features to modulate 512-dimensional features.

`models/attention.py` recognizes a predefined set of English stop words by decoding token IDs. It multiplies their post-softmax attention weights by 0.1 by default and renormalizes across tokens. This happens during training and inference when token IDs and the tokenizer are supplied. The token sequence is preserved.

## Shape model

`FlowVAE` and `GaussianVAE` use `PointNetEncoder`; the optional PointCNN source is retained separately. A noisy point cloud, shape latent and timestep feed the pointwise denoiser. Reverse diffusion starts from Gaussian point noise. The latent prior is either Gaussian or flow-based.

Training defaults are latent dimension 256, 100 diffusion steps, batch size 8 and 1,024-point training samples. These code defaults alone do not establish the full published reproduction protocol.

## Losses

The main reconstruction term predicts diffusion noise with mean squared error, alongside the VAE prior term. The optional text-shape alignment branch encodes a detached predicted clean point cloud; it does not directly send gradients through that cloud into the denoiser. See the source before interpreting this branch as a denoiser alignment objective.

## Paths and checkpoints

Repository-owned paths resolve with `utils.paths.project_path`. Explicit CLI overrides retain standard working-directory semantics. Model/utility package names and checkpoint keys are unchanged by the directory cleanup. Data preparation writes model-ID mappings aligned only with successfully stored HDF5 rows.

See [usage](usage.md) and [data preparation](data.md) for entry points and file schemas.
