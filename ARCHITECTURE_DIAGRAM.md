# Text-Conditioned Point Cloud Diffusion VAE Architecture

## 전체 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐         ┌──────────────────┐
│  ShapeNet HDF5   │         │  Caption CSV     │
│  (2048 points)   │         │  (Text Prompts)  │
└────────┬─────────┘         └────────┬─────────┘
         │                            │
         │ Load & Normalize           │ Load & Match
         │ (shape_unit/bbox)          │ (by modelId)
         ▼                            ▼
┌─────────────────────────────────────────────────┐
│         DataLoader (Batch Size: 32)             │
│     Point Cloud (B, 2048, 3) + Captions         │
└────────┬────────────────────────────────────────┘
         │
         │ Random Downsample
         ▼
┌─────────────────────────────────────────────────┐
│       Point Cloud (B, 1024, 3) + Captions       │
└────┬────────────────────────────────────────┬───┘
     │                                        │
     │                                        │
     ▼                                        ▼
┌─────────────────────┐           ┌──────────────────────┐
│  SHAPE ENCODER      │           │  TEXT ENCODER        │
│  (PointNet VAE)     │           │  (CLIP)              │
└─────────────────────┘           └──────────────────────┘
```

---

## 1. PointNet VAE Encoder (Shape → Latent)

```
Input: Point Cloud (B, N=1024, 3)
  │
  │ Transpose: (B, 3, N)
  ▼
┌─────────────────────────────────────┐
│  Conv1D: 3 → 128                    │
│  BatchNorm + ReLU                   │
└───────────┬─────────────────────────┘
            │ Point-wise Features
            ▼
┌─────────────────────────────────────┐
│  Conv1D: 128 → 128                  │
│  BatchNorm + ReLU                   │
└───────────┬─────────────────────────┘
            │ Local Structure
            ▼
┌─────────────────────────────────────┐
│  Conv1D: 128 → 256                  │
│  BatchNorm + ReLU                   │  ◄─── Mid-level Features
└───────────┬─────────────────────────┘      (Part Structure)
            │
            ▼
┌─────────────────────────────────────┐
│  Conv1D: 256 → 512                  │
│  BatchNorm + ReLU                   │  ◄─── High-level Features
└───────────┬─────────────────────────┘      (Global Semantics)
            │
            │ Global Max Pooling
            ▼
┌─────────────────────────────────────┐
│   Global Descriptor (B, 512)        │
└───────┬─────────────────────────┬───┘
        │                         │
        ▼                         ▼
┌───────────────┐         ┌───────────────┐
│  MLP → z_mu   │         │ MLP → z_logvar│
│  (B, 256)     │         │  (B, 256)     │
└───────┬───────┘         └───────┬───────┘
        │                         │
        └──────────┬──────────────┘
                   │ Reparameterization Trick
                   │ z = μ + exp(0.5σ) * ε
                   ▼
          ┌─────────────────┐
          │   Latent z      │
          │   (B, 256)      │
          └─────────────────┘
```

---

## 2. CLIP Text Encoder (Caption → Text Embedding)

```
Input: Text Caption "a wooden chair with armrests"
  │
  │ Tokenize (max 77 tokens)
  ▼
┌─────────────────────────────────────┐
│  CLIP Tokenizer                     │
│  token_ids: (B, 77)                 │
└───────────┬─────────────────────────┘
            │
            ▼
┌─────────────────────────────────────┐
│  CLIP Text Transformer              │
│  ViT-B/32 or ViT-L/14               │
└───────┬────────────────────┬────────┘
        │                    │
        │                    │ [CLS] token
        ▼                    ▼
┌──────────────────┐  ┌─────────────────┐
│  Token Features  │  │  Pooled Feature │
│  (B, 77, 512)    │  │  (B, 512)       │
└──────────────────┘  └─────────────────┘
        │                    │
        │                    │
        ▼                    ▼
   For Cross-Attn        For FiLM & Alignment

┌─────────────────────────────────────────┐
│  text_emb = {                           │
│    'tokens': (B, 77, 512),  ◄─ Attn    │
│    'pool': (B, 512),        ◄─ FiLM    │
│    'token_ids': (B, 77)     ◄─ Filter  │
│  }                                      │
└─────────────────────────────────────────┘
```

---

## 3. Forward Diffusion Process (Training)

```
Clean Point Cloud x₀ (B, 1024, 3)
  │
  │ Sample t ~ Uniform(1, T=100)
  │ Sample ε ~ N(0, I)
  │
  │ x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε
  ▼
Noisy Point Cloud x_t (B, 1024, 3)


Variance Schedule (Linear):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
β₁ = 1e-4  ───────────────►  β_T = 0.02

α_t = 1 - β_t
ᾱ_t = ∏(i=1→t) α_i

Noise Level: Low ─────────────► High
t = 1                         t = 100
```

---

## 4. PointwiseNet Denoiser (Core Architecture)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      POINTWISENET ARCHITECTURE                          │
└─────────────────────────────────────────────────────────────────────────┘

Inputs:
  • x_t: Noisy points (B, N, 3)
  • β_t: Noise level (B,)
  • z: Latent code (B, 256)
  • text_emb: Text embeddings (dict)

┌────────────────────────────────────────────────────────┐
│  CONTEXT PREPARATION                                   │
├────────────────────────────────────────────────────────┤
│                                                        │
│  ┌─────────────┐        ┌──────────────┐             │
│  │ β_t (B,)    │        │ z (B, 256)   │             │
│  └──────┬──────┘        └──────┬───────┘             │
│         │                      │                      │
│         │ Expand               │ Linear               │
│         ▼                      ▼                      │
│  ┌─────────────────┐    ┌──────────────┐             │
│  │ [β, sin(β),     │    │ z_proj       │             │
│  │  cos(β)]        │    │ (B, 1, 256)  │             │
│  │ (B, 1, 3)       │    │              │             │
│  └────────┬────────┘    └──────┬───────┘             │
│           │                    │                      │
│           └──────┬─────────────┘                      │
│                  │ Concatenate                        │
│                  ▼                                    │
│           ┌─────────────┐                             │
│           │   Context   │                             │
│           │  (B, 1, 259)│                             │
│           └─────────────┘                             │
└────────────────────────────────────────────────────────┘


┌────────────────────────────────────────────────────────┐
│  6-LAYER MLP WITH MULTI-SCALE CONDITIONING             │
└────────────────────────────────────────────────────────┘

x_t (B, N, 3)
  │
  ▼
┌──────────────────────────────────────────┐
│ Layer 0: ConcatSquashLinear(3→128)      │
│ ┌────────────────────────────────┐      │
│ │ gate = σ(W_gate(ctx))          │      │
│ │ bias = W_bias(ctx)             │      │
│ │ out = W(x) * gate + bias       │      │
│ └────────────────────────────────┘      │
│ LeakyReLU(0.2)                          │
└───────────┬──────────────────────────────┘
            │ (B, N, 128)
            ▼
┌──────────────────────────────────────────┐
│ Layer 1: ConcatSquashLinear(128→256)    │
│ LeakyReLU(0.2)                          │
└───────────┬──────────────────────────────┘
            │ (B, N, 256)
            │
            ├─────────────────────────────────┐
            │                                 │
            ▼                                 ▼
┌───────────────────────────┐    ┌──────────────────────────┐
│  CROSS-ATTENTION-256      │    │  MAIN PATH               │
│  (Part-Level)             │    │                          │
├───────────────────────────┤    │                          │
│ Q: point_feat (B,N,256)   │    │                          │
│ K,V: text_tokens          │    │                          │
│     (B, 77, 512)          │    │                          │
│                           │    │                          │
│ Multi-Head: 4 heads       │    │                          │
│ dim_head: 64              │    │                          │
│                           │    │                          │
│ ┌─────────────────────┐   │    │                          │
│ │ Attention Map       │   │    │                          │
│ │ (B,4,N,77)          │   │    │                          │
│ └──────┬──────────────┘   │    │                          │
│        │                  │    │                          │
│        ▼                  │    │                          │
│ ┌─────────────────────┐   │    │                          │
│ │ STOP WORD FILTER    │   │    │                          │
│ │ "a","the","with"    │   │    │                          │
│ │ → weight × 0.1      │   │    │                          │
│ └──────┬──────────────┘   │    │                          │
│        │                  │    │                          │
│        ▼                  │    │                          │
│ ┌─────────────────────┐   │    │                          │
│ │ Attn_out (B,N,256)  │   │    │                          │
│ └──────┬──────────────┘   │    │                          │
└────────┼──────────────────┘    │                          │
         │                       │                          │
         │ × 0.5 (residual)      │                          │
         └───────────────────────┤                          │
                                 ▼                          │
                          ┌──────────────┐                  │
                          │ feat (B,N,256)│                 │
                          └──────┬───────┘                  │
                                 └──────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────┐
│ Layer 2: ConcatSquashLinear(256→512)    │
│ LeakyReLU(0.2)                          │
└───────────┬──────────────────────────────┘
            │ (B, N, 512)
            │
            ├─────────────────────────────────┬──────────────┐
            │                                 │              │
            ▼                                 ▼              ▼
┌───────────────────────────┐    ┌────────────────┐  ┌──────────────┐
│  CROSS-ATTENTION-512      │    │  FiLM-1        │  │  MAIN PATH   │
│  (Semantic-Level)         │    │  (Global Cond) │  │              │
├───────────────────────────┤    ├────────────────┤  │              │
│ Q: point_feat (B,N,512)   │    │ γ = Linear_γ   │  │              │
│ K,V: text_tokens          │    │     (text_pool)│  │              │
│     (B, 77, 512)          │    │ β = Linear_β   │  │              │
│                           │    │     (text_pool)│  │              │
│ Multi-Head: 8 heads       │    │                │  │              │
│ dim_head: 64              │    │ out = γ*x + β  │  │              │
│                           │    │                │  │              │
│ + Stop Word Filtering     │    └───────┬────────┘  │              │
│                           │            │           │              │
│ Attn_out (B,N,512)        │            │           │              │
│ × 0.5 (residual)          │            │           │              │
└───────────┬───────────────┘            │           │              │
            │                            │           │              │
            └────────────┬───────────────┘           │              │
                         │                           │              │
                         └───────────────────────────┤              │
                                                     ▼              │
                                              ┌──────────────┐      │
                                              │ feat(B,N,512)│      │
                                              └──────┬───────┘      │
                                                     └──────────────┘
            │
            ▼
┌──────────────────────────────────────────┐
│ Layer 3: ConcatSquashLinear(512→256)    │
│ LeakyReLU(0.2)                          │
└───────────┬──────────────────────────────┘
            │ (B, N, 256)
            │
            ├──────────────────────────┐
            │                          │
            ▼                          ▼
┌────────────────────┐         ┌──────────────┐
│  FiLM-2            │         │  MAIN PATH   │
│  (Global Cond)     │         │              │
├────────────────────┤         │              │
│ γ = Linear_γ       │         │              │
│     (text_pool)    │         │              │
│ β = Linear_β       │         │              │
│     (text_pool)    │         │              │
│                    │         │              │
│ out = γ*x + β      │         │              │
└──────────┬─────────┘         │              │
           │                   │              │
           └───────────────────┤              │
                               ▼              │
                        ┌──────────────┐      │
                        │ feat(B,N,256)│      │
                        └──────┬───────┘      │
                               └──────────────┘
            │
            ▼
┌──────────────────────────────────────────┐
│ Layer 4: ConcatSquashLinear(256→128)    │
│ LeakyReLU(0.2)                          │
└───────────┬──────────────────────────────┘
            │ (B, N, 128)
            ▼
┌──────────────────────────────────────────┐
│ Layer 5: ConcatSquashLinear(128→3)      │
└───────────┬──────────────────────────────┘
            │
            ▼
   Noise Prediction ε_θ (B, N, 3)
```

---

## 5. Cross-Attention Mechanism (Detailed)

```
┌──────────────────────────────────────────────────────────────┐
│              MULTI-HEAD CROSS-ATTENTION                      │
└──────────────────────────────────────────────────────────────┘

Query: Point Features (B, N, query_dim)
Key/Value: Text Token Embeddings (B, 77, 512)

┌─────────────────┐         ┌─────────────────┐
│  Point Features │         │  Text Tokens    │
│  (B, N, d_q)    │         │  (B, 77, 512)   │
└────────┬────────┘         └────────┬────────┘
         │                           │
         │ W_q                       │ W_k, W_v
         ▼                           ▼
┌─────────────────┐         ┌─────────────────┐
│  Q (B,N,inner)  │         │  K,V(B,77,inner)│
└────────┬────────┘         └────────┬────────┘
         │                           │
         │ Reshape                   │ Reshape
         ▼                           ▼
┌─────────────────┐         ┌─────────────────┐
│ (B,h,N,d_h)     │         │ (B,h,77,d_h)    │
└────────┬────────┘         └────────┬────────┘
         │                           │
         └───────────┬───────────────┘
                     │
                     │ Attn = softmax(Q·K^T / √d_h)
                     ▼
              ┌─────────────────┐
              │ Attention Map   │
              │ (B, h, N, 77)   │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────────────────────┐
              │  STOP WORD FILTERING            │
              ├─────────────────────────────────┤
              │                                 │
              │  For each token position j:     │
              │                                 │
              │  ┌────────────────────────┐     │
              │  │ Decode token_id[j]     │     │
              │  └──────────┬─────────────┘     │
              │             │                   │
              │             ▼                   │
              │  ┌────────────────────────┐     │
              │  │ Is stop word?          │     │
              │  │ {"a","the","with",...} │     │
              │  └──────┬──────┬──────────┘     │
              │         │      │                │
              │    YES  │      │  NO            │
              │         ▼      ▼                │
              │  weight=0.1   weight=1.0        │
              │                                 │
              │  filter_mask (B, 1, 1, 77)      │
              │                                 │
              │  Attn' = Attn * filter_mask     │
              │  Attn' = Attn' / sum(Attn')     │
              │         (re-normalize)          │
              └────────┬────────────────────────┘
                       │
                       │ Attn' (B, h, N, 77)
                       ▼
              ┌─────────────────┐
              │   Out = Attn·V  │
              │  (B, h, N, d_h) │
              └────────┬────────┘
                       │
                       │ Reshape + Concat
                       ▼
              ┌─────────────────┐
              │  (B, N, inner)  │
              └────────┬────────┘
                       │
                       │ W_o (projection)
                       ▼
              ┌─────────────────┐
              │  Output         │
              │  (B, N, d_q)    │
              └─────────────────┘

Example:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Caption: "a wooden chair with four legs"

Token weights BEFORE filtering:
  a       wooden  chair   with    four    legs
 0.15     0.20    0.25    0.10    0.15    0.15

Token weights AFTER filtering (stop words: a, with):
  a       wooden  chair   with    four    legs
 0.02     0.29    0.36    0.01    0.21    0.21
        ↑ Enhanced attention to meaningful words
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 6. FiLM (Feature-wise Linear Modulation)

```
┌────────────────────────────────────────┐
│  FiLM: Global Text Conditioning        │
└────────────────────────────────────────┘

Input:
  • features: (B, N, C)
  • text_pool: (B, 512) - CLIP [CLS] token

┌─────────────────┐
│  text_pool      │
│  (B, 512)       │
└────────┬────────┘
         │
         ├───────────────────┐
         │                   │
         ▼                   ▼
┌─────────────────┐  ┌─────────────────┐
│  Linear_γ       │  │  Linear_β       │
│  (512 → C)      │  │  (512 → C)      │
└────────┬────────┘  └────────┬────────┘
         │                    │
         ▼                    ▼
┌─────────────────┐  ┌─────────────────┐
│  γ (B, C)       │  │  β (B, C)       │
└────────┬────────┘  └────────┬────────┘
         │                    │
         │ Expand             │ Expand
         ▼                    ▼
┌─────────────────┐  ┌─────────────────┐
│  γ (B, 1, C)    │  │  β (B, 1, C)    │
└────────┬────────┘  └────────┬────────┘
         │                    │
         └──────┬─────────────┘
                │
                │  Modulate features
                ▼
     ┌─────────────────────┐
     │ features (B, N, C)  │
     └──────────┬──────────┘
                │
                │  out = γ * features + β
                ▼
     ┌─────────────────────┐
     │ modulated (B, N, C) │
     └─────────────────────┘

Purpose:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
γ (scale):  Controls feature magnitude
            e.g., "large chair" → larger γ
β (shift):  Controls feature offset
            e.g., "wooden" → shift to wood-like features
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 7. Loss Functions (Training)

```
┌────────────────────────────────────────────────────────────┐
│                   TOTAL TRAINING LOSS                      │
└────────────────────────────────────────────────────────────┘

L_total = kl_weight × L_prior + L_recons + align_weight × L_align

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. PRIOR LOSS (KL Divergence)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

L_prior = -log p(z) - H[q(z|x)]

For Gaussian VAE:
  p(z) = N(z; 0, I)
  q(z|x) = N(z; μ, exp(σ))

  L_prior = 0.5 × Σ[μ² + exp(σ) - σ - 1]

  H[q] = 0.5 × zdim × (1 + log(2π)) + 0.5 × Σσ

Weight: kl_weight = 0.001

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2. RECONSTRUCTION LOSS (Denoising)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

L_recons = E_t [ ||ε - ε_θ(x_t, β_t, z, text)||² ]

Process:
  1. Sample t ~ Uniform(1, 100)
  2. Sample ε ~ N(0, I)
  3. x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
  4. ε_θ = PointwiseNet(x_t, β_t, z, text_emb)
  5. L_recons = MSE(ε, ε_θ)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3. ALIGNMENT LOSS (Text-Shape Consistency)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

L_align = -cosine_similarity(shape_features, text_pool)

Process:
┌─────────────────┐
│  x_t (noisy)    │
└────────┬────────┘
         │ Predict clean x_0
         │ x_0_pred = (x_t - √(1-ᾱ_t)·ε_θ) / √ᾱ_t
         ▼
┌─────────────────┐
│  x_0_pred       │
└────────┬────────┘
         │ Encode with PointNet
         ▼
┌──────────────────┐         ┌──────────────────┐
│ shape_features   │         │  text_pool       │
│ (B, 512)         │         │  (B, 512)        │
└────────┬─────────┘         └────────┬─────────┘
         │                            │
         │  Normalize                 │  Already normalized
         │  L2(f) = f / ||f||         │  by CLIP
         ▼                            ▼
┌──────────────────┐         ┌──────────────────┐
│  f_shape_norm    │         │  f_text_norm     │
└────────┬─────────┘         └────────┬─────────┘
         │                            │
         └──────────┬─────────────────┘
                    │
                    │  similarity = Σ(f_shape · f_text)
                    │  L_align = -mean(similarity)
                    ▼
              ┌──────────┐
              │ L_align  │
              └──────────┘

Weight: align_weight = 0.1

Purpose: Force generated shapes to semantically match text
         in CLIP embedding space

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TYPICAL LOSS VALUES:
  L_prior:   ~10-50 (scaled by 0.001 → ~0.01-0.05)
  L_recons:  ~0.001-0.01 (main signal)
  L_align:   ~0.1-0.3 (scaled by 0.1 → ~0.01-0.03)
  L_total:   ~0.02-0.08
```

---

## 8. Reverse Diffusion Process (Sampling)

```
┌────────────────────────────────────────────────────────────┐
│            REVERSE DIFFUSION SAMPLING                      │
└────────────────────────────────────────────────────────────┘

Input:
  • z ~ N(0, I) or z = flow(w)  [latent code]
  • text_emb from CLIP           [text condition]

┌─────────────────┐
│  x_T ~ N(0, I)  │  Pure Gaussian noise
│  (1, 1024, 3)   │
└────────┬────────┘
         │
         │  for t = 100 → 1:
         ▼
    ┌────────────────────────────────────┐
    │  Step t                            │
    ├────────────────────────────────────┤
    │                                    │
    │  ┌──────────────────────────┐      │
    │  │  x_t (1, 1024, 3)        │      │
    │  └──────────┬───────────────┘      │
    │             │                      │
    │             ▼                      │
    │  ┌──────────────────────────┐      │
    │  │  PointwiseNet            │      │
    │  │  ε_θ = f(x_t,β_t,z,text) │      │
    │  └──────────┬───────────────┘      │
    │             │                      │
    │             ▼                      │
    │  ┌──────────────────────────┐      │
    │  │  ε_θ (1, 1024, 3)        │      │
    │  └──────────┬───────────────┘      │
    │             │                      │
    │             │  DDPM Update:        │
    │             │                      │
    │             │  μ_θ = (x_t - (1-α_t)/√(1-ᾱ_t) · ε_θ) / √α_t
    │             │                      │
    │             │  σ_t = √((1-ᾱ_{t-1})/(1-ᾱ_t) · β_t)
    │             │                      │
    │             │  if t > 1:           │
    │             │    z ~ N(0, I)       │
    │             │  else:               │
    │             │    z = 0             │
    │             │                      │
    │             │  x_{t-1} = μ_θ + σ_t · z
    │             │                      │
    │             ▼                      │
    │  ┌──────────────────────────┐      │
    │  │  x_{t-1} (1, 1024, 3)    │      │
    │  └──────────────────────────┘      │
    │                                    │
    └────────────────────────────────────┘
         │
         │  Repeat for t-1
         ▼

t = 100: Very noisy  ████████████████
t = 75:              ████████░░░░░░░░
t = 50:              ████░░░░░░░░░░░░
t = 25:              ██░░░░░░░░░░░░░░
t = 1:   Almost clean ░░░░░░░░░░░░░░░░

         │
         ▼
┌─────────────────────┐
│  x_0 (clean)        │
│  (1, 1024, 3)       │
│  Generated          │
│  Point Cloud        │
└─────────────────────┘
```

---

## 9. Complete Training Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING ITERATION                           │
└─────────────────────────────────────────────────────────────────┘

Batch: (x, captions)
  x: (32, 1024, 3)
  captions: 32 text strings

     ┌──────────┐              ┌──────────┐
     │    x     │              │ captions │
     └────┬─────┘              └────┬─────┘
          │                         │
          ▼                         ▼
   ┌─────────────┐          ┌─────────────┐
   │  PointNet   │          │  CLIP Text  │
   │  Encoder    │          │  Encoder    │
   └──────┬──────┘          └──────┬──────┘
          │                        │
          ▼                        ▼
   ┌─────────────┐          ┌─────────────┐
   │ z_mu, z_σ   │          │  text_emb   │
   └──────┬──────┘          └──────┬──────┘
          │                        │
          │ Reparameterize         │
          ▼                        │
   ┌─────────────┐                 │
   │  z (32,256) │                 │
   └──────┬──────┘                 │
          │                        │
          └───────┬────────────────┘
                  │
                  ▼
         ┌────────────────┐
         │  Forward       │
         │  Diffusion     │
         │  x_0 → x_t     │
         └────────┬───────┘
                  │
                  ▼
         ┌────────────────┐
         │  PointwiseNet  │
         │  (Denoiser)    │
         └────────┬───────┘
                  │
                  ▼
         ┌────────────────┐
         │  Predict ε_θ   │
         └────────┬───────┘
                  │
                  ├──────────────┬──────────────┐
                  │              │              │
                  ▼              ▼              ▼
         ┌─────────────┐ ┌────────────┐ ┌────────────┐
         │  L_recons   │ │  L_prior   │ │  L_align   │
         │  MSE(ε,ε_θ) │ │  KL(q||p)  │ │-cos(s,t)   │
         └──────┬──────┘ └─────┬──────┘ └─────┬──────┘
                │              │              │
                └──────┬───────┴──────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  L_total        │
              │  0.001·L_p +    │
              │  L_r + 0.1·L_a  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Backward       │
              │  Grad Clip(10)  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Optimizer Step │
              │  (Adam lr=1e-3) │
              └─────────────────┘
```

---

## 10. Complete Inference Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                  INFERENCE PIPELINE                             │
└─────────────────────────────────────────────────────────────────┘

User Input: "a wooden chair with armrests"

                    ┌──────────────────┐
                    │ Text Prompt      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ CLIP Encoder     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ text_emb         │
                    │ {tokens, pool,   │
                    │  token_ids}      │
                    └────────┬─────────┘
                             │
         ┌───────────────────┤
         │                   │
         ▼                   │
┌─────────────────┐          │
│ Sample Latent   │          │
│ z ~ N(0,I)      │          │
│ (1, 256)        │          │
└────────┬────────┘          │
         │                   │
         └─────────┬─────────┘
                   │
                   ▼
          ┌────────────────┐
          │ Initialize     │
          │ x_T ~ N(0,I)   │
          │ (1, 1024, 3)   │
          └────────┬───────┘
                   │
                   │ Reverse Diffusion
                   │ t = 100 → 1
                   │
                   │ ┌─────────────────────┐
                   │ │  At each step t:    │
                   ▼ │                     │
          ┌────────────────────────┐       │
          │  PointwiseNet          │       │
          │  - Time embedding      │       │
          │  - Latent injection    │       │
          │  - Cross-Attention     │       │
          │    * 256-dim (parts)   │       │
          │    * 512-dim (semantic)│       │
          │    * Stop word filter  │       │
          │  - FiLM modulation     │       │
          │  Predict ε_θ           │       │
          └────────┬───────────────┘       │
                   │                       │
                   │ DDPM update           │
                   │ x_{t-1} from x_t      │
                   │                       │
                   └───────────────────────┘
                   │
                   ▼
          ┌────────────────┐
          │ Generated      │
          │ Point Cloud    │
          │ x_0 (1,1024,3) │
          └────────┬───────┘
                   │
                   │ Optional: Denormalize
                   ▼
          ┌────────────────┐
          │ Final Output   │
          │ (.npy, .ply)   │
          └────────────────┘
```

---

## 11. Multi-Scale Feature Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│              HIERARCHICAL FEATURE PROCESSING                    │
└─────────────────────────────────────────────────────────────────┘

Point Cloud Input (N=1024 points)
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                    ENCODER (PointNet)                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Layer 1: 3 → 128   ───► Low-level geometric features  │
│             ↓              • Point positions           │
│             ↓              • Local neighbors           │
│                                                         │
│  Layer 2: 128 → 128 ───► Local structure               │
│             ↓              • Edges, corners            │
│             ↓              • Surface normals           │
│                                                         │
│  Layer 3: 128 → 256 ───► Part-level features           │
│             ↓              • Chair legs                │
│             ↓              • Armrests                  │
│             ↓              • Seat, back                │
│                                                         │
│  Layer 4: 256 → 512 ───► Semantic features             │
│             ↓              • Object category           │
│             ↓              • Overall style             │
│             ↓              • Material type             │
│                                                         │
│  Global Pool: 512   ───► Global descriptor             │
│             ↓                                          │
│             ↓                                          │
│  MLP: 512 → 256     ───► Latent code z                 │
│                                                         │
└─────────────────────────────────────────────────────────┘

         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                  DECODER (Diffusion)                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Layer 0-1: 3 → 128 → 256                              │
│                    ↓                                    │
│                    ├───► Cross-Attn 256                │
│                    │     (Part-level alignment)         │
│                    │     Q: point features (256)        │
│                    │     K,V: text tokens (512)         │
│                    │     "armrests", "legs", "wooden"   │
│                    ↓                                    │
│                                                         │
│  Layer 2: 256 → 512                                     │
│                    ↓                                    │
│                    ├───► Cross-Attn 512                │
│                    │     (Semantic-level alignment)     │
│                    │     Q: point features (512)        │
│                    │     K,V: text tokens (512)         │
│                    │     "chair", "furniture", "wooden" │
│                    ↓                                    │
│                    ├───► FiLM-1                        │
│                    │     (Global style modulation)      │
│                    │     γ, β from text_pool            │
│                    ↓                                    │
│                                                         │
│  Layer 3: 512 → 256                                     │
│                    ↓                                    │
│                    ├───► FiLM-2                        │
│                    │     (Global refinement)            │
│                    ↓                                    │
│                                                         │
│  Layer 4-5: 256 → 128 → 3                              │
│                    ↓                                    │
│                    ↓                                    │
│              Noise Prediction ε_θ                       │
│                                                         │
└─────────────────────────────────────────────────────────┘

FEATURE SCALE CORRESPONDENCE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dimension  │ Encoder Role      │ Decoder Conditioning
━━━━━━━━━━━┼───────────────────┼──────────────────────
128        │ Geometry          │ Time + Latent injection
256        │ Parts             │ Cross-Attn (parts)
512        │ Semantics         │ Cross-Attn + FiLM (global)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 12. Key Hyperparameters

```
┌─────────────────────────────────────────────────────────────┐
│                  MODEL CONFIGURATION                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ARCHITECTURE:                                              │
│    • Latent dim:              256                           │
│    • Num points:              1024 (downsampled)            │
│    • MLP layers:              6                             │
│    • Hidden dims:             128→256→512→256→128→3         │
│                                                             │
│  DIFFUSION:                                                 │
│    • Num steps (T):           100                           │
│    • β_1 (start):             1e-4                          │
│    • β_T (end):               0.02                          │
│    • Schedule:                Linear                        │
│                                                             │
│  ATTENTION:                                                 │
│    • Cross-Attn 256:          4 heads × 64 dim              │
│    • Cross-Attn 512:          8 heads × 64 dim              │
│    • Stop word penalty:       0.1 (90% reduction)           │
│                                                             │
│  LOSS WEIGHTS:                                              │
│    • KL weight:               0.001                         │
│    • Reconstruction:          1.0 (implicit)                │
│    • Alignment:               0.1                           │
│                                                             │
│  TRAINING:                                                  │
│    • Batch size:              32                            │
│    • Learning rate:           1e-3                          │
│    • Optimizer:               Adam (β1=0.9, β2=0.999)       │
│    • Grad clip:               10.0                          │
│    • LR schedule:             Linear decay (50k→100k)       │
│                                                             │
│  TEXT ENCODER:                                              │
│    • Model:                   CLIP ViT-B/32 or ViT-L/14     │
│    • Token dim:               512                           │
│    • Max tokens:              77                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 13. Innovation Summary

```
┌──────────────────────────────────────────────────────────────┐
│              NOVEL CONTRIBUTIONS                             │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. HIERARCHICAL TEXT-SHAPE ALIGNMENT                        │
│     ┌────────────────────────────────────────┐              │
│     │ Multi-Scale Cross-Attention:           │              │
│     │   • 256-dim: Part-level alignment      │              │
│     │   • 512-dim: Semantic-level alignment  │              │
│     └────────────────────────────────────────┘              │
│                                                              │
│  2. STOP WORD FILTERING                                      │
│     ┌────────────────────────────────────────┐              │
│     │ Dynamically reduce attention weights   │              │
│     │ for meaningless tokens                 │              │
│     │ • "a", "the", "with" → weight × 0.1    │              │
│     │ • Focus on: "wooden", "armrests"       │              │
│     └────────────────────────────────────────┘              │
│                                                              │
│  3. ALIGNMENT LOSS IN CLIP SPACE                             │
│     ┌────────────────────────────────────────┐              │
│     │ Force semantic consistency between:    │              │
│     │ • Generated shape features             │              │
│     │ • Text embeddings                      │              │
│     │ via cosine similarity maximization     │              │
│     └────────────────────────────────────────┘              │
│                                                              │
│  4. VAE + DIFFUSION HYBRID                                   │
│     ┌────────────────────────────────────────┐              │
│     │ • PointNet VAE: Encode real shapes     │              │
│     │ • Diffusion Decoder: High-quality gen  │              │
│     │ • Best of both worlds                  │              │
│     └────────────────────────────────────────┘              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

이 아키텍처는 **CLIP 기반 텍스트 조건부 생성**, **계층적 어텐션**, 그리고 **불용어 필터링**을 통해
고품질의 텍스트-형상 정렬을 달성하는 혁신적인 3D 생성 모델입니다.
