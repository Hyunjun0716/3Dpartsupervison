# Text-Conditioned Point Cloud Diffusion VAE - Paper Architecture Diagrams

논문에 사용할 아키텍처 다이어그램 가이드

---

## Figure 1: Overall Architecture (Main Figure - 전체 구조)

이것을 논문의 메인 아키텍처 그림으로 사용하세요. 한 페이지에 전체 시스템을 보여줍니다.

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                   Text-Conditioned Point Cloud Diffusion VAE                             │
│                                                                                          │
│  Training Phase                                    Inference Phase                      │
├──────────────────────────────────────────────────┬───────────────────────────────────────┤
│                                                  │                                       │
│  Input Point Cloud                               │  Text Prompt                          │
│  x₀ ∈ ℝᴺˣ³ (N=1024)                             │  "a wooden chair                      │
│        │                                         │   with armrests"                      │
│        │                                         │        │                              │
│        ▼                                         │        ▼                              │
│  ┌──────────────┐                                │  ┌──────────────┐                     │
│  │   PointNet   │                                │  │ CLIP Text    │                     │
│  │   Encoder    │                                │  │ Encoder      │                     │
│  │   E_φ(x₀)    │                                │  │ T(c)         │                     │
│  └──────┬───────┘                                │  └──────┬───────┘                     │
│         │                                        │         │                             │
│         │ μ, σ ∈ ℝᶻᵈⁱᵐ                          │         │ 𝓣 = {t, p, ids}             │
│         ▼                                        │         │                             │
│  ┌──────────────┐      ┌──────────────┐         │         │                             │
│  │ z ~ q(z|x₀)  │      │ CLIP T(c)    │         │  ┌──────┴───────┐                     │
│  │ Latent Code  │      │ Text Emb 𝓣   │         │  │ z ~ p(z)     │                     │
│  └──────┬───────┘      └──────┬───────┘         │  │ Sample Prior │                     │
│         │                     │                 │  └──────┬───────┘                     │
│         │                     │                 │         │                             │
│         └──────────┬──────────┘                 │         │                             │
│                    │                            │         │                             │
│                    ▼                            │         ▼                             │
│         ┌────────────────────┐                  │  ┌────────────────────┐               │
│         │  Forward Diffusion │                  │  │ Reverse Diffusion  │               │
│         │  q(xₜ|x₀)          │                  │  │ p_θ(x₀|xₜ, z, 𝓣)  │               │
│         └──────────┬─────────┘                  │  └──────────┬─────────┘               │
│                    │                            │             │                         │
│         xₜ = √ᾱₜ x₀ + √(1-ᾱₜ) ε                │   xₜ ~ 𝓝(0,I), t=T→1                │
│                    │                            │             │                         │
│                    ▼                            │             ▼                         │
│  ┌───────────────────────────────────────────┐  │  ┌──────────────────────────────────┐ │
│  │     PointwiseNet Denoiser ε_θ             │  │  │  PointwiseNet Denoiser ε_θ       │ │
│  │  ┌─────────────────────────────────────┐  │  │  │ ┌─────────────────────────────┐  │ │
│  │  │                                     │  │  │  │ │                             │  │ │
│  │  │  Input: (xₜ, βₜ, z, 𝓣)             │  │  │  │ │ Input: (xₜ, βₜ, z, 𝓣)      │  │ │
│  │  │                                     │  │  │  │ │                             │  │ │
│  │  │  3 → 128 → 256                      │  │  │  │ │ Same architecture           │  │ │
│  │  │        ↓                            │  │  │  │ │                             │  │ │
│  │  │    Cross-Attn₂₅₆ ← 𝓣.tokens        │  │  │  │ │ generates clean x₀          │  │ │
│  │  │        ↓         (Part-level)      │  │  │  │ │                             │  │ │
│  │  │                                     │  │  │  │ │                             │  │ │
│  │  │  256 → 512                          │  │  │  │ │                             │  │ │
│  │  │        ↓                            │  │  │  │ │                             │  │ │
│  │  │    Cross-Attn₅₁₂ ← 𝓣.tokens        │  │  │  │ │                             │  │ │
│  │  │        ↓         (Semantic)        │  │  │  │ │                             │  │ │
│  │  │    FiLM ← 𝓣.pool                   │  │  │  │ │                             │  │ │
│  │  │        ↓                            │  │  │  │ │                             │  │ │
│  │  │  512 → 256 → 128 → 3                │  │  │  │ │                             │  │ │
│  │  │        ↓                            │  │  │  │ │                             │  │ │
│  │  │  Output: ε_θ(xₜ, βₜ, z, 𝓣)         │  │  │  │ │                             │  │ │
│  │  │                                     │  │  │  │ │                             │  │ │
│  │  └─────────────────────────────────────┘  │  │  │ └─────────────────────────────┘  │ │
│  └───────────────────┬───────────────────────┘  │  └──────────────┬───────────────────┘ │
│                      │                          │                 │                     │
│                      ▼                          │                 ▼                     │
│  ┌───────────────────────────────────────────┐  │  ┌──────────────────────────────────┐ │
│  │  Loss Computation                         │  │  │  Generated Point Cloud           │ │
│  │                                           │  │  │  x₀ ∈ ℝ¹⁰²⁴ˣ³                    │ │
│  │  𝓛 = λ_KL·𝓛_prior + 𝓛_recons + λ_align·𝓛_align  │  │                              │ │
│  │                                           │  │  │                                  │ │
│  │  𝓛_prior = -log p(z) - H[q(z|x₀)]        │  │  │                                  │ │
│  │  𝓛_recons = 𝔼ₜ[‖ε - ε_θ(xₜ,βₜ,z,𝓣)‖²]    │  │  │                                  │ │
│  │  𝓛_align = -cos⟨E_φ(x̂₀), 𝓣.pool⟩        │  │  │                                  │ │
│  │                                           │  │  │                                  │ │
│  └───────────────────────────────────────────┘  │  └──────────────────────────────────┘ │
│                                                  │                                       │
└──────────────────────────────────────────────────┴───────────────────────────────────────┘

Caption for Figure 1:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Overview of our text-conditioned point cloud diffusion VAE. Left: Training phase where the
PointNet encoder E_φ encodes input point clouds into latent codes z, while CLIP text encoder
T provides text embeddings 𝓣. The PointwiseNet denoiser ε_θ is trained with hierarchical
cross-attention (256-dim for part-level, 512-dim for semantic-level) and FiLM conditioning.
Right: Inference phase where we sample z from prior and generate point clouds conditioned on
text prompts through reverse diffusion. The model learns three objectives: prior matching
(𝓛_prior), denoising (𝓛_recons), and text-shape alignment in CLIP space (𝓛_align).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Figure 2: PointwiseNet Architecture (Detail Figure - 디테일 구조)

PointwiseNet의 상세 구조를 별도 figure로 제공하세요.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PointwiseNet Denoiser Architecture                    │
│                           ε_θ(xₜ, βₜ, z, 𝓣)                                 │
└─────────────────────────────────────────────────────────────────────────────┘

Inputs:
  • xₜ ∈ ℝᴺˣ³: Noisy point cloud at timestep t
  • βₜ ∈ ℝ: Noise schedule value
  • z ∈ ℝᶻᵈⁱᵐ: Latent code from encoder/prior
  • 𝓣 = {tokens, pool, ids}: Text embeddings from CLIP

┌────────────────────────────────────────────────────────────────────────────┐
│  Context Embedding                                                         │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  βₜ → [βₜ, sin(βₜ), cos(βₜ)] ∈ ℝ³              Time embedding             │
│  z → Linear(z) ∈ ℝ²⁵⁶                           Latent projection          │
│  ctx = Concat[time_emb, z_proj] ∈ ℝ²⁵⁹         Context vector            │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

                                   xₜ ∈ ℝᴺˣ³
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Layer 0: ConcatSquashLinear(3 → 128, ctx_dim=259)                      │
│  ┌────────────────────────────────────────────────────────┐             │
│  │  Hypernetwork-based conditional layer:                 │             │
│  │  gate = σ(W_gate(ctx))  ∈ ℝ¹²⁸                         │             │
│  │  bias = W_bias(ctx)      ∈ ℝ¹²⁸                         │             │
│  │  h = W(xₜ) ⊙ gate + bias                               │             │
│  └────────────────────────────────────────────────────────┘             │
│  Activation: LeakyReLU(0.2)                                             │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │ h₁ ∈ ℝᴺˣ¹²⁸
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Layer 1: ConcatSquashLinear(128 → 256, ctx_dim=259)                    │
│  + LeakyReLU(0.2)                                                        │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │ h₂ ∈ ℝᴺˣ²⁵⁶
                                 ├──────────────────────────────────┐
                                 │                                  │
                                 ▼                                  │
┌──────────────────────────────────────────────────┐                │
│  Multi-Head Cross-Attention (Part-level)        │                │
│  ┌────────────────────────────────────────────┐  │                │
│  │  Query: Q = W_q(h₂) ∈ ℝᴺˣ²⁵⁶              │  │                │
│  │  Key:   K = W_k(𝓣.tokens) ∈ ℝ⁷⁷ˣ²⁵⁶        │  │                │
│  │  Value: V = W_v(𝓣.tokens) ∈ ℝ⁷⁷ˣ²⁵⁶        │  │                │
│  │                                            │  │                │
│  │  Heads: 4, dim_head: 64                   │  │                │
│  │                                            │  │                │
│  │  Attn = Softmax(QKᵀ/√64)                  │  │                │
│  │                                            │  │                │
│  │  ┌──────────────────────────────────────┐ │  │                │
│  │  │  Stop Word Filtering:                │ │  │                │
│  │  │  For j ∈ {1,...,77}:                 │ │  │                │
│  │  │    if decode(𝓣.ids[j]) ∈ STOP_WORDS  │ │  │                │
│  │  │      Attn[:,:,:,j] *= 0.1            │ │  │                │
│  │  │  Attn = Attn / sum(Attn, dim=-1)     │ │  │                │
│  │  └──────────────────────────────────────┘ │  │                │
│  │                                            │  │                │
│  │  Out = W_o(Attn · V) ∈ ℝᴺˣ²⁵⁶             │  │                │
│  └────────────────────────────────────────────┘  │                │
└──────────────────────────┬───────────────────────┘                │
                           │ attn_out × 0.5                         │
                           └──────────────────┬─────────────────────┘
                                              ▼
                                      h₂ + 0.5·attn_out
                                              │
                                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Layer 2: ConcatSquashLinear(256 → 512, ctx_dim=259)                    │
│  + LeakyReLU(0.2)                                                        │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │ h₃ ∈ ℝᴺˣ⁵¹²
                                 ├──────────────┬───────────────────┐
                                 │              │                   │
                                 ▼              ▼                   │
┌──────────────────────────┐  ┌────────────────────────┐           │
│  Cross-Attention         │  │  FiLM Modulation       │           │
│  (Semantic-level)        │  │  ┌──────────────────┐  │           │
│  ┌────────────────────┐  │  │  │ γ = W_γ(𝓣.pool)  │  │           │
│  │ Heads: 8           │  │  │  │ β = W_β(𝓣.pool)  │  │           │
│  │ dim_head: 64       │  │  │  │                  │  │           │
│  │ + Stop Word Filter │  │  │  │ h = γ ⊙ h + β   │  │           │
│  │ Out ∈ ℝᴺˣ⁵¹²       │  │  │  └──────────────────┘  │           │
│  └────────────────────┘  │  └────────────────────────┘           │
└──────────┬───────────────┘              │                         │
           │ × 0.5                        │                         │
           └─────────────┬────────────────┘                         │
                         └───────────────────┬──────────────────────┘
                                             ▼
                                   h₃ + 0.5·attn + film
                                             │
                                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Layer 3: ConcatSquashLinear(512 → 256, ctx_dim=259)                    │
│  + LeakyReLU(0.2)                                                        │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │ h₄ ∈ ℝᴺˣ²⁵⁶
                                 ├──────────────────────┐
                                 │                      │
                                 ▼                      │
                        ┌────────────────┐              │
                        │ FiLM           │              │
                        │ (𝓣.pool)       │              │
                        └───────┬────────┘              │
                                │                       │
                                └───────┬───────────────┘
                                        ▼
                                  h₄ + film
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Layer 4: ConcatSquashLinear(256 → 128, ctx_dim=259)                    │
│  + LeakyReLU(0.2)                                                        │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │ h₅ ∈ ℝᴺˣ¹²⁸
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Layer 5: ConcatSquashLinear(128 → 3, ctx_dim=259)                      │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
                          ε_θ ∈ ℝᴺˣ³
                     (Predicted Noise)

Caption for Figure 2:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Detailed architecture of the PointwiseNet denoiser. The network takes noisy points xₜ, noise
level βₜ, latent code z, and text embeddings 𝓣 as input. Time and latent information are
injected through ConcatSquashLinear layers (hypernetwork-based gating). Multi-scale cross-
attention aligns point features with text tokens at two levels: 256-dim for part-level
features (e.g., "legs", "armrests") and 512-dim for semantic features (e.g., "chair",
"wooden"). Stop word filtering reduces attention weights on uninformative tokens by 90%.
FiLM layers provide global text conditioning using pooled CLIP features.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Figure 3: Cross-Attention with Stop Word Filtering (Innovation Figure)

이것은 당신의 핵심 기여를 보여주는 figure입니다.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│         Multi-Head Cross-Attention with Stop Word Filtering                │
└─────────────────────────────────────────────────────────────────────────────┘

Input Caption: "a wooden chair with armrests"
Tokens: [SOS, a, wooden, chair, with, armrests, EOS, PAD, ..., PAD]
         └─┘  └──────┘  └───┘  └──┘  └───────┘
       Stop   Content  Content Stop   Content
       Word    Word     Word   Word     Word

┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 1: Standard Multi-Head Attention                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Q ∈ ℝᴺˣᵈ                    K, V ∈ ℝ⁷⁷ˣᵈ                                 │
│  (Point features)             (Text tokens from CLIP)                      │
│         │                            │                                     │
│         └──────────┬─────────────────┘                                     │
│                    │                                                       │
│                    ▼                                                       │
│         Attn = Softmax(QKᵀ/√d_head)                                        │
│                    │                                                       │
│                    ▼                                                       │
│         ┌─────────────────────────────────────────┐                        │
│         │  Attention Map (Before Filtering)      │                        │
│         │  Shape: (B, heads, N_points, 77)       │                        │
│         │                                         │                        │
│         │  Point 1:  [0.15, 0.20, 0.25, 0.10, 0.15, 0.10, 0.03, ...]     │
│         │              ↑     ↑     ↑     ↑     ↑                          │
│         │              a   wooden chair with armrests                     │
│         │            stop content    stop content                         │
│         └─────────────────────────────────────────┘                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 2: Stop Word Filtering                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STOP_WORDS = {"a", "an", "the", "with", "of", "in", "on", ...}           │
│                                                                             │
│  For each token position j:                                                │
│    1. Decode token: token_str = CLIP_tokenizer.decode(token_ids[j])       │
│    2. Check if stop word: is_stop = (token_str in STOP_WORDS)             │
│    3. Assign weight:                                                       │
│         filter_mask[j] = 0.1   if is_stop                                 │
│                        = 1.0   otherwise                                  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────┐               │
│  │  Filter Mask:                                           │               │
│  │  [0.1,  1.0,   1.0,   0.1,   1.0,   1.0, ...]          │               │
│  │    ↑     ↑      ↑      ↑      ↑      ↑                 │               │
│  │    a  wooden  chair  with  armrests EOS ...            │               │
│  └─────────────────────────────────────────────────────────┘               │
│                                                                             │
│  Attn_filtered = Attn ⊙ filter_mask                                        │
│  Attn_filtered = Attn_filtered / sum(Attn_filtered, dim=-1)  (Renormalize)│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 3: Filtered Attention (After Filtering)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│         ┌─────────────────────────────────────────┐                        │
│         │  Attention Map (After Filtering)       │                        │
│         │  Shape: (B, heads, N_points, 77)       │                        │
│         │                                         │                        │
│         │  Point 1:  [0.02, 0.29, 0.36, 0.01, 0.22, 0.07, 0.02, ...]     │
│         │              ↑     ↑     ↑     ↑     ↑                          │
│         │              a   wooden chair with armrests                     │
│         │           reduced   ↑     ↑   reduced   ↑                       │
│         │                  enhanced  enhanced  enhanced                   │
│         └─────────────────────────────────────────┘                        │
│                                                                             │
│  Output = Attn_filtered · V                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Visualization: Attention Weight Redistribution                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Before Filtering:                                                         │
│  ┌───┬─────┬─────┬───┬─────────┬───┐                                      │
│  │ a │wood │chair│with│armrests│...│                                      │
│  ├───┼─────┼─────┼───┼─────────┼───┤                                      │
│  │15%│ 20% │ 25% │10%│  15%    │15%│                                      │
│  └───┴─────┴─────┴───┴─────────┴───┘                                      │
│                                                                             │
│  After Filtering (stop word penalty = 0.1):                                │
│  ┌───┬─────┬─────┬───┬─────────┬───┐                                      │
│  │ a │wood │chair│with│armrests│...│                                      │
│  ├───┼─────┼─────┼───┼─────────┼───┤                                      │
│  │ 2%│ 29% │ 36% │ 1%│  22%    │10%│  ← Renormalized                     │
│  └───┴─────┴─────┴───┴─────────┴───┘                                      │
│   ↓    ↑     ↑    ↓     ↑                                                 │
│  Down  Up    Up  Down   Up                                                 │
│                                                                             │
│  Key Insight: Content-bearing words receive more attention,               │
│               improving text-shape alignment quality.                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Caption for Figure 3:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stop word filtering mechanism in cross-attention. Given text caption, we first compute
standard attention weights between point features and text tokens. Then, we identify stop
words (e.g., "a", "the", "with") by decoding token IDs and reduce their attention weights
by 90% (×0.1). The attention map is renormalized, redistributing weights to content-bearing
words (e.g., "wooden", "chair", "armrests"). This focuses the model on semantically
meaningful tokens, improving text-shape alignment.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Figure 4: Training Objectives (Loss Figure)

손실 함수를 시각적으로 설명하는 figure입니다.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Training Objectives                                 │
│                                                                             │
│         𝓛_total = λ_KL · 𝓛_prior + 𝓛_recons + λ_align · 𝓛_align            │
│                  (0.001)                            (0.1)                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  (a) Prior Matching Loss: 𝓛_prior                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Goal: Match posterior q(z|x) to prior p(z) = 𝓝(0, I)                      │
│                                                                             │
│  x₀ → PointNet → (μ, log σ²)                                               │
│                       ↓                                                     │
│                  z = μ + σ ⊙ ε,  ε ~ 𝓝(0,I)                                │
│                       ↓                                                     │
│         ┌─────────────────────────────────────┐                            │
│         │ q(z|x) = 𝓝(z; μ, diag(σ²))         │                            │
│         └─────────────────┬───────────────────┘                            │
│                           │                                                │
│         ┌─────────────────┴───────────────────┐                            │
│         │  p(z) = 𝓝(z; 0, I)                 │                            │
│         └─────────────────┬───────────────────┘                            │
│                           │                                                │
│         𝓛_prior = KL[q(z|x) || p(z)] - H[q(z|x)]                          │
│                 = -log p(z) - H[q]                                         │
│                 = ½ Σᵢ[μᵢ² + σᵢ² - log σᵢ² - 1]                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  (b) Denoising Loss: 𝓛_recons                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Goal: Train denoiser to predict noise ε added to x₀                       │
│                                                                             │
│  x₀ ────┐                                                                  │
│         │ Sample t ~ Uniform(1, T)                                         │
│         │ Sample ε ~ 𝓝(0, I)                                               │
│         │                                                                  │
│         └──→ xₜ = √ᾱₜ x₀ + √(1-ᾱₜ) ε                                       │
│                 ↓                                                          │
│         ┌───────────────────┐                                              │
│         │  PointwiseNet     │                                              │
│         │  ε_θ(xₜ,βₜ,z,𝓣)   │                                              │
│         └────────┬──────────┘                                              │
│                  │                                                          │
│         ┌────────┴───────────┐                                             │
│         │  Predicted noise   │                                             │
│         │  ε_θ ∈ ℝᴺˣ³        │                                             │
│         └────────┬───────────┘                                             │
│                  │                                                          │
│         𝓛_recons = 𝔼ₜ,ε[‖ε - ε_θ(xₜ, βₜ, z, 𝓣)‖²]                         │
│                  = MSE(ε, ε_θ)                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  (c) Text-Shape Alignment Loss: 𝓛_align                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Goal: Align generated shapes with text in CLIP embedding space            │
│                                                                             │
│  xₜ, ε_θ ──→ x̂₀ = (xₜ - √(1-ᾱₜ) ε_θ) / √ᾱₜ   (Predict clean x₀)          │
│              ↓                                                              │
│         ┌────────────┐                                                     │
│         │  PointNet  │                                                     │
│         │  Encoder   │                                                     │
│         └─────┬──────┘                                                     │
│               │                                                             │
│               ▼                                                             │
│      Shape Features: f_s ∈ ℝ⁵¹²                                            │
│                                                                             │
│  Caption ──→ CLIP ──→ Text Features: f_t ∈ ℝ⁵¹² (𝓣.pool)                  │
│                                                                             │
│         ┌──────────┬──────────┐                                            │
│         │   f_s    │   f_t    │                                            │
│         └─────┬────┴────┬─────┘                                            │
│               │         │                                                  │
│               └────┬────┘                                                  │
│                    │                                                        │
│         Normalize: f̂_s = f_s / ‖f_s‖,  f̂_t = f_t / ‖f_t‖                  │
│                    │                                                        │
│         Similarity: s = ⟨f̂_s, f̂_t⟩ = Σᵢ f̂_s,i · f̂_t,i                    │
│                    │                                                        │
│         𝓛_align = -s  (Maximize similarity → Minimize negative)            │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │  CLIP Space Alignment                                    │              │
│  │                                                           │              │
│  │    ┌─────────┐                    ┌─────────┐            │              │
│  │    │Generated│                    │  Text   │            │              │
│  │    │ Shape   │◄──────close───────►│"wooden  │            │              │
│  │    │         │   (high similarity)│ chair"  │            │              │
│  │    └─────────┘                    └─────────┘            │              │
│  │        ↕                                                  │              │
│  │        │ far                                              │              │
│  │        ↕                                                  │              │
│  │    ┌─────────┐                                            │              │
│  │    │Unrelated│                                            │              │
│  │    │  Text   │                                            │              │
│  │    │"airplane"│                                           │              │
│  │    └─────────┘                                            │              │
│  └──────────────────────────────────────────────────────────┘              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Caption for Figure 4:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Three training objectives of our model. (a) Prior matching loss 𝓛_prior ensures the
posterior distribution q(z|x) matches the prior p(z)=𝓝(0,I) via KL divergence. (b) Denoising
loss 𝓛_recons trains the network to predict noise ε added to clean point clouds. (c)
Alignment loss 𝓛_align enforces semantic consistency between generated shapes and text
descriptions in CLIP embedding space by maximizing cosine similarity. The total loss is a
weighted combination: 𝓛 = 0.001·𝓛_prior + 𝓛_recons + 0.1·𝓛_align.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Figure 5: Diffusion Process (Conceptual Figure)

Forward/Reverse diffusion을 직관적으로 보여주는 figure입니다.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              Forward and Reverse Diffusion Process                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Forward Process (Training): q(xₜ|x₀)                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  x₀        x₂₅       x₅₀       x₇₅       x₁₀₀                              │
│  🪑   →    🪑̃    →    ░▒    →    ▓▓    →    ██                             │
│ Clean   Low Noise  Medium   High Noise  Pure                               │
│ Chair              Noise               Gaussian                            │
│                                                                             │
│  ᾱ₁₀₀=0.01 ←─── ᾱ₇₅=0.05 ←─── ᾱ₅₀=0.20 ←─── ᾱ₂₅=0.50 ←─── ᾱ₀=1.0         │
│                                                                             │
│  xₜ = √ᾱₜ x₀ + √(1-ᾱₜ) ε,  where ε ~ 𝓝(0, I)                               │
│                                                                             │
│  • As t increases: signal decreases, noise increases                       │
│  • At t=T: xₜ ≈ 𝓝(0, I), original shape information lost                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Reverse Process (Sampling): p_θ(xₜ₋₁|xₜ, z, 𝓣)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  x₁₀₀      x₇₅       x₅₀       x₂₅       x₀                                │
│  ██    →    ▓▓    →    ░▒    →    🪑̃    →    🪑                             │
│  Pure    High Noise  Medium   Low Noise   Clean                            │
│ Gaussian             Noise               "wooden                           │
│  Noise                                    chair"                           │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────┐               │
│  │  At each step t:                                        │               │
│  │                                                          │               │
│  │  1. Predict noise: ε_θ = PointwiseNet(xₜ, βₜ, z, 𝓣)    │               │
│  │                                                          │               │
│  │  2. Estimate clean: x̂₀ = (xₜ - √(1-ᾱₜ)ε_θ) / √ᾱₜ       │               │
│  │                                                          │               │
│  │  3. Compute mean: μ_θ = (xₜ - (1-αₜ)/√(1-ᾱₜ)·ε_θ)/√αₜ  │               │
│  │                                                          │               │
│  │  4. Add noise (if t>1): xₜ₋₁ = μ_θ + σₜ·𝓝(0,I)         │               │
│  │                                                          │               │
│  └─────────────────────────────────────────────────────────┘               │
│                                                                             │
│  Text Conditioning: "a wooden chair with armrests"                         │
│  • Cross-Attention: Aligns point features with text tokens                 │
│  • FiLM: Global style modulation                                           │
│  • Stop Word Filtering: Focus on "wooden", "chair", "armrests"            │
│                                                                             │
│  Latent Conditioning: z ∈ ℝ²⁵⁶                                              │
│  • Controls global shape structure                                         │
│  • Sampled from 𝓝(0, I) or learned prior (FlowVAE)                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Variance Schedule: β₁, β₂, ..., β_T                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  βₜ  │                                                    ╱                │
│      │                                                ╱                    │
│ 0.02 │                                            ╱                        │
│      │                                        ╱                            │
│      │                                    ╱                                │
│      │                                ╱                                    │
│ 0.01 │                            ╱                                        │
│      │                        ╱                                            │
│      │                    ╱                                                │
│      │                ╱                                                    │
│1e-4  │────────────╱─────────────────────────────────────────────          │
│      └────────────────────────────────────────────────────────── t        │
│      0           25          50          75          100                  │
│                                                                             │
│  Linear schedule: βₜ = β₁ + (β_T - β₁) · t/T                               │
│  αₜ = 1 - βₜ                                                               │
│  ᾱₜ = ∏ᵢ₌₁ᵗ αᵢ  (cumulative product)                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Caption for Figure 5:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Illustration of forward and reverse diffusion processes. Top: In the forward process,
Gaussian noise is gradually added to clean point clouds over T=100 steps, eventually
destroying all shape information. Bottom: In the reverse process, we start from pure noise
and iteratively denoise using the learned PointwiseNet conditioned on latent code z and text
embeddings 𝓣, recovering a clean point cloud that matches the text description. The variance
schedule βₜ increases linearly from 1e-4 to 0.02.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 논문 작성 권장사항

### Method 섹션 구성:

```
3. Method

3.1 Overview
  → Figure 1 참조

3.2 Point Cloud Encoding
  → PointNet VAE 설명
  → Latent space z ∈ ℝ²⁵⁶

3.3 Text Encoding
  → CLIP text encoder
  → Output: {tokens, pool, token_ids}

3.4 Diffusion Process
  → Figure 5 참조
  → Forward: q(xₜ|x₀)
  → Reverse: p_θ(xₜ₋₁|xₜ, z, 𝓣)

3.5 PointwiseNet Architecture
  → Figure 2 참조
  → ConcatSquashLinear layers
  → Multi-scale cross-attention
  → FiLM modulation

3.6 Stop Word Filtering (핵심 기여!)
  → Figure 3 참조
  → Motivation
  → Implementation
  → Benefits

3.7 Training Objectives
  → Figure 4 참조
  → Prior loss
  → Reconstruction loss
  → Alignment loss (핵심 기여!)

3.8 Implementation Details
  → Hyperparameters (Table 1)
```

---

## 추가 제안: Algorithm Pseudocode

논문에 알고리즘 의사코드도 포함하면 좋습니다:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Algorithm 1: Training                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: Dataset D = {(x, c)} of point clouds and captions                  │
│  Output: Trained model parameters θ, φ                                     │
│                                                                             │
│  1:  Initialize PointNet encoder E_φ, CLIP encoder T, denoiser ε_θ         │
│  2:  Initialize optimizer (Adam, lr=1e-3)                                  │
│  3:  for epoch = 1 to num_epochs do                                        │
│  4:      for (x, c) in DataLoader(D, batch_size=32) do                     │
│  5:          // Encode shape                                               │
│  6:          μ, log σ² ← E_φ(x)                                            │
│  7:          z ← μ + exp(0.5 log σ²) ⊙ ε₁, where ε₁ ~ 𝓝(0, I)             │
│  8:                                                                         │
│  9:          // Encode text                                                │
│ 10:          𝓣 ← T(c)  // {tokens, pool, token_ids}                        │
│ 11:                                                                         │
│ 12:          // Forward diffusion                                          │
│ 13:          t ~ Uniform(1, T)                                             │
│ 14:          ε₂ ~ 𝓝(0, I)                                                  │
│ 15:          xₜ ← √ᾱₜ x + √(1-ᾱₜ) ε₂                                       │
│ 16:                                                                         │
│ 17:          // Predict noise                                              │
│ 18:          ε_θ ← PointwiseNet(xₜ, βₜ, z, 𝓣)                              │
│ 19:                                                                         │
│ 20:          // Compute losses                                             │
│ 21:          𝓛_prior ← -log p(z) - H[q(z|x)]                               │
│ 22:          𝓛_recons ← ‖ε₂ - ε_θ‖²                                        │
│ 23:                                                                         │
│ 24:          // Alignment loss                                             │
│ 25:          x̂₀ ← (xₜ - √(1-ᾱₜ) ε_θ) / √ᾱₜ                                 │
│ 26:          f_s, _ ← E_φ(x̂₀)                                              │
│ 27:          𝓛_align ← -cos_sim(f_s, 𝓣.pool)                               │
│ 28:                                                                         │
│ 29:          // Total loss                                                 │
│ 30:          𝓛 ← 0.001·𝓛_prior + 𝓛_recons + 0.1·𝓛_align                    │
│ 31:                                                                         │
│ 32:          // Backprop                                                   │
│ 33:          𝓛.backward()                                                  │
│ 34:          clip_grad_norm(θ, φ, max_norm=10)                             │
│ 35:          optimizer.step()                                              │
│ 36:      end for                                                           │
│ 37:  end for                                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Algorithm 2: Inference (Sampling)                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: Text caption c, trained model ε_θ                                  │
│  Output: Generated point cloud x₀                                          │
│                                                                             │
│  1:  // Encode text                                                        │
│  2:  𝓣 ← CLIP_encoder(c)                                                   │
│  3:                                                                         │
│  4:  // Sample latent                                                      │
│  5:  z ~ 𝓝(0, I)  // or z = flow(w) for FlowVAE                            │
│  6:                                                                         │
│  7:  // Initialize from noise                                              │
│  8:  x_T ~ 𝓝(0, I)                                                         │
│  9:                                                                         │
│ 10:  // Reverse diffusion                                                  │
│ 11:  for t = T to 1 do                                                     │
│ 12:      // Predict noise                                                  │
│ 13:      ε_θ ← PointwiseNet(xₜ, βₜ, z, 𝓣)                                  │
│ 14:                                                                         │
│ 15:      // Compute denoising mean                                         │
│ 16:      μ_θ ← (xₜ - (1-αₜ)/√(1-ᾱₜ) · ε_θ) / √αₜ                          │
│ 17:                                                                         │
│ 18:      // Compute variance                                               │
│ 19:      σₜ² ← (1-ᾱₜ₋₁)/(1-ᾱₜ) · βₜ                                        │
│ 20:                                                                         │
│ 21:      // Sample next step                                               │
│ 22:      if t > 1 then                                                     │
│ 23:          xₜ₋₁ ← μ_θ + σₜ · 𝓝(0, I)                                     │
│ 24:      else                                                              │
│ 25:          xₜ₋₁ ← μ_θ  // No noise at final step                        │
│ 26:      end if                                                            │
│ 27:  end for                                                               │
│ 28:                                                                         │
│ 29:  return x₀                                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Table 1: Hyperparameters

```
┌────────────────────────────────────────┬──────────────────┐
│ Parameter                              │ Value            │
├────────────────────────────────────────┼──────────────────┤
│ Architecture                           │                  │
│   Latent dimension (zdim)              │ 256              │
│   Number of points (N)                 │ 1024             │
│   Denoiser layers                      │ 6                │
│   Hidden dimensions                    │ 128→256→512→...  │
│                                        │                  │
│ Diffusion                              │                  │
│   Number of timesteps (T)              │ 100              │
│   β₁ (beta_1)                          │ 1e-4             │
│   β_T (beta_T)                         │ 0.02             │
│   Schedule type                        │ Linear           │
│                                        │                  │
│ Attention                              │                  │
│   Cross-attn heads (256-dim)           │ 4                │
│   Cross-attn heads (512-dim)           │ 8                │
│   Attention head dimension             │ 64               │
│   Stop word penalty                    │ 0.1              │
│                                        │                  │
│ Training                               │                  │
│   Batch size                           │ 32               │
│   Learning rate                        │ 1e-3             │
│   Optimizer                            │ Adam             │
│   β₁ (Adam)                            │ 0.9              │
│   β₂ (Adam)                            │ 0.999            │
│   Weight decay                         │ 0                │
│   Gradient clip norm                   │ 10.0             │
│   Training iterations                  │ 100k             │
│   LR schedule                          │ Linear (50k→100k)│
│                                        │                  │
│ Loss weights                           │                  │
│   λ_KL (kl_weight)                     │ 0.001            │
│   λ_recons (implicit)                  │ 1.0              │
│   λ_align (align_weight)               │ 0.1              │
│                                        │                  │
│ Text encoder                           │                  │
│   Model                                │ CLIP ViT-B/32    │
│   Token dimension                      │ 512              │
│   Max tokens                           │ 77               │
│                                        │                  │
└────────────────────────────────────────┴──────────────────┘
```

---

## LaTeX 코드 예시

논문에서 수식을 작성할 때 사용할 LaTeX 코드:

```latex
% Forward diffusion
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)\mathbf{I})

% Reverse diffusion
p_\theta(x_{t-1} | x_t, z, \mathcal{T}) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, \beta_t, z, \mathcal{T}), \sigma_t^2 \mathbf{I})

% Total loss
\mathcal{L} = \lambda_{KL} \mathcal{L}_{\text{prior}} + \mathcal{L}_{\text{recons}} + \lambda_{\text{align}} \mathcal{L}_{\text{align}}

% Prior loss
\mathcal{L}_{\text{prior}} = -\log p(z) - \mathcal{H}[q(z|x)]

% Reconstruction loss
\mathcal{L}_{\text{recons}} = \mathbb{E}_{t,\epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, \beta_t, z, \mathcal{T})\|^2\right]

% Alignment loss
\mathcal{L}_{\text{align}} = -\text{cos\_sim}(E_\phi(\hat{x}_0), \mathcal{T}.\text{pool})

% Clean prediction
\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta}{\sqrt{\bar{\alpha}_t}}

% Stop word filtering
\text{Attn}_{\text{filtered}} = \frac{\text{Attn} \odot M}{\sum(\text{Attn} \odot M)}
\quad \text{where} \quad M_j = \begin{cases}
0.1 & \text{if token}_j \in \text{STOP\_WORDS} \\
1.0 & \text{otherwise}
\end{cases}
```

이제 논문에 사용할 준비가 완료되었습니다! 🎓
