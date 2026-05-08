# Attention 시각화 가이드 (3-Level Analysis)

## 개요

이 프로젝트는 **3단계 Multi-scale Cross-Attention 시각화**를 제공합니다:
- **Level 1**: Global Token Weights (전체적인 단어 중요도)
- **Level 2**: Token→Point Heatmap (단어가 어느 포인트에 영향을 주는가)
- **Level 3**: Multi-scale Comparison (256-dim vs 512-dim 비교)

---

## Level 1: Global Token Weights (전체 단어 중요도)

### 방법
```python
# 평균화 과정:
# (B, heads, N_points, N_tokens) → (N_tokens,)
attn = attention_weights[0].mean(axis=0).mean(axis=0)
```

### 계산 과정
1. **Batch 제거**: 첫 번째 샘플만 사용 `[0]`
2. **Head 평균**: 8개 attention head를 평균 `mean(axis=0)`
3. **Point 평균**: 1024개 point를 평균 `mean(axis=0)`

### 시각화 형태
- **Horizontal Bar Chart**: 각 단어의 평균 attention weight
- **Top-K 강조**: 중요한 단어는 빨간색, 나머지는 하늘색
- **X축**: Attention Weight (0~1)
- **Y축**: 단어 목록

### 해석
- **높은 값**: 모델이 이 단어에 전반적으로 많은 attention을 줌
- **낮은 값**: 이 단어는 생성에 크게 영향을 주지 않음
- **Stop Word Filtering 효과**: "a", "the" 같은 단어의 weight가 ~10%로 감소

### 출력 파일
```
attention/level1_global_weights_sample_0000.png
```

---

## Level 2: Token→Point Spatial Visualization (단어-포인트 공간 대응)

Level 2는 **2가지 시각화**를 제공합니다:

### Level 2a: Token→Point Heatmap (Abstract View)

#### 방법
```python
# (B, heads, N_points, N_tokens) → (N_points, N_tokens)
attn = attention_weights[0].mean(axis=0)  # Head만 평균, Point는 유지
```

#### 계산 과정
1. **Batch 제거**: 첫 번째 샘플 `[0]`
2. **Head 평균**: 8개 head 평균
3. **Top-K 단어 선택**: 평균 attention이 높은 K개 단어 (default: 5)
4. **Top-K 포인트 선택**: 선택된 단어들에 높은 attention을 가진 K개 포인트 (default: 100)
5. **Row-wise Normalization**: 각 포인트별로 합이 1이 되도록 정규화

#### 시각화 형태
- **Heatmap**: Seaborn heatmap (YlOrRd colormap)
- **X축**: Top-K 중요 단어
- **Y축**: Top-K 관련 포인트 (P0, P1, P2, ...)
- **Color**: 밝을수록 높은 attention

#### 해석
- **가로줄(row)**: 특정 포인트가 어떤 단어들에 attention하는지
- **세로줄(column)**: 특정 단어가 어떤 포인트들에 영향을 주는지
- **밝은 영역**: 강한 word-point 연결
- **어두운 영역**: 약한 연결

#### 출력 파일
```
attention/level2a_token_point_heatmap_sample_0000.png
```

---

### Level 2b: 3D Point Cloud Visualization (Spatial View) ⭐ NEW!

#### 방법
```python
# 각 중요 단어마다 3D point cloud를 색칠
for word in top_k_words:
    word_attn = attention[:, word_idx]  # (N_points,)
    # Normalize to [0, 1]
    word_attn_norm = (word_attn - word_attn.min()) / (word_attn.max() - word_attn.min())

    # Color: Red = High attention, Blue = Low attention
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                        c=word_attn_norm, cmap='RdYlBu_r')
```

#### 계산 과정
1. **Batch & Head 평균**: `(B, heads, N_points, N_tokens) → (N_points, N_tokens)`
2. **Top-K 단어 선택**: 평균 attention이 높은 K개 단어 (default: 5)
3. **각 단어별 attention 추출**: `word_attn = attn[:, word_idx]` → (N_points,)
4. **정규화**: Min-max normalization to [0, 1]
5. **Color mapping**: RdYlBu_r colormap (빨강=높음, 파랑=낮음)

#### 시각화 형태
- **Grid of 3D Scatter Plots**: 2열 × N행 (단어별로 하나씩)
- **각 subplot**:
  - 제목: "Attention to [word]" + 평균/최대 attention 값
  - X, Y, Z 축: 3D 좌표
  - Color: 빨강(높은 attention) → 노랑 → 파랑(낮은 attention)
  - Viewing angle: elev=25°, azim=45°

#### 해석
- **빨간 영역**: 모델이 이 단어를 생성할 때 **집중한 공간 위치**
- **파란 영역**: 이 단어와 **관련 없는 공간 위치**
- **전체가 빨강**: 단어가 전역적 속성 (예: "wooden", "modern")
- **특정 부분만 빨강**: 단어가 특정 부품 (예: "legs", "armrest")

#### 예시
```
Caption: "a wooden chair with four legs"

📊 Level 2b 시각화 결과:

[Subplot 1] "chair"
- 전체적으로 골고루 빨강/노랑 → 전역 개념

[Subplot 2] "wooden"
- 좌석, 등받이, 다리 모두 빨강 → 재질 속성

[Subplot 3] "legs"
- 다리 부분만 빨강, 나머지는 파랑 → 명확한 공간 대응!

[Subplot 4] "four"
- 네 다리 영역만 빨강 → 개수 정보

[Subplot 5] "with"
- 전체적으로 낮은 attention (파랑) → Stop word filtering 효과
```

#### 장점
- ✅ **직관적**: 포인트 인덱스(P0, P1...)가 아닌 **실제 3D 공간**에서 확인
- ✅ **공간 대응**: "legs"라는 단어가 실제로 다리 부분에 attention하는지 검증
- ✅ **Stop word 검증**: "a", "the" 같은 단어가 낮은 attention을 받는지 시각적으로 확인
- ✅ **Part-level 이해**: 모델이 부품별로 단어를 이해하는지 분석 가능

#### 출력 파일
```
attention/level2b_3d_point_cloud_sample_0000.png
```

---

### Level 2 요약

| 시각화 | 형태 | 장점 | 단점 |
|--------|------|------|------|
| **2a: Heatmap** | 2D 히트맵 (포인트 인덱스) | 정량적, 많은 포인트 표현 가능 | 공간 정보 없음 (추상적) |
| **2b: 3D Cloud** | 3D 포인트 클라우드 | 직관적, 공간 대응 명확 | 각도에 따라 보이지 않는 부분 존재 |

**추천**: 둘 다 함께 사용!
- **2a**: 정량적 분석 (어느 포인트가 얼마나 attention받는지)
- **2b**: 정성적 검증 (실제로 올바른 부분에 attention하는지)

---

## Level 3: Multi-scale Comparison (256 vs 512)

### 방법
```python
# 256-dim attention (Local/Part level)
attn_256 = attention_weights_256[0].mean(axis=0).mean(axis=0)  # (N_tokens,)

# 512-dim attention (Global level)
attn_512 = attention_weights_512[0].mean(axis=0).mean(axis=0)  # (N_tokens,)

# KL Divergence 계산
from scipy.stats import entropy
kl_256_to_512 = entropy(attn_256, attn_512)
kl_512_to_256 = entropy(attn_512, attn_256)
```

### 시각화 형태 (2x2 subplot)

#### Plot 1 (좌상): Side-by-side Bar Chart
- **256-dim**: 하늘색 (Local/Part focus)
- **512-dim**: 연어색 (Global focus)
- 모든 단어에 대해 두 attention을 나란히 비교

#### Plot 2 (우상): Attention Difference
- **차이**: `512-dim - 256-dim`
- **녹색 바**: 512-dim이 더 높음 (더 global하게 중요)
- **빨간색 바**: 256-dim이 더 높음 (더 local하게 중요)
- **Zero line**: 차이 없음

#### Plot 3 (좌하): Top-K Words for 256-dim
- 256-dim에서 가장 중요한 K개 단어 (default: 10)
- **해석**: Local/Part level에서 중요한 단어
- **예상**: 구체적인 부품 이름 ("leg", "armrest", "seat")

#### Plot 4 (우하): Top-K Words for 512-dim
- 512-dim에서 가장 중요한 K개 단어
- **해석**: Global level에서 중요한 단어
- **예상**: 전반적인 형태/스타일 ("wooden", "modern", "comfortable")

### KL Divergence 해석
```
KL(256→512): 256-dim을 기준으로 512-dim이 얼마나 다른지
KL(512→256): 512-dim을 기준으로 256-dim이 얼마나 다른지
```

- **낮은 KL (~0.1)**: 두 attention이 유사 (비슷한 역할)
- **높은 KL (~1.0+)**: 두 attention이 매우 다름 (다른 역할)

### 기대되는 패턴
1. **256-dim (Local/Part)**:
   - 구체적 부품 단어에 높은 attention
   - 예: "legs", "armrests", "backrest", "wheels"

2. **512-dim (Global)**:
   - 전체 스타일/속성 단어에 높은 attention
   - 예: "wooden", "modern", "office", "comfortable"

### 출력 파일
```
attention/level3_multiscale_comparison_sample_0000.png
```

---

## 구현 상세

### 1. Model Forward Pass

```python
# models/diffusion.py:226-229
if return_attention:
    # Return both 256-dim and 512-dim attention weights
    return result, {'attn_256': attention_weights_256, 'attn_512': attention_weights_512}
return result
```

**변경 사항**:
- 이전: 512-dim만 반환
- 현재: Dict로 256-dim과 512-dim 모두 반환

### 2. Visualization Pipeline

```python
# visualize_training_samples.py:818-856
if isinstance(attention_weights, dict):
    attn_256 = attention_weights.get('attn_256')
    attn_512 = attention_weights.get('attn_512')

    # Level 1: Global weights
    visualize_attention_weights(attn_512, ...)

    # Level 2: Token-point heatmap
    visualize_token_point_heatmap(attn_512, ...)

    # Level 3: Multi-scale comparison
    visualize_multiscale_comparison(attn_256, attn_512, ...)
```

### 3. Stop Word Filtering

```python
# models/attention.py:93-95
# Apply stop word filtering (always active, both training and inference)
if self.use_stopword_filter and token_ids is not None and tokenizer is not None:
    attn = self._apply_stopword_filter(attn, token_ids, tokenizer)
```

**효과**:
- Stop words (a, the, with, of, ...) attention이 10%로 감소
- Level 1 시각화에서 확인 가능

---

## 사용 방법

### 실행 커맨드
```bash
python visualize_training_samples.py \
    --ckpt logs_gen/GEN_xxx/ckpt_xxx.pt \
    --num_samples 3 \
    --visualize_attention
```

### 출력 구조
```
save_dir/
├── attention/
│   ├── level1_global_weights_sample_0000.png           # Level 1: 전체 단어 중요도
│   ├── level2a_token_point_heatmap_sample_0000.png     # Level 2a: 히트맵
│   ├── level2b_3d_point_cloud_sample_0000.png          # Level 2b: 3D 시각화 ⭐
│   ├── level3_multiscale_comparison_sample_0000.png    # Level 3: 256 vs 512
│   ├── level1_global_weights_sample_0001.png
│   ├── level2a_token_point_heatmap_sample_0001.png
│   ├── level2b_3d_point_cloud_sample_0001.png
│   └── ...
├── comparison_page_1.png
├── best_worst_comparison.png
└── statistics.txt
```

**샘플당 4개 PNG 파일 생성**:
- Level 1 (1개): 단어 중요도 바 차트
- Level 2 (2개): 히트맵 + 3D 포인트 클라우드
- Level 3 (1개): Multi-scale 비교

---

## 기존 연구와의 비교

### Average Pooling (Level 1)
- **사용**: CLIP, DALL-E, Flamingo, Stable Diffusion
- **장점**: 간단, 전체 경향 파악
- **단점**: 공간 정보 손실

### Token-Point Heatmap (Level 2)
- **사용**: ViLT, ALBEF, BLIP
- **장점**: 단어-영역 대응 관계 명확
- **단점**: 3D point cloud에서는 시각화 복잡

### Multi-scale Comparison (Level 3)
- **사용**: 본 프로젝트 오리지널
- **장점**: Local vs Global 역할 구분 가능
- **특징**: 256-dim (part) vs 512-dim (global) 비교

---

## 주요 의존성

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Level 2에서 heatmap 생성
from scipy.stats import entropy  # Level 3에서 KL divergence 계산
```

---

## 문제 해결

### 1. "No attention weights" 경고
```
[WARNING] No attention weights or token IDs for sample 0
```
**해결**: `--visualize_attention` 플래그를 추가했는지 확인

### 2. Heatmap이 너무 작음
```python
# visualize_training_samples.py:287
# top_k_points를 조절
visualize_token_point_heatmap(..., top_k_points=50)  # 100 → 50
```

### 3. 단어가 겹침
```python
# visualize_training_samples.py:378
# fontsize 조절
ax1.set_xticklabels(content_tokens, rotation=45, ha='right', fontsize=6)  # 8 → 6
```

---

## 참고 논문

1. **Average Pooling**:
   - CLIP (Radford et al., 2021)
   - Flamingo (Alayrac et al., 2022)

2. **Cross-Attention Visualization**:
   - ViLT (Kim et al., 2021)
   - ALBEF (Li et al., 2021)

3. **Multi-scale Attention**:
   - Perceiver (Jaegle et al., 2021)
   - Hierarchical Vision Transformer (Liu et al., 2021)

4. **Stop Word Filtering**:
   - 본 프로젝트에서 구현
   - 일반적으로 0.1~0.2 penalty 사용

---

## 향후 개선 방향

### 1. 3D Point Cloud Overlay
```python
# Point cloud를 3D로 그리고 attention 기반 색칠
for word in ['wooden', 'legs']:
    point_attn = attention[:, word_idx]  # (N_points,)
    colors = cm.Reds(point_attn / point_attn.max())
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors)
```

### 2. Attention Flow (Gradient-based)
```python
# Grad-CAM 스타일로 어느 단어가 생성에 가장 영향을 미쳤는지
from torch.autograd import grad
gradients = grad(loss, attention_weights)
```

### 3. Head-wise Analysis
```python
# 각 attention head가 어떤 역할을 하는지 분석
for h in range(8):
    attn_h = attention_weights[0, h].mean(axis=0)
    # 각 head별로 시각화
```

---

## 요약

| Level | 목적 | 방법 | 출력 | 파일 |
|-------|------|------|------|------|
| **1** | 어떤 단어가 전반적으로 중요한가 | Average pooling (heads + points) | Bar chart | level1_global_weights_*.png |
| **2a** | 단어가 어느 포인트에 영향을 주었나 (정량) | Average heads, keep points | Heatmap | level2a_token_point_heatmap_*.png |
| **2b** | 단어가 3D 공간 어디에 영향을 주었나 (정성) | Color-coded 3D scatter | 3D Point Clouds | level2b_3d_point_cloud_*.png |
| **3** | Local vs Global 역할 구분 | 256 vs 512 비교 + KL divergence | 2x2 subplots | level3_multiscale_comparison_*.png |

**핵심**:
- Level 1: **WHAT** (어떤 단어가 중요한가)
- Level 2a: **WHERE (Abstract)** (어느 포인트 인덱스에)
- Level 2b: **WHERE (Spatial)** (3D 공간 어디에) ⭐ **직관적!**
- Level 3: **HOW** (어떤 수준에서 - Local/Global)
