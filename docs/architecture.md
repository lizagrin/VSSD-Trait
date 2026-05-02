# Architecture

VSSD-Trait is a three-branch multimodal regressor for the **Big Five
personality traits** (Openness, Conscientiousness, Extraversion,
Agreeableness, Neuroticism) on 15-second self-presentation video clips
from the **First Impressions V2** corpus.

![figure_0_architecture.jpg](..%2Fresults%2Ffigures%2Ffigure_0_architecture.jpg)

## Visual branch (`src/av_traits/models/visual.py`)

* **Backbone** — VSSD-Small (`vssd_small_mesa.pth`), the ICCV-2025
  state-space-duality vision model.  Replaceable with `ResNet-50` or
  `ViT-B/16` (`backbone_kind: vssd | resnet50 | vit_b16`) for the
  architectural ablation in Table 2.
* **Frame pooling** — attention pool (default) or mean pool over the
  K = 8 frames inside each window.
* **Window encoder** — `Linear(D+1 → H)` followed by a
  2-layer **biGRU** plus a single-block **Multi-head Self-attention**
  (residual + LayerNorm), giving every window a global temporal
  context.
* **Window pooling** — attention pool over the W = 14 windows. The
  attention weights αw are exposed in the forward output and power
  Table 9 (temporal saliency).

## Audio branch (`src/av_traits/models/audio.py`)

Two parallel streams:

| Stream | Inputs | Architecture | Output dim |
| ------ | ------ | ------------ | ---------- |
| Deep   | log-mel (128 × T) | 2-D CNN → adaptive avg-pool → MLP | 256 |
| HC     | prosody (38) ⊕ eGeMAPS (88) | 2-layer MLP | 128 |

Concatenated → `Linear(384 → 512)`.

## Text branch (`src/av_traits/models/text.py`)

* **SBERT** (`all-MiniLM-L6-v2`, 22.7 M params) — **frozen**, encodes
  the full transcript into a 384-dim sentence embedding.
* Small projection MLP `Linear(384) → 512` brings it into the shared
  hidden space.

The frozen weights are not counted in the trainable parameter budget
(Table 3).

## Trait-wise late fusion (`src/av_traits/models/fusion.py`)

For every trait *t* and modality *m* we compute a sigmoid score
ŷ_{t,m} via a per-modality linear head, plus a softmax gate
α_{t,m} over modalities (per trait). The final prediction is

```
ŷ_t = Σ_m α_{t,m} · ŷ_{t,m}
```

The per-trait gate weights — averaged over the test set — are reported
in Table 8.

## Auxiliary classification head (`BinHead`)

A small linear head on top of the joint embedding maps each trait to a
3-class soft label (low / mid / high, boundaries `0.40` / `0.60`).
Adds a cross-entropy term that empirically improves CCC by ~0.005.

## Loss (`src/av_traits/training/losses.py`)

```
L = L_huber                                        ← regression
  + λ_aux  · Σ_m L_huber(per-modality)            ← deep supervision
  + λ_ccc  · (1 − mean_CCC)                        ← align with metric
  + λ_bin  · CE(bin_logits)                        ← classification
  + λ_pair · L_pairwise(joint_emb)                 ← metric learning
```

Default coefficients: `λ_aux = 0.15, λ_ccc = 0.5, λ_bin = 0.2,
λ_pair = 0.1`. Disable `aux` or `ccc` via `cfg.use_aux_heads = False`
or `cfg.use_ccc_loss = False`.

## Training schedule

A 3-stage staged-unfreezing curriculum:

| Stage | Epochs | Backbone   | LR (heads) | LR (backbone) |
| ----- | ------ | ---------- | ---------- | ------------- |
| 1     | 2      | frozen     | 3 × 10⁻⁴  | 0             |
| 2     | 4      | last stage | 1.5 × 10⁻⁴| 8 × 10⁻⁶     |
| 3     | 6      | all        | 8 × 10⁻⁵  | 3 × 10⁻⁶     |

Linear warmup (8% of steps) → half-cosine decay; AMP enabled; gradient
accumulation = 2; max-norm clipping at 1.0.
