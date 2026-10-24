# SCRIBE: Nusantara Script Classification

**1st place, [INFEST XII 2026 Data Science](https://www.kaggle.com/competitions/infest-xii-data-science/leaderboard)** (Institut Teknologi Bandung). Team **DoaAyahIbu**.

**S**ynthetic-**C**orruption and **R**outed **I**nference across **B**ackbone **E**xperts. The model classifies an image of text into one of seven Indonesian scripts: Bali, Jawa, Jawi, Lampung, Lontara, Pegon, Sunda.

| Leaderboard | Macro F1 | Rank | Submission |
|---|---|---|---|
| Public | 0.91820 | 10th | Hybrid B |
| Private | 0.95305 | 2nd | v5 + v11 ensemble |
| **Final (leaderboard + notebook + presentation)** | | **1st** | |

- Final notebook: [`notebooks/DoaAyahIbu_Institut_Teknologi_Bandung.ipynb`](notebooks/DoaAyahIbu_Institut_Teknologi_Bandung.ipynb) (Hybrid B: public 0.91820, private 0.93349)
- Presentation: [`docs/INFEST XII.pdf`](docs/INFEST%20XII.pdf)

## Problem

The baseline reached 0.976 on a random holdout but only 0.863 on the public leaderboard. EDA showed the cause was distribution shift, not model capacity:

| Finding | Train | Test |
|---|---|---|
| Blurry / colored images | 12% / 3% | 69% / 44% |
| Skewed (rotation ≥ 3°) | 4% | 30% |
| Full-page photos (height ≥ 250 px) | 5.3% | 24.6% |
| Train images with near-duplicates | 312 (8.3%) | 17 test copies of train |

The test lines are clean train-like scans with synthetic corruption added (gray triangles, black bars, tint, blur, JPEG). Pages shrunk onto a line canvas leave characters about 3.5 px tall.

## Approach

```
raw image ─┬─ height ≥ 250 px → Page expert  : grayscale, letterbox 640×640      (DINOv3 ViT-L/16, 5 folds)
           └─ otherwise       → Line expert  : ink crop, height 160, 640 windows (DINOv3 ViT-L/16, 5 folds)
                                                trained on synthetic corruption
```

1. **Group validation.** Near-duplicates are found with pHash/dHash, DINOv3-S embeddings and pixel correlation, then split with `StratifiedGroupKFold` (5 folds). Validation is scored on corrupted images so the CV score reflects the test set.
2. **Synthetic corruption.** Every test artifact is reproduced at its measured rate and applied to the raw image before preprocessing: rotation ±15°, tint, blur/pixelation, gray triangles, black bars, JPEG, noise.
3. **Routed experts.** Original image height routes pages and lines to separate fine-tuned DINOv3 ViT-L/16 models. The 640×640 page view raises correct pages from 160 to 168 of 199.
4. **Post-processing.** The 17 test images that are copies of train images take their train label.

## Results

| Version | Change | OOF F1 | Public | Private |
|---|---|---|---|---|
| v1 | Letterbox, random holdout | 0.976* | 0.86289 | 0.87594 |
| v2 | Synthetic corruption + group CV (line expert) | 0.977 | 0.91514 | 0.91946 |
| **Hybrid B** | v2 lines + v5 page expert (final notebook) | **0.9801** | **0.91820** | 0.93349 |
| v5 + v11 | Hybrid averaged with SigLIP2 NaFlex | 0.9836 | 0.91815 | **0.95305** |

\* random holdout; all other OOF scores are 5-fold group CV on corrupted images.

Public and private rankings across the team's 12 submissions correlate weakly (Spearman 0.58), so public score differences below about 0.004 are noise. See [`analysis/lb_public_vs_private.py`](analysis/lb_public_vs_private.py).

## Repository

```
notebooks/   final end-to-end notebook (training, inference, submission)
docs/        presentation and competition description
eda/         37 analysis scripts; eda/README.md summarizes all findings
prepro/      preprocessing pipeline, corruption calibration and visual checks
analysis/    pre-submission checks and public vs private LB analysis
results/     config, metrics and submission per version; results/final = submitted file
```

Each script writes its figures and tables to `<folder>/outputs/<script_name>/`.

## Reproduce

1. Download the competition data from Kaggle into `data/` (`train.csv`, `test.csv`, `sample_submission.csv`, `images/`). The data is not redistributed.
2. Run the notebook top to bottom. It is self-contained: duplicate groups, folds, both experts and the final submission are built inside it.

Tested on one RTX 5090 (PyTorch ≥ 2.7, CUDA 12.8, `timm ≥ 1.0.27`); training both experts takes about 1.5 to 2 hours.

Only competition data was used. No external data and no pseudo-labeling; the pretrained DINOv3 weights serve as the fine-tuning base.

## Team

M. Fachri Alfaridzi · Riantama Putra · Tiara Kusuma W.

## References

- Adilazuarda et al., *NusaAksara: A Multimodal and Multilingual Benchmark for Preserving Indonesian Indigenous Scripts*, ACL 2025.
- Gomez et al., *Improving patch-based scene text script identification with ensembles of conjoined networks*, [arXiv:1602.07480](https://arxiv.org/abs/1602.07480).
- Bhunia et al., *Script Identification in Natural Scene Image and Video Frame using Attention based Convolutional-LSTM Network*, [arXiv:1801.00470](https://arxiv.org/abs/1801.00470).
- *Patch Aggregator for Scene Text Script Identification*, [arXiv:1912.03818](https://arxiv.org/abs/1912.03818).
