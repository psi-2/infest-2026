# Final submission

`submission.csv` is the output of `notebooks/DoaAyahIbu_Institut_Teknologi_Bandung.ipynb` (hybrid B, 1,220 rows).

| | Macro F1 |
|---|---|
| Public LB | 0.91820 |
| Private LB | 0.93349 |

Non-page images use the v2 line expert (5 folds, 160×640 canvas); pages use the v5 page expert (640×640 view).

Prediction counts: bali 112, jawa 221, jawi 170, lampung 232, lontara 114, pegon 202, sunda 169.

The team's best private score, 0.95305, comes from `sub_v5_v11naflex_avg.csv` (v5 averaged with SigLIP2 NaFlex, public 0.91815). The comparison of all 12 submissions is in `analysis/lb_public_vs_private.py`.
