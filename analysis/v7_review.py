"""Why did v7 (friend: ArcFace + SupCon, full train, k-NN read-out, LB 0.88808) score below v2 (0.91514)?
v7's notebook has no outputs, so this tests its DESIGN choices on frozen DINOv3-S features of the same 1500 non-page train
images (prepro/12 cache: clean canvas and corrupt() canvas of each image), group 5-fold (data/folds.csv):
  a. k-NN, reference = CLEAN train tiles (v7: ref_tiles built from uncorrupted train_pp), query = CORRUPTED val  <- v7
  b. k-NN, reference = corrupted train, query = corrupted val
  c. linear head (logistic regression) trained on corrupted train, query corrupted val  (~ v2 recipe)
  d. k-NN clean->clean (what v7 would look like on a clean holdout)
Also structural differences vs v2/v5 that the LB also pays for: no page view (v5 +0.003), single model on full data
(no fold ensemble, no epoch selection), no twin override.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eda"))
from common import LABELS, out_dir  # noqa: E402

OUT = ROOT / "analysis/outputs/v7_review"; OUT.mkdir(parents=True, exist_ok=True)


def knn(ref, ry, q, k=5, t=0.03):
    s = q @ ref.T
    idx = np.argpartition(-s, k - 1, axis=1)[:, :k]; ss = np.take_along_axis(s, idx, 1); w = np.exp((ss - ss.max(1, keepdims=True)) / t)
    p = np.zeros((len(q), 7))
    for c in range(7):
        p[:, c] = np.where(ry[idx] == c, w, 0).sum(1)
    return p.argmax(1)


if __name__ == "__main__":
    z = np.load(ROOT / "prepro/outputs/12_domain_gap_deep/emb.npz"); Xc, Xk = z["c"], z["k"]
    tr = pd.read_csv(out_dir("08_v2_gap_decomposition") / "train_types_oof.csv"); tr = tr[tr.ctype != "page"].sample(1500, random_state=0).reset_index(drop=True)
    y = tr.label.map({l: i for i, l in enumerate(LABELS)}).to_numpy()
    fold = pd.read_csv(ROOT / "data/folds.csv").set_index("image_id").fold.reindex(tr.image_id).to_numpy()
    P = {k: np.zeros(len(y), int) for k in ["a_knn_cleanref_corruptq (v7)", "b_knn_corruptref_corruptq", "c_linear_corrupt (v2-like)", "d_knn_clean_clean"]}
    for f in range(5):
        a, b = fold != f, fold == f
        P["a_knn_cleanref_corruptq (v7)"][b] = knn(Xc[a], y[a], Xk[b])
        P["b_knn_corruptref_corruptq"][b] = knn(Xk[a], y[a], Xk[b])
        P["c_linear_corrupt (v2-like)"][b] = LogisticRegression(C=2, max_iter=3000, class_weight="balanced").fit(Xk[a], y[a]).predict(Xk[b])
        P["d_knn_clean_clean"][b] = knn(Xc[a], y[a], Xc[b])
    rows = [dict(setup=k, acc=np.mean(p == y), macro_f1=f1_score(y, p, average="macro"),
                 pegon_recall=np.mean(p[y == 5] == 5), jawi_recall=np.mean(p[y == 2] == 2)) for k, p in P.items()]
    r = pd.DataFrame(rows).round(3); print(r.to_string(index=False)); r.to_csv(OUT / "knn_vs_linear.csv", index=False)
    print("\nclean->clean vs clean-ref->corrupt-query gap = what a clean holdout hides; v7 had no holdout at all")
