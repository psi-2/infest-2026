"""33 - Page TTA on the frozen probe (eda/22 features, no new compute): does averaging views help held-out train pages?
single 640 / single 448 / mean(448,640) / mean(640, v2-view) / mean(448,640,v2-view). Group folds, several C, and
bootstrap CI of (combo - 640) over dup groups. If a combo helps a frozen probe it is a cheap TTA for v8's page head.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import LABELS, ROOT, load_frames, out_dir

if __name__ == "__main__":
    df = load_frames()
    ct = pd.concat([pd.read_csv(out_dir("08_v2_gap_decomposition") / f)[["image_id", "split", "ctype"]] for f in ["train_types_oof.csv", "test_types_pred.csv"]])
    df = df.merge(ct, on=["image_id", "split"])
    d = df[((df.split == "train") & (df.ctype != "line")) | ((df.split == "test") & (df.ctype == "page"))].reset_index(drop=True)
    fo = pd.read_csv(ROOT / "data/folds.csv").set_index("image_id")
    d["fold"] = fo.fold.reindex(d.image_id).to_numpy(); d["grp"] = fo.dup_group.reindex(d.image_id).to_numpy()
    F = dict(np.load(out_dir("22_page_view_probe") / "feats_small.npz"))
    y = d.label.map({l: i for i, l in enumerate(LABELS)}).fillna(-1).astype(int).to_numpy()
    tr = (d.split == "train").to_numpy(); pg = tr & (d.ctype == "page").to_numpy()
    P = {}
    for k in ["v2", "sq448_gray", "sq640_gray"]:
        X = F[k] / np.linalg.norm(F[k], axis=1, keepdims=True); p = np.zeros((len(d), 7))
        for f in range(5):
            a, b = tr & (d.fold != f).to_numpy(), tr & (d.fold == f).to_numpy()
            p[b] = LogisticRegression(C=4, max_iter=3000, class_weight="balanced").fit(X[a], y[a]).predict_proba(X[b])
        P[k] = p
    combos = {"640": ["sq640_gray"], "448": ["sq448_gray"], "448+640": ["sq448_gray", "sq640_gray"], "640+v2view": ["sq640_gray", "v2"],
              "448+640+v2view": ["sq448_gray", "sq640_gray", "v2"]}
    base = P["sq640_gray"].argmax(1) == y
    g = d.grp.to_numpy()[pg]; ug = np.unique(g); rng = np.random.default_rng(0); B = [rng.choice(ug, len(ug)) for _ in range(1000)]
    idx = {k: np.where(g == k)[0] for k in ug}
    for name, ks in combos.items():
        r = np.mean([P[k] for k in ks], 0).argmax(1) == y
        dd = (r.astype(float) - base)[pg]
        bs = [np.concatenate([dd[idx[k]] for k in b]).mean() for b in B]
        print(f"   {name:15s} page acc {r[pg].mean():.3f} | vs 640 {dd.mean():+.3f} CI [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
