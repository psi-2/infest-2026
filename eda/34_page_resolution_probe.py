"""34 - eda/22 found page acc rising with resolution (v2-view .60, 448 .69, 640 .73). Does it keep rising at 768 / 896?
Same protocol as eda/22 (frozen DINOv3-S, [CLS, mean patch], LR on non-line train of other folds, held-out train pages).
Token cost: 640^2 = 1600, 768^2 = 2304, 896^2 = 3136 tokens (ViT-L training ~1.4x / ~2x the 640 page cost)."""
import importlib

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from common import LABELS, ROOT, load_frames, open_rgb, out_dir, pmap

M = importlib.import_module("22_page_view_probe")


def views(path):
    g = open_rgb(path).convert("L")
    return {f"sq{S}_gray": np.asarray(M.letterbox(g, S, 255).convert("RGB")) for S in (768, 896)}


if __name__ == "__main__":
    df = load_frames()
    ct = pd.concat([pd.read_csv(out_dir("08_v2_gap_decomposition") / f)[["image_id", "split", "ctype"]] for f in ["train_types_oof.csv", "test_types_pred.csv"]])
    df = df.merge(ct, on=["image_id", "split"])
    d = df[((df.split == "train") & (df.ctype != "line")) | ((df.split == "test") & (df.ctype == "page"))].reset_index(drop=True)
    d["fold"] = pd.read_csv(ROOT / "data/folds.csv").set_index("image_id").fold.reindex(d.image_id).to_numpy()
    f = out_dir("22_page_view_probe") / "feats_small_hires.npz"
    if f.exists():
        F = dict(np.load(f))
    else:
        V = pmap(views, list(d.path), procs=8)
        F = {k: M.embed([v[k] for v in V]) for k in V[0]}; np.savez(f, **F)
    F.update({k: v for k, v in np.load(out_dir("22_page_view_probe") / "feats_small.npz").items() if k in ("sq448_gray", "sq640_gray")})
    y = d.label.map({l: i for i, l in enumerate(LABELS)}).fillna(-1).astype(int).to_numpy()
    tr = (d.split == "train").to_numpy(); pg = tr & (d.ctype == "page").to_numpy()
    for k in ["sq448_gray", "sq640_gray", "sq768_gray", "sq896_gray"]:
        X = F[k] / np.linalg.norm(F[k], axis=1, keepdims=True)
        for C in [2, 8]:
            p = np.full(len(d), -1)
            for fo in range(5):
                a, b = tr & (d.fold != fo).to_numpy(), tr & (d.fold == fo).to_numpy()
                p[b] = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(X[a], y[a]).predict(X[b])
            print(f"   {k:11s} C={C}: page acc {np.mean(p[pg] == y[pg]):.3f} | page F1 {f1_score(y[pg], p[pg], average='macro'):.3f} | block acc {np.mean(p[tr & (d.ctype == 'block').to_numpy()] == y[tr & (d.ctype == 'block').to_numpy()]):.3f}")
