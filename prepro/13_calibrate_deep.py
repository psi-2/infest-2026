"""13 - Calibrate corrupt_v8 knobs against test with the deep domain classifier of prepro/12 (same 1500 train images,
same test embeddings, same canvas = v6 preprocess). One factor per variant, then the combination.
  domain AUC (corrupt train vs test): lower = model input closer to test         (v2 corrupt: .832)
  guard: frozen probe acc/F1, trained on corrupted canvases of other folds, scored on corrupted held-out fold
         (the augmentation must not buy a lower AUC by destroying the script)
  guard 2: probe trained on corrupted canvases, scored on CLEAN held-out canvases (clean train must stay easy)
Usage: python 13_calibrate_deep.py  -> table + per-class sample grid of the best variant.
"""
import importlib.util
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eda"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "outputs"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELS, ROOT, out_dir, pmap  # noqa: E402
import corrupt_v8 as C8  # noqa: E402

spec = importlib.util.spec_from_file_location("m12", Path(__file__).with_name("12_domain_gap_deep.py")); M12 = importlib.util.module_from_spec(spec); spec.loader.exec_module(M12)
OUT = Path(__file__).resolve().parent / "outputs" / "13_calibrate_deep"; OUT.mkdir(parents=True, exist_ok=True)

VARIANTS = {
    "v2": {},
    "occ": dict(p_occ=0.3, sliver=0.7, n_occ=(1, 4)),
    "vcrop": dict(p_vcrop=0.25),
    "noise_first": dict(noise_before_blur=True),
    "occ+vcrop": dict(p_occ=0.3, sliver=0.7, n_occ=(1, 4), p_vcrop=0.25),
    "occ+vcrop+noise_first": dict(p_occ=0.3, sliver=0.7, n_occ=(1, 4), p_vcrop=0.25, noise_before_blur=True),
}
CFG = None


def canvas(args):
    path, seed, name = args
    rng = random.Random(seed)
    img = M12.V2.load_rgb(path)
    if max(img.size) > 1600:
        img.thumbnail((1600, 1600))
    img = C8.corrupt(img, rng, VARIANTS[name])
    tiles = M12.V2.to_canvas(M12.pp6(img), train=False)
    return np.asarray(tiles[rng.randrange(len(tiles))], dtype=np.uint8)


if __name__ == "__main__":
    z = np.load(M12.OUT / "emb.npz"); Xt, Xc = z["t"], z["c"]
    tr = pd.read_csv(out_dir("08_v2_gap_decomposition") / "train_types_oof.csv"); tr = tr[tr.ctype != "page"].sample(1500, random_state=0).reset_index(drop=True)
    te = pd.read_csv(out_dir("26_uncertainty_map_B") / "uncertainty.csv"); te = te[te.ctype != "page"].reset_index(drop=True)
    y = tr.label.map({l: i for i, l in enumerate(LABELS)}).to_numpy()
    fold = pd.read_csv(ROOT / "data/folds.csv").set_index("image_id").fold.reindex(tr.image_id).to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    rows, arrays = [], {}
    for name in VARIANTS:
        f = OUT / f"emb_{name}.npz"
        if f.exists():
            d = np.load(f); Xk, Ak = d["x"], d["a"]
        else:
            Ak = np.stack(pmap(canvas, [(str(ROOT / "data/images/train" / i), k, name) for k, i in enumerate(tr.image_id)]))
            Xk = M12.embed(Ak); np.savez(f, x=Xk, a=Ak)
        arrays[name] = Ak
        Z = np.vstack([Xk, Xt]); yy = np.r_[np.zeros(len(Xk)), np.ones(len(Xt))]
        p = cross_val_predict(LogisticRegression(C=1, max_iter=3000), Z, yy, cv=cv, method="predict_proba")[:, 1]
        r = dict(variant=name, auc=roc_auc_score(yy, p))
        for t in ["line", "block", "glyph"]:
            m = np.r_[(tr.ctype == t).to_numpy(), (te.ctype == t).to_numpy()]
            r[f"auc_{t}"] = roc_auc_score(yy[m], p[m])
        pk, pc = np.zeros(len(y), int), np.zeros(len(y), int)
        for k in range(5):
            a, b = fold != k, fold == k
            mdl = LogisticRegression(C=2, max_iter=3000, class_weight="balanced").fit(Xk[a], y[a])
            pk[b], pc[b] = mdl.predict(Xk[b]), mdl.predict(Xc[b])
        r.update(probe_acc_corrupt=np.mean(pk == y), probe_f1_corrupt=f1_score(y, pk, average="macro"), probe_acc_clean=np.mean(pc == y))
        rows.append(r); print(r, flush=True)
    res = pd.DataFrame(rows).round(3); print(res.to_string(index=False)); res.to_csv(OUT / "variants.csv", index=False)
    best = res.sort_values("auc").variant.iloc[0]
    fig, axes = plt.subplots(len(LABELS), 3, figsize=(30, 1.6 * len(LABELS) * 1.2))
    for r_, lab in enumerate(LABELS):
        ks = np.where(y == r_)[0][:3]
        for c_, k in enumerate(ks):
            axes[r_, c_].imshow(arrays[best][k], cmap="gray", vmin=0, vmax=255, aspect="auto"); axes[r_, c_].set_title(f"{lab} {tr.image_id[k]} [{best}]", fontsize=10); axes[r_, c_].axis("off")
    plt.tight_layout(); plt.savefig(OUT / f"samples_{best}.png", dpi=45); plt.close()
    print("best by AUC:", best, "->", OUT)
