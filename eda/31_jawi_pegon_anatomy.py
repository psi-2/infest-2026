"""31 - jawi vs pegon: the biggest uncertain pair of B (38 low-conf rows, eda/26). Both are Arabic script; pegon = Javanese /
Sundanese written in Arabic letters, usually FULLY VOWELLED (harakat) in pesantren texts; jawi = Malay, usually unvowelled,
and in THIS train set jawi also holds printed numerals / Latin (eda/15). Measurable, text-free cues on the ink:
  * diacritic density: share of small connected components (area < 6% of the median glyph-body area) among all components
  * slant: dominant stroke orientation (pegon gloss is often slanted, eda/15 blocks)
  * print vs handwriting: stroke-width variation (std/mean of distance-transform on the skeleton)
Per class x content type in TRAIN (true labels) and in TEST (B labels, confident vs uncertain).
Theory: if pegon = many diacritics, anything that erases 2-4 px dots (blur, down-up sampling, JPEG, crop_to_ink trim)
removes the one feature that separates the classes -> check how much corrupt() + preprocess destroys diacritics.
"""
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from common import ROOT, open_rgb, out_dir, pmap

sys.path.insert(0, str(ROOT / "prepro" / "outputs"))
import pipeline_v2_snapshot as V2  # noqa: E402

OUT = out_dir("31_jawi_pegon_anatomy")


def anatomy(g):
    ink = np.asarray(g) < 128
    lab, n = ndi.label(ink)
    if n < 3:
        return dict(n_cc=n, diac=np.nan, sw_cv=np.nan, slant=np.nan)
    areas = np.bincount(lab.ravel())[1:]
    body = np.median(areas[areas >= np.percentile(areas, 50)])
    diac = float((areas < 0.06 * body).mean())
    dt = ndi.distance_transform_edt(ink); sk = ink & (dt >= ndi.maximum_filter(dt, 3)) & (dt > 0)  # DT ridge ~ skeleton
    w = dt[sk]; sw_cv = float(w.std() / (w.mean() + 1e-6)) if w.size > 20 else np.nan
    gy, gx = np.gradient(ndi.gaussian_filter(np.asarray(g, np.float32), 1))
    ang = np.degrees(np.arctan2(gy, gx))[ink & (np.hypot(gx, gy) > 10)]
    slant = float(np.median(np.abs(((ang + 90) % 180) - 90))) if ang.size else np.nan
    return dict(n_cc=n, diac=diac, sw_cv=sw_cv, slant=slant)


def job(args):
    path, corrupt_seed = args
    img = V2.load_rgb(path)
    out = {f"clean_{k}": v for k, v in anatomy(V2.preprocess(img)).items()}
    if corrupt_seed is not None:
        out.update({f"cor_{k}": v for k, v in anatomy(V2.preprocess(V2.corrupt(img, random.Random(corrupt_seed)))).items()})
    return out


if __name__ == "__main__":
    tr = pd.read_csv(out_dir("08_v2_gap_decomposition") / "train_types_oof.csv")
    tr = tr[tr.label.isin(["jawi", "pegon"]) & tr.ctype.isin(["line", "block"])]
    u = pd.read_csv(out_dir("26_uncertainty_map_B") / "uncertainty.csv")
    te = u[u.B.isin(["jawi", "pegon"]) & u.ctype.isin(["line", "block"])].copy()
    T = pd.DataFrame(pmap(job, [(str(ROOT / "data/images/train" / i), k) for k, i in enumerate(tr.image_id)]))
    tr = pd.concat([tr.reset_index(drop=True), T], axis=1)
    E = pd.DataFrame(pmap(job, [(str(ROOT / "data/images/test" / i), None) for i in te.image_id]))
    te = pd.concat([te.reset_index(drop=True), E], axis=1); te["grp"] = np.where(te.conf > .9, "conf>.9", "conf<=.9")
    cols = ["diac", "sw_cv", "slant", "n_cc"]
    print("== TRAIN (true label), clean preprocess, medians")
    print(tr.groupby(["ctype", "label"])[[f"clean_{c}" for c in cols]].median().round(3).to_string())
    print("\n== TRAIN after corrupt()+preprocess (what the model trains on)")
    print(tr.groupby(["ctype", "label"])[[f"cor_{c}" for c in cols]].median().round(3).to_string())
    kept = (tr.cor_diac / tr.clean_diac).replace([np.inf], np.nan)
    print(f"   diacritic share kept after corruption (median ratio): {kept.groupby(tr.label).median().round(2).to_dict()}")
    print("\n== TEST (B label), preprocess only (test is already corrupted)")
    print(te.groupby(["ctype", "B", "grp"])[[f"clean_{c}" for c in cols]].median().round(3).to_string())
    from sklearn.metrics import roc_auc_score
    for c in cols:
        m = tr[f"clean_{c}"].notna()
        print(f"   train AUC pegon-vs-jawi by {c}: clean {roc_auc_score(tr.label[m] == 'pegon', tr[f'clean_{c}'][m]):.3f}", end="")
        m2 = tr[f"cor_{c}"].notna()
        print(f" | corrupted {roc_auc_score(tr.label[m2] == 'pegon', tr[f'cor_{c}'][m2]):.3f}")
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for a, c in zip(axes, ["diac", "sw_cv", "slant"]):
        for lab, col in [("jawi", "tab:blue"), ("pegon", "tab:red")]:
            a.hist(tr[tr.label == lab][f"clean_{c}"].dropna(), bins=30, alpha=.4, density=True, color=col, label=f"train {lab} clean")
            a.hist(tr[tr.label == lab][f"cor_{c}"].dropna(), bins=30, histtype="step", density=True, color=col, label=f"train {lab} corrupt")
        a.set_title(c); a.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(OUT / "jawi_pegon_features.png", dpi=70); plt.close()
    fig, axes = plt.subplots(4, 4, figsize=(28, 8))
    for r, (lab, ct) in enumerate([("jawi", "line"), ("pegon", "line"), ("jawi", "block"), ("pegon", "block")]):
        for c, x in enumerate(tr[(tr.label == lab) & (tr.ctype == ct)].sample(4, random_state=0).itertuples()):
            axes[r, c].imshow(open_rgb(ROOT / "data/images/train" / x.image_id)); axes[r, c].set_title(f"{lab} {ct} diac={x.clean_diac:.2f}", fontsize=10); axes[r, c].axis("off")
    plt.tight_layout(); plt.savefig(OUT / "train_examples.png", dpi=50); plt.close()
    tr.to_csv(OUT / "train_anatomy.csv", index=False); te.to_csv(OUT / "test_anatomy.csv", index=False)
    print("->", OUT)
