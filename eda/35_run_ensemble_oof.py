"""35 - Which combination of EXISTING runs is best on OOF (same folds.csv for all runs)? Non-page: corrupted OOF; page: clean.
Runs: v2 (seed 42), v4 seeds 7 / 2026 (v2 recipe), v5 (joint page view, v2 prepro), v6 (v6 prepro + page view).
eda/30: public-LB noise for a 10-40 row change is sd .003-.004 -> larger than the +.0018 we need; decisions must rest on
OOF evidence over thousands of rows, with a paired bootstrap over dup groups.
Also: how many TEST rows each OOF-best combination changes vs B (LB .91820).
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from common import LABELS, ROOT, out_dir


def sm(x):
    e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)


def load(d, prefix):
    idx = np.concatenate([np.load(d / f"{prefix}{k}_val_idx.npy") for k in range(5)])
    out = {}
    for kind in ["clean", "corrupt"]:
        p = sm(np.concatenate([np.load(d / f"{prefix}{k}_oof_{kind}_logits.npy") for k in range(5)])); r = np.empty_like(p); r[idx] = p; out[kind] = r
    out["test"] = np.mean([sm(np.load(d / f"{prefix}{k}_test_logits.npy")) for k in range(5)], 0)
    return out


if __name__ == "__main__":
    train = pd.read_csv(ROOT / "data/train.csv"); y = train.label.map({l: i for i, l in enumerate(LABELS)}).to_numpy()
    ct = pd.read_csv(out_dir("08_v2_gap_decomposition") / "train_types_oof.csv").set_index("image_id").ctype.reindex(train.image_id).to_numpy()
    grp = pd.read_csv(ROOT / "data/folds.csv").set_index("image_id").dup_group.reindex(train.image_id).to_numpy()
    R = {"v2": load(ROOT / "results/v2/main", "fold"), "v4s7": load(ROOT / "results/v4", "cv_s7_f"), "v4s2026": load(ROOT / "results/v4", "cv_s2026_f"),
         "v5": load(ROOT / "results/v5/main", "fold"), "v6": load(ROOT / "results/v6/main", "fold")}
    np_ = ct != "page"; pg = ct == "page"
    rows = []
    for n in range(1, 6):
        for combo in itertools.combinations(R, n):
            pn = np.mean([R[c]["corrupt"] for c in combo], 0); pp = np.mean([R[c]["clean"] for c in combo], 0)
            tl = np.where(pg[:, None], pp, pn)
            rows.append(dict(combo="+".join(combo), nonpage_acc=(pn.argmax(1) == y)[np_].mean(), page_acc=(pp.argmax(1) == y)[pg].mean(),
                             f1_testlike=f1_score(y, tl.argmax(1), average="macro")))
    r = pd.DataFrame(rows)
    print("== non-page (corrupted OOF), top 8 and singles"); print(r.sort_values("nonpage_acc", ascending=False).head(8)[["combo", "nonpage_acc"]].round(4).to_string(index=False))
    print(r[r.combo.isin(R)][["combo", "nonpage_acc", "page_acc"]].round(4).to_string(index=False))
    print("== page (clean OOF), top 6"); print(r.sort_values("page_acc", ascending=False).head(6)[["combo", "page_acc"]].round(4).to_string(index=False))
    # paired bootstrap: best non-page combo vs v2 (the non-page part of B)
    best = r.sort_values("nonpage_acc", ascending=False).combo.iloc[0]
    pb = np.mean([R[c]["corrupt"] for c in best.split("+")], 0).argmax(1) == y
    p2 = R["v2"]["corrupt"].argmax(1) == y
    d = (pb.astype(float) - p2)[np_]; g = grp[np_]; ug = np.unique(g); rng = np.random.default_rng(0)
    ix = {k: np.where(g == k)[0] for k in ug}
    bs = [np.concatenate([d[ix[k]] for k in rng.choice(ug, len(ug))]).mean() for _ in range(1000)]
    print(f"\n== non-page best '{best}' vs v2: {d.mean() * np_.sum():+.1f} rows of {np_.sum()} (acc {d.mean():+.4f}, CI [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}])"
          f" | fixes {int(((pb & ~p2) & np_).sum())} breaks {int(((p2 & ~pb) & np_).sum())}")
    tv = pd.read_csv(ROOT / "analysis/outputs/v5_submission_check/test_versions.csv")
    tpg = (tv.ctype == "page").to_numpy()
    tb = np.mean([R[c]["test"] for c in best.split("+")], 0).argmax(1)
    Bl = tv.B.map({l: i for i, l in enumerate(LABELS)}).to_numpy()
    print(f"   test: non-page rows where '{best}' differs from B: {int(((tb != Bl) & ~tpg).sum())}")
    r.to_csv(out_dir("35_run_ensemble_oof") / "combos.csv", index=False)
