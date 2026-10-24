"""37 - The 'jawi attractor': every change the LB punished moved uncertain rows TO jawi (v3-only: pegon->jawi 7,
lontara->jawi 6; C: lontara->jawi 3; all run-ensembles of eda/36: lontara/pegon/bali->jawi). jawi is 21% of train but
~12% of test (eda/15 EM & BBSE). An uncertain model falls back on the TRAIN prior -> jawi.
Prior correction on B's probabilities (non-page v2, page v5), temperature-calibrated on the matching OOF:
  1. method check: resample OOF rows to the test prior (EM estimate) -> does EM correction raise macro-F1 there? (sim)
  2. apply to test with strength alpha in {0.5, 1.0}: rows changed vs B, transitions, direction w.r.t. jawi
  3. consistency with LB evidence: on the rows where v3-only / C changes were punished, does the correction keep B?
"""
import importlib

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score

from common import LABELS, ROOT, out_dir

P15 = importlib.import_module("15_prior_shift")
OUT = out_dir("37_prior_on_B")
L = np.array(LABELS)


def oof_logits(d):
    idx = np.concatenate([np.load(d / f"fold{k}_val_idx.npy") for k in range(5)])
    out = {}
    for kind in ["clean", "corrupt"]:
        x = np.concatenate([np.load(d / f"fold{k}_oof_{kind}_logits.npy") for k in range(5)]); r = np.empty_like(x); r[idx] = x; out[kind] = r
    return out


if __name__ == "__main__":
    train = pd.read_csv(ROOT / "data/train.csv"); y = train.label.map({l: i for i, l in enumerate(LABELS)}).to_numpy()
    ct = pd.read_csv(out_dir("08_v2_gap_decomposition") / "train_types_oof.csv").set_index("image_id").ctype.reindex(train.image_id).to_numpy()
    o2, o5 = oof_logits(ROOT / "results/v2/main"), oof_logits(ROOT / "results/v5/main")
    lo = np.where((ct == "page")[:, None], o5["clean"], o2["corrupt"])
    nll = lambda T: -np.log(P15.softmax(lo, T)[np.arange(len(y)), y] + 1e-12).mean()
    T = minimize_scalar(nll, bounds=(0.3, 5), method="bounded").x
    po = P15.softmax(lo, T); src = po.mean(0)
    print(f"== temperature T={T:.2f}; model's effective train prior (mean OOF prob): {dict(zip(LABELS, (src * 100).round(1)))}")

    tv = pd.read_csv(ROOT / "analysis/outputs/v5_submission_check/test_versions.csv"); pg = (tv.ctype == "page").to_numpy()
    t2 = np.stack([np.load(ROOT / f"results/v2/main/fold{k}_test_logits.npy") for k in range(5)])
    t5 = np.stack([np.load(ROOT / f"results/v5/main/fold{k}_test_logits.npy") for k in range(5)])
    pt = np.where(pg[:, None], P15.softmax(t5, T).mean(0), P15.softmax(t2, T).mean(0))
    pri, _ = P15.em_prior(pt, src)
    print(f"== EM test prior on B: {dict(zip(LABELS, (pri * 100).round(1)))}")

    print("\n== 1. method check: OOF resampled to the EM test prior (500 draws of 1220 rows)")
    rng = np.random.default_rng(0); cls = [np.where(y == c)[0] for c in range(7)]
    res = {a: [] for a in ["none", "0.5", "1.0"]}
    for _ in range(300):
        idx = np.concatenate([rng.choice(cls[c], rng.binomial(1220, pri[c]), replace=True) for c in range(7)])
        p = po[idx]; yy = y[idx]
        pr, _ = P15.em_prior(p, src)
        res["none"].append(f1_score(yy, p.argmax(1), average="macro"))
        for a in ["0.5", "1.0"]:
            res[a].append(f1_score(yy, (p * (pr / src) ** float(a)).argmax(1), average="macro"))
    for a, v in res.items():
        print(f"   alpha={a:4s}: macro-F1 {np.mean(v):.4f} (vs none {np.mean(np.array(v) - np.array(res['none'])):+.4f}, P(better) {np.mean(np.array(v) > np.array(res['none'])):.2f})")

    print("\n== 2. applied to B's test probabilities")
    B = tv.B.to_numpy(dtype=object)
    twin = tv.image_id.isin(pd.read_csv(ROOT / "data/test_train_twins.csv").image_id).to_numpy()
    for a in [0.5, 1.0]:
        adj = L[(pt * (pri / src) ** a).argmax(1)].astype(object); adj[twin] = B[twin]
        ch = adj != B
        tr_ = pd.Series([f"{u}->{v}" for u, v in zip(B[ch], adj[ch])]).value_counts()
        print(f"   alpha={a}: {ch.sum()} rows changed | by type {pd.Series(tv.ctype[ch]).value_counts().to_dict()}")
        print(f"      from jawi {int(sum(v for k, v in tr_.items() if k.startswith('jawi->')))} | to jawi {int(sum(v for k, v in tr_.items() if k.endswith('->jawi')))} | top {tr_.head(8).to_dict()}")
        pd.DataFrame({"image_id": tv.image_id, "label": adj}).to_csv(OUT / f"submission_B_prior_a{a}.csv", index=False)

    print("\n== 3. rows the LB punished (v3-only changes, C changes): does the correction side with B or with the punished label?")
    v3 = tv.v3.to_numpy(dtype=object); v2 = tv.v2.to_numpy(dtype=object)
    C = pd.read_csv(ROOT / "results/v6/main/submission_C_targeted.csv").set_index("image_id").label.reindex(tv.image_id).to_numpy(dtype=object)
    adj = L[(pt * (pri / src)).argmax(1)]
    for name, alt in [("v3-only (vs v2)", v3), ("C (vs B)", C)]:
        base = v2 if name.startswith("v3") else B
        m = (alt != base) & ~((alt == B) & (base != B))
        print(f"   {name}: {m.sum()} rows | corrected B keeps base label {np.mean(adj[m] == base[m]):.2f} | takes punished label {np.mean(adj[m] == alt[m]):.2f}")
    print("->", OUT)
