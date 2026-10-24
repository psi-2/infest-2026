"""30 - Why did C (v6 targeted, 10 changed LINE rows vs B) score 0.91273, i.e. -0.00547 vs B 0.91820?
  1. LB forensics on the 10 rows (eda/12 simulate): q = share of C's changes that are right; plus how much of the drop the
     public/private split alone can explain (10 rows is tiny: which rows land in public matters a lot)
  2. 'corruption style' hypothesis: test images were corrupted per SOURCE; if the style (paper tint colour, ink colour,
     bar/noise type) is class-specific, a model can be right on text-less images and a 'readable' fix can hurt.
     Test: on CONFIDENT test lines (B conf>.9, labels as proxy), predict B's class from style features only (no text),
     group-CV. Same on train lines (no synthetic tint there) for reference.
"""
import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from common import LABELS, ROOT, open_rgb, out_dir, pmap

F12 = importlib.import_module("12_lb_forensics")
OUT = out_dir("30_forensics_C_and_style")
L = {l: i for i, l in enumerate(LABELS)}


def style(path):
    """text-free appearance: paper colour (bright pixels), ink colour (dark pixels), noise, bar share, size."""
    a = np.asarray(open_rgb(path), dtype=np.float32); g = a.mean(2)
    h, w = g.shape
    lo, hi = np.percentile(g, 15), np.percentile(g, 70)
    paper = a[g >= hi].mean(0); ink = a[g <= lo].mean(0)
    rows_dark = (g < 60).mean(1) > .9
    noise = float(np.median(np.abs(np.diff(g, axis=1))))
    return dict(p_r=paper[0], p_g=paper[1], p_b=paper[2], i_r=ink[0], i_g=ink[1], i_b=ink[2], p_sat=paper.max() - paper.min(),
                i_sat=ink.max() - ink.min(), bar=rows_dark.mean(), noise=noise, h=h, w=w, aspect=w / h)


if __name__ == "__main__":
    test = pd.read_csv(ROOT / "data/test.csv")
    rd = lambda p: pd.read_csv(p).set_index("image_id").label.reindex(test.image_id).map(L).values
    B = rd(ROOT / "analysis/outputs/v5_submission_check/submission_B_hybrid.csv")
    C = rd(ROOT / "results/v6/main/submission_C_targeted.csv")
    v2 = rd(ROOT / "results/v2/main/submission.csv")
    p2 = F12.softmax(np.stack([np.load(ROOT / f"results/v2/main/fold{k}_test_logits.npy") for k in range(5)])).mean(0)
    second = np.argsort(-p2, 1)[:, 1]
    rng = np.random.default_rng(1); a = 0.93
    for _ in range(8):
        truth = np.where(rng.random(len(p2)) < a, v2, second); a += (0.91514 - f1_score(truth, v2, average="macro")) * 0.9
    obs = 0.91273 - 0.91820
    print(f"== B -> C: {int((B != C).sum())} rows changed, observed {obs:+.5f}")
    rows = []
    for frac in [0.3, 0.5, 1.0]:
        for q in np.arange(0, 1.01, 0.1):
            m, s = F12.simulate(B, C, second, q, a, frac, n_sim=300, seed=int(q * 100) + int(frac * 10))
            rows.append(dict(frac=frac, q=round(q, 1), mean=m, sd=s))
    r = pd.DataFrame(rows); r["z"] = (obs - r["mean"]) / r.sd
    print(r.pivot(index="q", columns="frac", values="mean").round(4).to_string())
    print("   q consistent |z|<=1:", r[r.z.abs() <= 1].groupby("frac").q.agg(lambda s: f"{s.min():.1f}-{s.max():.1f}").to_dict())
    print("   q consistent |z|<=2:", r[r.z.abs() <= 2].groupby("frac").q.agg(lambda s: f"{s.min():.1f}-{s.max():.1f}").to_dict())
    print("   sd of the delta at q=0.5 (pure public-sampling noise):", r[r.q == 0.5].set_index("frac").sd.round(4).to_dict())

    print("\n== style-only predictability of the class")
    u = pd.read_csv(out_dir("26_uncertainty_map_B") / "uncertainty.csv")
    tl = u[(u.ctype == "line")].reset_index(drop=True)
    S = pd.DataFrame(pmap(style, [str(ROOT / "data/images/test" / i) for i in tl.image_id]))
    conf = tl.conf.values > .9
    yB = tl.B.map(L).values
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    pt = cross_val_predict(HistGradientBoostingClassifier(max_iter=200), S[conf], yB[conf], cv=cv)
    print(f"   TEST lines conf>.9 (n={conf.sum()}): style-only acc {np.mean(pt == yB[conf]):.3f} | macro-F1 {f1_score(yB[conf], pt, average='macro'):.3f}"
          f" | majority baseline {np.bincount(yB[conf]).max() / conf.sum():.3f}")
    tr = pd.read_csv(out_dir("08_v2_gap_decomposition") / "train_types_oof.csv"); tr = tr[tr.ctype == "line"].sample(1200, random_state=0)
    St = pd.DataFrame(pmap(style, [str(ROOT / "data/images/train" / i) for i in tr.image_id])); yt = tr.label.map(L).values
    grp = pd.read_csv(ROOT / "data/folds.csv").set_index("image_id").fold.reindex(tr.image_id).values
    ptr = np.zeros(len(yt), int)
    for k in range(5):
        m = grp == k
        ptr[m] = HistGradientBoostingClassifier(max_iter=200).fit(St[~m], yt[~m]).predict(St[m])
    print(f"   TRAIN lines (group CV): style-only acc {np.mean(ptr == yt):.3f} | macro-F1 {f1_score(yt, ptr, average='macro'):.3f}")
    # train-fit style model applied to test confident lines: does TRAIN style transfer to test?
    mdl = HistGradientBoostingClassifier(max_iter=200).fit(St, yt)
    print(f"   train-fit style model on TEST conf>.9 lines: acc vs B {np.mean(mdl.predict(S[conf]) == yB[conf]):.3f}")
    print("\n   median paper colour (R,G,B) per class, TEST conf>.9 lines (B label):")
    print(pd.concat([S[conf].reset_index(drop=True), pd.Series(np.array(LABELS)[yB[conf]], name="cls")], axis=1).groupby("cls")[["p_r", "p_g", "p_b", "i_r", "i_g", "i_b", "bar", "noise"]].median().round(1).to_string())
    print("   same, TRAIN lines:")
    print(pd.concat([St, pd.Series(np.array(LABELS)[yt], name="cls")], axis=1).groupby("cls")[["p_r", "p_g", "p_b", "i_r", "i_g", "i_b", "bar", "noise"]].median().round(1).to_string())
    ch = tl.image_id.isin(test.image_id[B != C])
    sp = HistGradientBoostingClassifier(max_iter=200).fit(S[conf], yB[conf]).predict(S[ch])
    d = tl[ch].assign(style_pred=np.array(LABELS)[sp], C=np.array(LABELS)[C[test.set_index("image_id").index.get_indexer(tl[ch].image_id)]])
    print("\n== the 10 changed rows: B | C | style-only model (fit on confident test lines)")
    print(d[["image_id", "B", "C", "style_pred", "conf"]].to_string(index=False))
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for c in LABELS:
        m = np.array(LABELS)[yB[conf]] == c
        ax[0].scatter(S[conf].p_r[m], S[conf].p_b[m], s=6, label=c)
        mt = np.array(LABELS)[yt] == c
        ax[1].scatter(St.p_r[mt], St.p_b[mt], s=6, label=c)
    ax[0].set_title("TEST conf>.9 lines: paper R vs B"); ax[1].set_title("TRAIN lines: paper R vs B"); ax[0].legend()
    plt.tight_layout(); plt.savefig(OUT / "paper_colour_by_class.png", dpi=70); plt.close()
    print("->", OUT)
