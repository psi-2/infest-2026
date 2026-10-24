"""36 - Before proposing a run-ensemble for non-page rows: does it just redo C's 10 changes (LB said -0.0055)?
Builds (locally, from saved test logits) E = pages from v5 (as B) + non-page mean of a run combo, for the OOF-best combos
of eda/35, and reports rows changed vs B, overlap with C's rows, and agreement with C on those rows."""
import numpy as np
import pandas as pd

from common import LABELS, ROOT, out_dir

M = __import__("importlib").import_module("35_run_ensemble_oof")

if __name__ == "__main__":
    test = pd.read_csv(ROOT / "data/test.csv")
    tv = pd.read_csv(ROOT / "analysis/outputs/v5_submission_check/test_versions.csv"); pg = (tv.ctype == "page").to_numpy()
    L = np.array(LABELS)
    C = pd.read_csv(ROOT / "results/v6/main/submission_C_targeted.csv").set_index("image_id").label.reindex(test.image_id).to_numpy(dtype=object)
    B = tv.B.to_numpy(dtype=object); twin = test.image_id.isin(pd.read_csv(ROOT / "data/test_train_twins.csv").image_id).to_numpy()
    R = {"v2": M.load(ROOT / "results/v2/main", "fold"), "v4s7": M.load(ROOT / "results/v4", "cv_s7_f"), "v4s2026": M.load(ROOT / "results/v4", "cv_s2026_f"),
         "v5": M.load(ROOT / "results/v5/main", "fold"), "v6": M.load(ROOT / "results/v6/main", "fold")}
    chC = C != B
    for combo in ["v2+v4s2026+v6", "v2+v4s7+v6", "v2+v4s7+v4s2026+v6", "v2+v4s7+v4s2026+v5+v6", "v2+v4s7+v4s2026"]:
        p = np.mean([R[c]["test"] for c in combo.split("+")], 0)
        E = np.where(pg, B, L[p.argmax(1)]).astype(object); E[twin] = B[twin]
        ch = E != B
        print(f"{combo:24s}: changes {ch.sum():2d} non-page rows | overlap with C's 10: {(ch & chC).sum()} | same label as C there: {((E == C) & ch & chC).sum()}"
              f" | new rows not in C: {(ch & ~chC).sum()} | transitions {pd.Series([f'{a}->{b}' for a, b in zip(B[ch], E[ch])]).value_counts().head(6).to_dict()}")
