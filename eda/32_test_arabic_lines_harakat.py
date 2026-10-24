"""32 - Follow-up of eda/31: B-'jawi' test lines/blocks sorted by diacritic share. In train, jawi lines are UNvowelled
(diac .07) and pegon lines vowelled (.47). If B-jawi test rows with many harakat exist, they are pegon candidates
(or jawi from a different, vowelled source). Grids of the top / bottom diacritic B-jawi rows and of B-pegon rows,
plus counts. Visual evidence only (no labels are created)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ROOT, open_rgb, out_dir

OUT = out_dir("32_test_arabic_lines_harakat")

if __name__ == "__main__":
    te = pd.read_csv(out_dir("31_jawi_pegon_anatomy") / "test_anatomy.csv")
    for cls in ["jawi", "pegon"]:
        x = te[(te.B == cls) & (te.ctype == "line")]
        print(f"== B={cls} lines n={len(x)}: diac quartiles {x.clean_diac.quantile([.25, .5, .75]).round(3).tolist()} | diac>.35: {(x.clean_diac > .35).sum()}"
              f" | n_cc median {x.clean_n_cc.median()}")
    for name, x in [("jawi_high_diac", te[(te.B == "jawi") & (te.ctype == "line")].sort_values("clean_diac", ascending=False).head(24)),
                    ("jawi_low_diac", te[(te.B == "jawi") & (te.ctype == "line")].sort_values("clean_diac").head(12)),
                    ("pegon_lines", te[(te.B == "pegon") & (te.ctype == "line")].sample(12, random_state=0))]:
        fig, axes = plt.subplots(len(x), 1, figsize=(20, 1.5 * len(x)))
        for a, r in zip(np.atleast_1d(axes), x.itertuples()):
            a.imshow(open_rgb(ROOT / "data/images/test" / r.image_id), aspect="auto")
            a.set_title(f"{r.image_id} B={r.B} conf={r.conf:.2f} diac={r.clean_diac:.2f} ncc={r.clean_n_cc}", fontsize=11); a.axis("off")
        plt.tight_layout(); plt.savefig(OUT / f"{name}.png", dpi=45); plt.close()
    print("->", OUT)
