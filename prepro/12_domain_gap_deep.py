"""12 - Where does corrupt(train) still NOT look like test? Deep-feature domain classifier on the exact model input.
Canvas = v6 preprocess (prepro/09 pre_v6 + crop_to_ink) -> to_canvas 160x640 (one random tile), non-page images only.
Sets: test (all non-page), train clean, train corrupt (corrupt() -> preprocess, like training).
DINOv3-S [CLS, mean patch] -> logistic regression, 5-fold: AUC(train corrupt vs test), AUC(train clean vs test), per type.
Then LOOK: the 32 test canvases that are the most confidently 'test' (= the corruption / source we do not simulate) and the
16 corrupted-train canvases that are most 'train' (= augmentation that test never shows).
"""
import importlib.util
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eda"))
from common import ROOT, out_dir, pmap  # noqa: E402

spec = importlib.util.spec_from_file_location("m09", Path(__file__).with_name("09_stretch_fix.py")); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
V2 = M.V2
OUT = Path(__file__).resolve().parent / "outputs" / "12_domain_gap_deep"; OUT.mkdir(parents=True, exist_ok=True)


def pp6(img):
    return Image.fromarray(V2.crop_to_ink(M.pre_v6(np.asarray(img.convert("L")))))


def canvas(args):
    path, seed, mode = args
    rng = random.Random(seed)
    img = V2.load_rgb(path)
    if max(img.size) > 1600:
        img.thumbnail((1600, 1600))
    if mode == "corrupt":
        img = V2.corrupt(img, rng)
    tiles = V2.to_canvas(pp6(img), train=False)
    return np.asarray(tiles[rng.randrange(len(tiles))], dtype=np.uint8)


def embed(arr):
    m = timm.create_model("vit_small_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0, dynamic_img_size=True).eval(); torch.set_num_threads(20)
    mean, std = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1), torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
    out = []
    with torch.inference_mode():
        for s in range(0, len(arr), 32):
            x = (torch.from_numpy(arr[s:s + 32]).float().div(255).unsqueeze(1).repeat(1, 3, 1, 1) - mean) / std
            t = m.forward_features(x); out.append(torch.cat([t[:, 0], t[:, m.num_prefix_tokens:].mean(1)], 1))
    X = torch.cat(out).numpy(); return X / np.linalg.norm(X, axis=1, keepdims=True)


if __name__ == "__main__":
    u = pd.read_csv(out_dir("26_uncertainty_map_B") / "uncertainty.csv")
    te = u[u.ctype != "page"].reset_index(drop=True)
    tr = pd.read_csv(out_dir("08_v2_gap_decomposition") / "train_types_oof.csv"); tr = tr[tr.ctype != "page"].sample(1500, random_state=0).reset_index(drop=True)
    f = OUT / "emb.npz"
    if f.exists():
        z = np.load(f); Xt, Xc, Xk, At, Ak = z["t"], z["c"], z["k"], z["at"], z["ak"]
    else:
        At = np.stack(pmap(canvas, [(str(ROOT / "data/images/test" / i), 0, "clean") for i in te.image_id]))
        Ac = np.stack(pmap(canvas, [(str(ROOT / "data/images/train" / i), k, "clean") for k, i in enumerate(tr.image_id)]))
        Ak = np.stack(pmap(canvas, [(str(ROOT / "data/images/train" / i), k, "corrupt") for k, i in enumerate(tr.image_id)]))
        Xt, Xc, Xk = embed(At), embed(Ac), embed(Ak)
        np.savez(f, t=Xt, c=Xc, k=Xk, at=At, ak=Ak)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    res = {}
    for name, X in [("train_clean", Xc), ("train_corrupt", Xk)]:
        Z = np.vstack([X, Xt]); yy = np.r_[np.zeros(len(X)), np.ones(len(Xt))]
        p = cross_val_predict(LogisticRegression(C=1, max_iter=3000), Z, yy, cv=cv, method="predict_proba")[:, 1]
        res[name] = p
        print(f"== domain AUC {name} vs test: {roc_auc_score(yy, p):.3f}")
        for t in ["line", "block", "glyph"]:
            mt = np.r_[(tr.ctype == t).to_numpy(), (te.ctype == t).to_numpy()]
            print(f"     {t:5s}: {roc_auc_score(yy[mt], p[mt]):.3f}")
    p = res["train_corrupt"]; pt, pk = p[len(Xk):], p[:len(Xk)]
    te["p_test"] = pt
    print(f"\n== test rows by 'test-ness' (p_test from corrupt-vs-test): quartiles {np.quantile(pt, [.25, .5, .75]).round(3).tolist()}")
    te["q"] = pd.qcut(te.p_test, 4, labels=["Q1 train-like", "Q2", "Q3", "Q4 most test-only"])
    print("   B confidence and low-conf share by quartile:")
    print(te.groupby("q", observed=True).agg(n=("conf", "size"), conf_med=("conf", "median"), lowconf=("conf", lambda s: (s < .6).mean()),
                                             line=("ctype", lambda s: (s == "line").mean())).round(3).to_string())
    print("   B class mix in Q4 vs all (%):")
    print(pd.DataFrame({"Q4": te[te.q == "Q4 most test-only"].B.value_counts(normalize=True), "all": te.B.value_counts(normalize=True)}).mul(100).round(1).T.to_string())
    top = np.argsort(-pt)[:32]
    fig, axes = plt.subplots(16, 2, figsize=(28, 30))
    for a, k in zip(axes.T.ravel(), top):
        a.imshow(At[k], cmap="gray", vmin=0, vmax=255, aspect="auto"); a.set_title(f"TEST {te.image_id[k]} p_test={pt[k]:.3f} B={te.B[k]}({te.conf[k]:.2f})", fontsize=11); a.axis("off")
    plt.tight_layout(); plt.savefig(OUT / "test_most_testlike_canvas.png", dpi=35); plt.close()
    fig, axes = plt.subplots(16, 2, figsize=(28, 30))
    for a, k in zip(axes.T.ravel(), top):
        a.imshow(np.asarray(M.V2.load_rgb(ROOT / "data/images/test" / te.image_id[k])), aspect="auto"); a.set_title(f"RAW {te.image_id[k]}", fontsize=11); a.axis("off")
    plt.tight_layout(); plt.savefig(OUT / "test_most_testlike_raw.png", dpi=35); plt.close()
    bot = np.argsort(pk)[:16]
    fig, axes = plt.subplots(8, 2, figsize=(28, 15))
    for a, k in zip(axes.T.ravel(), bot):
        a.imshow(Ak[k], cmap="gray", vmin=0, vmax=255, aspect="auto"); a.set_title(f"TRAIN-CORRUPT {tr.image_id[k]} p_test={pk[k]:.3f} {tr.label[k]}", fontsize=11); a.axis("off")
    plt.tight_layout(); plt.savefig(OUT / "train_corrupt_least_testlike.png", dpi=35); plt.close()
    te.to_csv(OUT / "test_ptest.csv", index=False)
    print("->", OUT)
