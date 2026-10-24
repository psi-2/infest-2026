"""corrupt() for v8: v2's corrupt (prepro/outputs/pipeline_v2_snapshot.py) with the test-only patterns of prepro/12 as knobs.
prepro/12 (deep domain classifier, corrupt(train) vs test, AUC .832): the most 'test-only' canvases show
  (1) several long, thin, semi-transparent grey wedges across the line (v2: p=.08, 1-3 random triangles)
  (2) vertical crops that cut glyph tops/bottoms (v2 keeps 85-100% of the height)
  (3) noise on BLURRED text; v2 adds sharp noise after blur (most 'train-only' canvases = crisp text + speckle)
Defaults = v2 behaviour; prepro/13 picks the values by domain AUC and visual checks.
"""
import io
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import pipeline_v2_snapshot as V2

BASE = dict(p_occ=0.08, sliver=0.0, n_occ=(1, 3), p_vcrop=0.0, vcrop=(0.5, 0.85), noise_before_blur=False)


def _wedge(w, h, rng):
    """long thin triangle across the image: two far-apart points on one side, apex on the other side of the line."""
    x0, x1 = sorted(rng.uniform(-.2, 1.2) * w for _ in range(2))
    if x1 - x0 < .4 * w:
        x1 = x0 + .4 * w + rng.uniform(0, .6) * w
    y_base, y_apex = (rng.uniform(-.1, .4) * h, rng.uniform(.6, 1.1) * h) if rng.random() < .5 else (rng.uniform(.6, 1.1) * h, rng.uniform(-.1, .4) * h)
    return [(x0, y_base), (x1, y_base + rng.uniform(-.2, .2) * h), (rng.uniform(x0, x1), y_apex)]


def corrupt(img, rng=random, cfg=None):
    c = dict(BASE, **(cfg or {}))
    w, h = img.size
    if rng.random() < 0.5:
        for _ in range(4):
            angle = rng.uniform(-15, 15)
            if V2.ink_retained(img, angle) >= 0.6:
                img = V2.rotate(img, angle, rng.choice(["edge"] * 5 + ["white"] * 2 + ["black"])); break
    if rng.random() < 0.25 and w > 60:
        cw, ch = int(w * rng.uniform(0.6, 1.0)), int(h * rng.uniform(0.85, 1.0))
        x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
        img = img.crop((x, y, x + cw, y + ch)); w, h = img.size
    if c["p_vcrop"] and rng.random() < c["p_vcrop"] and h >= 24 and w / h >= 2:  # (2) cut glyph tops / bottoms
        ch = max(8, int(h * rng.uniform(*c["vcrop"])))
        y = rng.choice([0, h - ch, rng.randint(0, h - ch)])
        img = img.crop((0, y, w, y + ch)); w, h = img.size
    if rng.random() < 0.45:
        g = np.asarray(img.convert("L"), dtype=np.float32)[..., None] / 255
        bg = np.array(rng.choice(V2.PARCHMENT), dtype=np.float32) * rng.uniform(0.85, 1.05)
        ink = np.array(rng.choice(V2.INKS), dtype=np.float32) + rng.uniform(0, 60)
        img = Image.fromarray(np.clip(ink * (1 - g) + bg * g, 0, 255).astype(np.uint8))
    if c["noise_before_blur"] and rng.random() < 0.15:  # (3) noise then blur (only this variant draws here: default keeps v2's rng stream)
        img = _noise(img, rng)
    if rng.random() < 0.45:
        if rng.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2) * max(h, 32) / 64))
        else:
            f = rng.uniform(0.35, 0.8)
            small = img.resize((max(4, int(w * f)), max(4, int(h * f))), Image.Resampling.BILINEAR)
            img = small.resize((w, h), rng.choice([Image.Resampling.NEAREST, Image.Resampling.BILINEAR, Image.Resampling.BICUBIC]))
    if rng.random() < c["p_occ"]:  # (1) semi-transparent grey polygons, optionally long wedges
        over = Image.new("RGBA", img.size, (0, 0, 0, 0)); d = ImageDraw.Draw(over)
        for _ in range(rng.randint(*c["n_occ"])):
            v = rng.randint(60, 150)
            pts = _wedge(w, h, rng) if c["sliver"] and rng.random() < c["sliver"] else [(rng.uniform(-.1, 1.1) * w, rng.uniform(-.1, 1.1) * h) for _ in range(3)]
            d.polygon(pts, fill=(v, v, v, rng.randint(120, 230)))
        img = Image.alpha_composite(img.convert("RGBA"), over).convert("RGB")
    if rng.random() < 0.15:
        d = ImageDraw.Draw(img)
        for _ in range(rng.randint(1, 4)):
            rw, rh = rng.uniform(.03, .1) * w, rng.uniform(.03, .1) * h + 1
            x, y = rng.uniform(0, w - rw), rng.uniform(0, h - rh)
            d.rectangle([x, y, x + rw, y + rh], fill=(0, 0, 0))
    if rng.random() < 0.07:
        a = np.asarray(img).copy(); k = int(h * rng.uniform(0.05, 0.25)) + 1
        if rng.random() < 0.5:
            a[:-k] = a[k:]; a[-k:] = 0
        else:
            a[k:] = a[:-k]; a[:k] = 0
        img = Image.fromarray(a)
    if not c["noise_before_blur"] and rng.random() < 0.15:
        img = _noise(img, rng)
    if rng.random() < 0.3:
        buf = io.BytesIO(); img.save(buf, "JPEG", quality=rng.randint(20, 75)); buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img


def _noise(img, rng):
    a = np.asarray(img, dtype=np.float32) + np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(5, 25), (img.height, img.width, 1))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


if __name__ == "__main__":
    im = Image.new("RGB", (600, 60), "white"); ImageDraw.Draw(im).text((10, 20), "abc def ghi jkl mno", fill="black")
    # defaults must reproduce v2 exactly (same rng stream)
    for s in range(30):
        a = np.asarray(corrupt(im, random.Random(s))); b = np.asarray(V2.corrupt(im, random.Random(s)))
        assert a.shape == b.shape and (a == b).all(), s
    for s in range(30):
        assert corrupt(im, random.Random(s), dict(p_occ=1, sliver=1, p_vcrop=1, noise_before_blur=True)).mode == "RGB"
    print("corrupt_v8 self-check OK (defaults == v2)")
