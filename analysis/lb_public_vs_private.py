"""Public vs private LB untuk 12 submission tim (angka dari halaman submission Kaggle setelah lomba selesai).
Pertanyaan: seberapa jauh public LB menyesatkan keputusan final? Mencetak tabel urutan + korelasi peringkat,
menulis submissions.csv, dan menggambar scatter public-vs-private sebagai SVG.
Hanya memakai pustaka standar, jadi bisa dijalankan tanpa menyiapkan environment.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/outputs/lb_public_vs_private"
OUT.mkdir(parents=True, exist_ok=True)

#        versi,            public,  private, catatan
ROWS = [("Laksamada awal", 0.82721, 0.83675, "percobaan awal anggota lain"),
        ("v1",             0.86289, 0.87594, "baseline letterbox, holdout acak"),
        ("v7 kNN",         0.88808, 0.91306, "ArcFace + SupCon, dibaca kNN"),
        ("v3",             0.89077, 0.92827, "tile page"),
        ("v11 naflex",     0.90707, 0.95147, "SigLIP2 NaFlex"),
        ("ViT-7B",         0.90966, 0.93773, "LoRA, holdout fold 3"),
        ("B + v11 page",   0.91240, 0.94015, "page dari v11 1024"),
        ("v6 C",           0.91273, 0.93592, "perbaikan stretch, 10 baris"),
        ("ens8 + TTA",     0.91285, 0.94785, "ensemble 8 model + TTA"),
        ("v2",             0.91514, 0.91946, "corrupt sebelum preprocess, group CV"),
        ("v5 + v11",       0.91815, 0.95305, "rata-rata v5 dan v11 - PRIVATE TERBAIK"),
        ("hybrid B",       0.91820, 0.93349, "page v5 + non-page v2 - DIPILIH")]


def rank(values):  # peringkat 1 = tertinggi, rata-rata untuk nilai seri
    order = sorted(range(len(values)), key=lambda i: -values[i])
    out = [0.0] * len(values)
    for pos, i in enumerate(order, 1):
        out[i] = pos
    return out


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** .5
    vb = sum((y - mb) ** 2 for y in rb) ** .5
    return cov / (va * vb)


def kendall(a, b):
    n, con, dis = len(a), 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (a[i] - a[j]) * (b[i] - b[j])
            con += s > 0; dis += s < 0
    return (con - dis) / (n * (n - 1) / 2)


def svg(rows, path):
    W, H, M = 940, 700, 90
    pub = [r[1] for r in rows]; pri = [r[2] for r in rows]
    x0, x1 = min(pub) - .005, max(pub) + .005
    y0, y1 = min(pri) - .005, max(pri) + .008
    sx = lambda v: M + (v - x0) / (x1 - x0) * (W - M - 230)
    sy = lambda v: H - M - (v - y0) / (y1 - y0) * (H - 2 * M)
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI,Arial,sans-serif">',
         f'<rect width="{W}" height="{H}" fill="white"/>']
    for k in range(6):  # grid + sumbu
        gx = x0 + (x1 - x0) * k / 5; gy = y0 + (y1 - y0) * k / 5
        p.append(f'<line x1="{sx(gx):.1f}" y1="{M - 20}" x2="{sx(gx):.1f}" y2="{H - M}" stroke="#e8e8e8"/>')
        p.append(f'<line x1="{M}" y1="{sy(gy):.1f}" x2="{W - 230}" y2="{sy(gy):.1f}" stroke="#e8e8e8"/>')
        p.append(f'<text x="{sx(gx):.1f}" y="{H - M + 24}" font-size="13" fill="#555" text-anchor="middle">{gx:.3f}</text>')
        p.append(f'<text x="{M - 12}" y="{sy(gy) + 5:.1f}" font-size="13" fill="#555" text-anchor="end">{gy:.3f}</text>')
    best_pub = max(rows, key=lambda r: r[1]); best_pri = max(rows, key=lambda r: r[2])
    for r in rows:
        cx, cy = sx(r[1]), sy(r[2])
        colour = "#EE6666" if r is best_pri else ("#F0A202" if r is best_pub else "#5470C6")
        if r in (best_pub, best_pri):
            p.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="14" fill="none" stroke="{colour}" stroke-width="2.5"/>')
        p.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6.5" fill="{colour}"/>')
        p.append(f'<text x="{cx + 12:.1f}" y="{cy + 4:.1f}" font-size="14" fill="#222">{r[0]}</text>')
    p.append(f'<text x="{(W - 230) / 2 + M / 2:.0f}" y="{H - 28}" font-size="15" text-anchor="middle">Public LB</text>')
    p.append(f'<text x="26" y="{H / 2:.0f}" font-size="15" text-anchor="middle" transform="rotate(-90 26 {H / 2:.0f})">Private LB</text>')
    p.append(f'<text x="{M}" y="40" font-size="20" font-weight="600">Public LB menyesatkan: Spearman {spearman(pub, pri):.2f} dari 12 submission</text>')
    p.append(f'<text x="{M}" y="64" font-size="14" fill="#666">Lingkaran oranye: public tertinggi (dipilih). Lingkaran merah: private tertinggi.</text>')
    p.append('</svg>')
    path.write_text("\n".join(p))


if __name__ == "__main__":
    rows = sorted(ROWS, key=lambda r: -r[2])
    rp, rr = rank([r[1] for r in ROWS]), rank([r[2] for r in ROWS])
    pos = {r[0]: (int(rp[i]), int(rr[i])) for i, r in enumerate(ROWS)}
    print(f"{'versi':15s} {'public':>8s} {'private':>8s} {'selisih':>8s}  {'rank pub':>8s} {'rank priv':>9s}  catatan")
    for v, pub, pri, note in rows:
        print(f"{v:15s} {pub:8.5f} {pri:8.5f} {pri - pub:+8.5f}  {pos[v][0]:8d} {pos[v][1]:9d}  {note}")
    pub = [r[1] for r in ROWS]; pri = [r[2] for r in ROWS]
    print(f"\nkorelasi peringkat public vs private: Spearman {spearman(pub, pri):.3f} | Kendall {kendall(pub, pri):.3f}")
    gap = [b - a for a, b in zip(pub, pri)]
    print(f"private rata-rata {sum(gap) / len(gap):+.4f} di atas public (rentang {min(gap):+.4f} s/d {max(gap):+.4f})")
    bp = max(ROWS, key=lambda r: r[1]); br = max(ROWS, key=lambda r: r[2])
    print(f"public terbaik  : {bp[0]} (public {bp[1]:.5f}, private {bp[2]:.5f})")
    print(f"private terbaik : {br[0]} (public {br[1]:.5f}, private {br[2]:.5f})")
    print(f"biaya memilih lewat public: {bp[2] - br[2]:+.5f} private, padahal selisih public hanya {bp[1] - br[1]:+.5f}")
    with open(OUT / "submissions.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["versi", "public", "private", "selisih", "rank_public", "rank_private", "catatan"])
        for v, a, b, note in rows:
            w.writerow([v, a, b, round(b - a, 5), pos[v][0], pos[v][1], note])
    svg(ROWS, OUT / "public_vs_private.svg")
    print("->", OUT)
