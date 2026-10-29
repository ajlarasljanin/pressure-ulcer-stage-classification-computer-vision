"""
phash grupisanje slika za group-aware split.
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import scipy.fftpack as fftpack

ROOT = Path(".")
DATA = ROOT / "Dataset - expanded" / "pressure_ulcers_expanded"
OUT_CSV = ROOT / "phash_groups.csv"

CLASSES = ["Invalid", "SDTI", "Stage_I", "Stage_II", "Stage_III", "Stage_IV", "Unstageable"]
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
COLORS = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#a65628", "#f781bf", "#666666"]

MIN_KB = 2.0
HASH_SIZE = 8
HIGHFREQ = 4
PHASH_THRESH = 10

random.seed(42)
np.random.seed(42)


def list_images(folder):
    if not folder.exists():
        return []
    files = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXT and p.stat().st_size / 1024 >= MIN_KB:
            files.append(p)
    return files


def phash_bits(im):
    n = HASH_SIZE * HIGHFREQ
    gray = im.convert("L").resize((n, n), Image.BICUBIC)
    pix = np.asarray(gray, dtype=np.float64)
    dct = fftpack.dct(fftpack.dct(pix, axis=0, norm="ortho"), axis=1, norm="ortho")
    low = dct[:HASH_SIZE, :HASH_SIZE]
    return (low > np.median(low)).ravel()


def phash_pair(path):
    with Image.open(path) as im:
        im = im.convert("RGB")
        return phash_bits(im), phash_bits(im.transpose(Image.FLIP_LEFT_RIGHT))


def hamming(a, b):
    return int(np.count_nonzero(a != b))


def distance(pair_a, pair_b):
    a_n, a_f = pair_a
    b_n, b_f = pair_b
    return min(
        hamming(a_n, b_n),
        hamming(a_n, b_f),
        hamming(a_f, b_n),
        hamming(a_f, b_f),
    )


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def group_class(paths, pairs, thresh):
    n = len(paths)
    dsu = DSU(n)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = distance(pairs[i], pairs[j])
            dists.append(d)
            if d <= thresh:
                dsu.union(i, j)
    roots = [dsu.find(i) for i in range(n)]
    remap = {r: k for k, r in enumerate(sorted(set(roots)))}
    return [remap[r] for r in roots], dists


rows = []
dist_by_class = {}
print("prag =", PHASH_THRESH)

for cls in CLASSES:
    paths = list_images(DATA / cls)
    if not paths:
        continue

    pairs = [phash_pair(p) for p in paths]
    gids, dists = group_class(paths, pairs, PHASH_THRESH)
    dist_by_class[cls] = dists

    for p, g in zip(paths, gids):
        rows.append({"class": cls, "path": str(p), "name": p.name, "group_id": f"{cls}_{g}"})

    vc = pd.Series(gids).value_counts()
    print(
        f"{cls:12s} n={len(paths):4d}  grupa={len(vc):4d}  "
        f"avg={len(paths)/len(vc):.2f}  max={int(vc.max()):3d}  "
        f"singl={int((vc == 1).sum()):4d}  singl%={100*(vc==1).mean():.1f}"
    )

df = pd.DataFrame(rows)
assert len(df) > 0, f"nema slika u {DATA}"
df.to_csv(OUT_CSV, index=False)
print("sacuvano:", OUT_CSV.name, len(df), "| ukupno grupa:", df["group_id"].nunique())

fig, axes = plt.subplots(2, 4, figsize=(14, 6))
axes = axes.ravel()
for i, cls in enumerate(CLASSES):
    ax = axes[i]
    d = dist_by_class.get(cls, [])
    if d:
        ax.hist(d, bins=range(0, 66, 2), color=COLORS[i], edgecolor="white")
    ax.axvline(PHASH_THRESH, color="k", ls="--", lw=1)
    ax.set_title(cls, fontsize=9)
    ax.set_xlabel("Hamming")
axes[-1].axis("off")
fig.suptitle("Parne distance unutar klase")
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 4.5))
data, labels = [], []
for cls in CLASSES:
    sub = df[df["class"] == cls]
    if len(sub):
        data.append(sub["group_id"].value_counts().values)
        labels.append(cls)
bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=True)
for patch, col in zip(bp["boxes"], COLORS):
    patch.set_facecolor(col)
    patch.set_alpha(0.8)
ax.set_ylabel("velicina grupe")
ax.set_title("Velicine grupa")
ax.tick_params(axis="x", rotation=25)
fig.tight_layout()
plt.show()

multi = df.groupby("group_id").filter(lambda g: len(g) > 1)
gids = list(multi["group_id"].unique())
random.shuffle(gids)
gids = gids[:6]
if gids:
    cols = min(5, int(multi.groupby("group_id").size().max()))
    fig, axes = plt.subplots(len(gids), cols, figsize=(2.2 * cols, 2.2 * len(gids)), squeeze=False)
    for r, gid in enumerate(gids):
        members = df.loc[df["group_id"] == gid, "path"].tolist()[:cols]
        for c in range(cols):
            axes[r, c].axis("off")
            if c < len(members):
                axes[r, c].imshow(Image.open(members[c]).convert("RGB"))
                if c == 0:
                    axes[r, c].set_title(gid, fontsize=7, loc="left")
    fig.suptitle("Primeri grupa (>1)")
    fig.tight_layout()
    plt.show()
else:
    print("nema grupa >1 — povecaj PHASH_THRESH")

print("gotovo")
