"""
Analiza radnog skupa pressure_ulcers_expanded.
"""

from pathlib import Path
import re
import random

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(".")
DATA = ROOT / "Dataset - expanded" / "pressure_ulcers_expanded"

CLASSES = ["Invalid", "SDTI", "Stage_I", "Stage_II", "Stage_III", "Stage_IV", "Unstageable"]
NPUAP = ["SDTI", "Stage_I", "Stage_II", "Stage_III", "Stage_IV", "Unstageable"]
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
OLD_NAME_RE = re.compile(
    r"^(Invalid|SDTI|Stage_I|Stage_II|Stage_III|Stage_IV|Unstageable)_\d+"
)
COLORS = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#a65628", "#f781bf", "#666666"]

random.seed(42)


def list_images(folder):
    if not folder.exists():
        return []
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT]


def load_table(base):
    rows = []
    for cls in CLASSES:
        for p in list_images(base / cls):
            try:
                with Image.open(p) as im:
                    w, h = im.size
                    mode = im.mode
                ok = True
                err = ""
            except Exception as e:
                w = h = None
                mode = None
                ok = False
                err = str(e)

            rows.append({
                "class": cls,
                "name": p.name,
                "ext": p.suffix.lower(),
                "size_kb": round(p.stat().st_size / 1024, 2),
                "width": w,
                "height": h,
                "aspect": round(w / h, 3) if ok and h else None,
                "mpix": round((w * h) / 1e6, 3) if ok else None,
                "mode": mode,
                "is_aug": p.name.lower().startswith("aug_"),
                "is_224": bool(ok and w == 224 and h == 224),
                "is_old_style": bool(OLD_NAME_RE.match(p.name)),
                "readable": ok,
                "error": err,
            })
    return pd.DataFrame(rows)


df = load_table(DATA)
assert len(df) > 0, f"Nema slika u {DATA.resolve()}"

print("Dataset:", DATA.resolve())
print("Ukupno:", len(df))
print("Ekstenzije:", df["ext"].value_counts().to_dict())
print("Color mode:", df["mode"].value_counts(dropna=False).to_dict())
print("Necitljive:", int((~df["readable"]).sum()))

summary = (
    df.groupby("class")
    .agg(
        n=("name", "count"),
        aug_n=("is_aug", "sum"),
        pct_aug=("is_aug", "mean"),
        n_224=("is_224", "sum"),
        pct_224=("is_224", "mean"),
        old_style=("is_old_style", "sum"),
        avg_kb=("size_kb", "mean"),
        min_kb=("size_kb", "min"),
        max_kb=("size_kb", "max"),
        med_w=("width", "median"),
        med_h=("height", "median"),
        med_mpix=("mpix", "median"),
    )
    .reindex(CLASSES)
    .reset_index()
)
summary["pct_aug"] = (summary["pct_aug"] * 100).round(1)
summary["pct_224"] = (summary["pct_224"] * 100).round(1)
summary["avg_kb"] = summary["avg_kb"].round(1)
summary["share_%"] = (100 * summary["n"] / summary["n"].sum()).round(1)

print("\nPo klasama:")
print(summary.to_string(index=False))

n_min, n_max = int(summary["n"].min()), int(summary["n"].max())
print(f"\nDisbalans max/min = {n_max}/{n_min} = {n_max / n_min:.2f}")
print("Bez Invalid:", int(df["class"].isin(NPUAP).sum()))
print(f"aug_*: {int(df['is_aug'].sum())} ({100 * df['is_aug'].mean():.1f}%)")
print(f"224x224: {int(df['is_224'].sum())} ({100 * df['is_224'].mean():.1f}%)")

tiny = df[df["size_kb"] < 2].sort_values("size_kb")
print("\nFajlovi < 2 KB:", len(tiny))
if len(tiny):
    print(tiny[["class", "name", "size_kb"]].to_string(index=False))

res = (
    df.dropna(subset=["width", "height"])
    .assign(res=lambda x: x["width"].astype(int).astype(str) + "x" + x["height"].astype(int).astype(str))
    ["res"].value_counts().head(15)
)
print("\nNajcesce rezolucije:")
print(res.to_string())

fig, ax = plt.subplots(figsize=(10, 4.5))
bars = ax.bar(CLASSES, summary["n"], color=COLORS, edgecolor="white")
ax.set_title("Expanded — broj slika po klasama")
ax.set_ylabel("Broj slika")
ax.tick_params(axis="x", rotation=25)
for b, v in zip(bars, summary["n"]):
    ax.text(b.get_x() + b.get_width() / 2, v + 6, str(int(v)), ha="center", fontsize=8)
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 4.5))
aug = summary["aug_n"].values
rest = summary["n"].values - aug
ax.bar(CLASSES, rest, label="bez aug_*", color="#7570b3", edgecolor="white")
ax.bar(CLASSES, aug, bottom=rest, label="aug_*", color="#1b9e77", edgecolor="white")
ax.set_title("Udeo aug_* po klasama")
ax.set_ylabel("Broj slika")
ax.tick_params(axis="x", rotation=25)
ax.legend()
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(CLASSES, summary["pct_224"], color="#e7298a", edgecolor="white")
ax.set_ylim(0, 100)
ax.set_ylabel("%")
ax.set_title("Udeo slika 224x224")
ax.tick_params(axis="x", rotation=25)
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 4.5))
data = [df.loc[df["class"] == c, "size_kb"].values for c in CLASSES]
bp = ax.boxplot(data, tick_labels=CLASSES, patch_artist=True, showfliers=False)
for patch, col in zip(bp["boxes"], COLORS):
    patch.set_facecolor(col)
    patch.set_alpha(0.8)
ax.set_yscale("log")
ax.set_title("Velicina fajlova po klasama")
ax.set_ylabel("KB (log)")
ax.tick_params(axis="x", rotation=25)
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.barh(res.index[::-1], res.values[::-1], color="#2166ac", edgecolor="white")
ax.set_title("Najcesce rezolucije — expanded")
ax.set_xlabel("Broj slika")
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(6.2, 4.5))
n_inv = int((df["class"] == "Invalid").sum())
n_val = len(df) - n_inv
ax.pie(
    [n_val, n_inv],
    labels=["Valid (6 klasa)", "Invalid"],
    colors=["#1b9e77", "#d95f02"],
    autopct="%1.1f%%",
    startangle=90,
    explode=(0.02, 0.05),
    wedgeprops={"edgecolor": "white"},
)
ax.set_title("Valid vs Invalid")
fig.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.ravel()
for i, cls in enumerate(CLASSES):
    ax = axes[i]
    files = list_images(DATA / cls)
    ax.set_title(f"{cls} (n={len(files)})", fontsize=9, color=COLORS[i])
    if files:
        ax.imshow(Image.open(random.choice(files)).convert("RGB"))
    ax.axis("off")
axes[-1].axis("off")
fig.suptitle("Primeri snimaka — expanded")
fig.tight_layout()
plt.show()