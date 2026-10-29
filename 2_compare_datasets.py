"""
Poredjenje polaznog Dataseta i pressure_ulcers_expanded.
"""

from pathlib import Path
import hashlib
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(".")
OLD = ROOT / "Dataset"
NEW = ROOT / "Dataset - expanded" / "pressure_ulcers_expanded"

CLASSES = ["Invalid", "SDTI", "Stage_I", "Stage_II", "Stage_III", "Stage_IV", "Unstageable"]
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
COLORS = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#a65628", "#f781bf", "#666666"]
C_OLD = "#d95f02"
C_NEW = "#1b9e77"

random.seed(42)


def list_images(folder):
    if not folder.exists():
        return []
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT]


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_table(base, tag):
    rows = []
    for cls in CLASSES:
        for p in list_images(base / cls):
            try:
                with Image.open(p) as im:
                    w, h = im.size
                ok = True
            except Exception:
                w = h = None
                ok = False
            rows.append({
                "set": tag,
                "class": cls,
                "name": p.name,
                "size_kb": p.stat().st_size / 1024,
                "width": w,
                "height": h,
                "is_aug": p.name.lower().startswith("aug_"),
                "is_224": bool(ok and w == 224 and h == 224),
                "readable": ok,
            })
    return pd.DataFrame(rows)


df_old = load_table(OLD, "stari")
df_new = load_table(NEW, "expanded")
assert len(df_old) and len(df_new)

old_n = df_old["class"].value_counts().reindex(CLASSES).fillna(0).astype(int)
new_n = df_new["class"].value_counts().reindex(CLASSES).fillna(0).astype(int)

cmp = pd.DataFrame({
    "class": CLASSES,
    "stari": old_n.values,
    "expanded": new_n.values,
})
cmp["delta"] = cmp["expanded"] - cmp["stari"]
cmp["faktor"] = (cmp["expanded"] / cmp["stari"].replace(0, np.nan)).round(2)
cmp["udeo_stari_%"] = (100 * cmp["stari"] / cmp["stari"].sum()).round(1)
cmp["udeo_exp_%"] = (100 * cmp["expanded"] / cmp["expanded"].sum()).round(1)

print("Obim")
print(f"Stari:    {len(df_old)}")
print(f"Expanded: {len(df_new)}")
print(f"Faktor:   x{len(df_new) / len(df_old):.2f}")
print("\nPo klasama:")
print(cmp.to_string(index=False))

imb_old = old_n.max() / max(old_n.min(), 1)
imb_new = new_n.max() / max(new_n.min(), 1)
print(f"\nDisbalans max/min  stari={imb_old:.2f}  expanded={imb_new:.2f}")

overlap_rows = []
for cls in CLASSES:
    old_names = set(df_old.loc[df_old["class"] == cls, "name"])
    new_names = set(df_new.loc[df_new["class"] == cls, "name"])
    inter = old_names & new_names
    identical = 0
    for name in inter:
        p1 = OLD / cls / name
        p2 = NEW / cls / name
        if p1.exists() and p2.exists() and md5(p1) == md5(p2):
            identical += 1
    overlap_rows.append({
        "class": cls,
        "old_n": len(old_names),
        "exp_n": len(new_names),
        "same_name": len(inter),
        "byte_identical": identical,
        "only_old": len(old_names - new_names),
    })

ov = pd.DataFrame(overlap_rows)
print("\nPreklapanje fajlova:")
print(ov.to_string(index=False))
print(f"Bajt-identicno: {int(ov['byte_identical'].sum())}/{int(ov['old_n'].sum())}")


def prep_stats(df):
    return pd.Series({
        "pct_aug": 100 * df["is_aug"].mean(),
        "pct_224": 100 * df["is_224"].mean(),
        "med_kb": df["size_kb"].median(),
        "med_w": df["width"].median(),
        "med_h": df["height"].median(),
    })


print("\nGlobalne razlike:")
print("stari:\n", prep_stats(df_old).round(2).to_string())
print("expanded:\n", prep_stats(df_new).round(2).to_string())

x = np.arange(len(CLASSES))
w = 0.38

fig, ax = plt.subplots(figsize=(11, 5))
b1 = ax.bar(x - w / 2, cmp["stari"], w, label="Polazni", color=C_OLD, edgecolor="white")
b2 = ax.bar(x + w / 2, cmp["expanded"], w, label="Expanded", color=C_NEW, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(CLASSES, rotation=25)
ax.set_ylabel("Broj slika")
ax.set_title("Broj primera po klasama")
ax.legend()
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 4,
            int(b.get_height()), ha="center", fontsize=7)
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 4.8))
ax.barh(CLASSES[::-1], cmp["faktor"].values[::-1], color=C_NEW, edgecolor="white")
ax.axvline(1.0, color="#b2182b", ls="--", label="polazni = 1x")
ax.set_xlabel("Faktor (expanded / polazni)")
ax.set_title("Uvecanje broja slika po klasama")
ax.legend()
for i, v in enumerate(cmp["faktor"].values[::-1]):
    ax.text(v + 0.1, i, f"x{v:.1f}", va="center", fontsize=9)
fig.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
axes[0].bar(["Polazni", "Expanded"], [len(df_old), len(df_new)],
            color=[C_OLD, C_NEW], edgecolor="white", width=0.55)
axes[0].set_title("Ukupan obim")
axes[0].set_ylabel("Broj slika")
for i, v in enumerate([len(df_old), len(df_new)]):
    axes[0].text(i, v + 40, str(v), ha="center", fontweight="bold")

axes[1].bar(["Polazni", "Expanded"], [imb_old, imb_new],
            color=["#b2182b", "#1a9850"], edgecolor="white", width=0.55)
axes[1].set_title("Disbalans max/min")
axes[1].set_ylabel("Odnos")
for i, v in enumerate([imb_old, imb_new]):
    axes[1].text(i, v + 0.03, f"{v:.2f}", ha="center", fontweight="bold")
fig.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].pie(cmp["stari"], labels=CLASSES, colors=COLORS, startangle=90,
            wedgeprops={"edgecolor": "white"},
            autopct=lambda p: f"{p:.0f}%" if p >= 8 else "")
axes[0].set_title("Struktura — polazni")
axes[1].pie(cmp["expanded"], labels=CLASSES, colors=COLORS, startangle=90,
            wedgeprops={"edgecolor": "white"},
            autopct=lambda p: f"{p:.0f}%" if p >= 8 else "")
axes[1].set_title("Struktura — expanded")
fig.tight_layout()
plt.show()

fig, axes = plt.subplots(2, len(CLASSES), figsize=(14, 4.8))
for j, cls in enumerate(CLASSES):
    for i, (base, tag) in enumerate([(OLD, "polazni"), (NEW, "expanded")]):
        ax = axes[i, j]
        files = list_images(base / cls)
        if i == 0:
            ax.set_title(cls, fontsize=8, color=COLORS[j])
        if files:
            ax.imshow(Image.open(random.choice(files)).convert("RGB"))
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(tag, fontsize=9)
fig.suptitle("Uzorci po klasama")
fig.tight_layout()
plt.show()