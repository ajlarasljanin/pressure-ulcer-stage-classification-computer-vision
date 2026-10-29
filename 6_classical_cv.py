"""
Klasicni CV: segmentacija rane, konture, deskriptori, RF/SVM baseline.
Isti expanded skup i group-aware split kao Model B (5_train_pipeline.py).
"""

from pathlib import Path
from collections import Counter, defaultdict
import json
import random

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
# %%
    accuracy_score,

    f1_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(".")
DATA = ROOT / "Dataset - expanded" / "pressure_ulcers_expanded"
GROUPS_CSV = ROOT / "phash_groups.csv"
METRICS_JSON = ROOT / "cnn_test_metrics.json"

NPUAP = ["SDTI", "Stage_I", "Stage_II", "Stage_III", "Stage_IV", "Unstageable"]
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

SEED = 42
MIN_KB = 2.0
MAX_SIDE = 800
OVERLAYS_PER_CLASS = 2

random.seed(SEED)
np.random.seed(SEED)


def load_cnn_metrics(path=METRICS_JSON):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Nema {path} — prvo pokreni 5_train_pipeline.py "
            "(upisuje test metrike Model A/B)."
        )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    b = data["model_b"]
    return float(b["accuracy"]), float(b["macro_f1"])


def collect_paths(data_dir, class_dirs, min_kb=MIN_KB):
    paths, labels, skipped = [], [], []
    for cls in class_dirs:
        folder = data_dir / cls
        if not folder.exists():
            raise FileNotFoundError(folder)
        for p in sorted(folder.iterdir()):
            if not p.is_file() or p.suffix.lower() not in IMG_EXT:
                continue
            size_kb = p.stat().st_size / 1024
            if size_kb < min_kb:
                skipped.append((cls, p.name, round(size_kb, 2)))
                continue
            paths.append(p)
            labels.append(cls)
    return paths, labels, skipped


def path_key(p):
    return Path(p).as_posix().lower()


def load_group_map(csv_path=GROUPS_CSV):
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Nema {csv_path} — prvo pokreni 4_phash_groups.py")
    df = pd.read_csv(csv_path)
    return {path_key(p): g for p, g in zip(df["path"], df["group_id"])}


def group_split(paths, labels, group_map, seed=SEED, tag=""):
    """~70/15/15 po phash grupama, stratifikovano po labeli (kao u 5_)."""
    rng = random.Random(seed)

    groups = defaultdict(list)
    for p, y in zip(paths, labels):
        gid = group_map.get(path_key(p), f"solo::{path_key(p)}")
        groups[gid].append((p, y))

    group_label = {}
    for gid, members in groups.items():
        labs = [y for _, y in members]
        group_label[gid] = Counter(labs).most_common(1)[0][0]

    by_lab = defaultdict(list)
    for gid, lab in group_label.items():
        by_lab[lab].append(gid)

    train_g, val_g, test_g = [], [], []

    def n_img(gids):
        return sum(len(groups[g]) for g in gids)

    for lab in sorted(by_lab.keys()):
        gids = by_lab[lab][:]
        biggest = max(gids, key=lambda g: len(groups[g]))
        if tag:
            print(f"  [{tag}] train-force {lab}: {biggest} (n={len(groups[biggest])})")

        rest = [g for g in gids if g != biggest]
        rng.shuffle(rest)

        total_lab = n_img(gids)
        n_train_tgt = int(round(0.70 * total_lab))
        n_val_tgt = int(round(0.15 * total_lab))

        lab_train = [biggest]
        lab_val, lab_test = [], []
        for g in rest:
            if n_img(lab_train) < n_train_tgt:
                lab_train.append(g)
            elif n_img(lab_val) < n_val_tgt:
                lab_val.append(g)
            else:
                lab_test.append(g)

        train_g.extend(lab_train)
        val_g.extend(lab_val)
        test_g.extend(lab_test)

    assert not (set(train_g) & set(val_g))
    assert not (set(train_g) & set(test_g))
    assert not (set(val_g) & set(test_g))

    def expand(gids):
        ps, ys = [], []
        for g in gids:
            for p, y in groups[g]:
                ps.append(p)
                ys.append(y)
        return ps, ys

    splits = {
        "train": expand(train_g),
        "val": expand(val_g),
        "test": expand(test_g),
    }

    if tag:
        print(
            f"  [{tag}] split slike: "
            f"train={len(splits['train'][0])}  "
            f"val={len(splits['val'][0])}  "
            f"test={len(splits['test'][0])}"
        )
        for split_name in ("train", "val", "test"):
            _, ys = splits[split_name]
            print(f"  [{tag}] {split_name} po klasi: {dict(Counter(ys))}")

    return splits["train"], splits["val"], splits["test"]


def resize_long_side(bgr, max_side=MAX_SIDE):
    h, w = bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return bgr
    return cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def segment_wound(img_bgr):
    h, w = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    a_norm = cv2.normalize(lab[:, :, 1].astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
    s_norm = cv2.normalize(hsv[:, :, 1].astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
    score = cv2.addWeighted(a_norm, 0.6, s_norm, 0.4, 0).astype(np.uint8)
    score = cv2.GaussianBlur(score, (7, 7), 0)

    _, mask = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fallback = False

    if not contours:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fallback = True

    if not contours:
        full = np.full((h, w), 255, dtype=np.uint8)
        return full, None, True

    largest = max(contours, key=cv2.contourArea)
    area_frac = cv2.contourArea(largest) / (h * w)

    if area_frac < 0.02 and not fallback:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, mask2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours2:
            c2 = max(contours2, key=cv2.contourArea)
            if cv2.contourArea(c2) / (h * w) > area_frac:
                largest, fallback = c2, True

    clean = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(clean, [largest], -1, 255, thickness=cv2.FILLED)
    return clean, largest, fallback


def contour_features(contour, shape):
    h, w = shape[:2]
    area = float(cv2.contourArea(contour))
    peri = float(cv2.arcLength(contour, True))
    x, y, bw, bh = cv2.boundingRect(contour)
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))

    return {
        "area_px": area,
        "area_frac": area / (h * w),
        "perimeter_px": peri,
        "circularity": (4 * np.pi * area / (peri ** 2)) if peri > 0 else 0.0,
        "aspect_ratio": (bw / bh) if bh > 0 else 0.0,
        "extent": (area / (bw * bh)) if (bw * bh) > 0 else 0.0,
        "solidity": (area / hull_area) if hull_area > 0 else 0.0,
    }


def color_features(img_bgr, mask):
    m = mask > 0
    if not np.any(m):
        m = np.ones(mask.shape, dtype=bool)

    spaces = {
        "rgb": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        "hsv": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV),
        "lab": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB),
    }
    feats = {}
    for name, arr in spaces.items():
        for i in range(3):
            vals = arr[:, :, i][m].astype(np.float32)
            feats[f"{name}{i + 1}_mean"] = float(vals.mean())
            feats[f"{name}{i + 1}_std"] = float(vals.std())
    return feats


def _offset_pair(img, dy, dx):
    h, w = img.shape
    if dy >= 0:
        y0_a, y1_a, y0_b, y1_b = 0, h - dy, dy, h
    else:
        y0_a, y1_a, y0_b, y1_b = -dy, h, 0, h + dy
    if dx >= 0:
        x0_a, x1_a, x0_b, x1_b = 0, w - dx, dx, w
    else:
        x0_a, x1_a, x0_b, x1_b = -dx, w, 0, w + dx
    if y1_a <= y0_a or x1_a <= x0_a:
        return None, None
    return img[y0_a:y1_a, x0_a:x1_a], img[y0_b:y1_b, x0_b:x1_b]


def glcm_props(gray_q, levels=32):
    """Haralick (contrast, homogeneity, energy, correlation), d=1, 4 ugla."""
    offsets = [(0, 1), (1, 0), (1, 1), (1, -1)]
    acc = {"contrast": [], "homogeneity": [], "energy": [], "correlation": []}

    for dy, dx in offsets:
        a, b = _offset_pair(gray_q, dy, dx)
        if a is None or a.size == 0:
            continue

        g = np.zeros((levels, levels), dtype=np.float64)
        np.add.at(g, (a.ravel().astype(np.int32), b.ravel().astype(np.int32)), 1)
        g = g + g.T
        s = g.sum()
        if s <= 0:
            continue
        p = g / s

        i, j = np.indices((levels, levels))
        acc["contrast"].append(float((p * (i - j) ** 2).sum()))
        acc["homogeneity"].append(float((p / (1.0 + (i - j) ** 2)).sum()))
        acc["energy"].append(float((p ** 2).sum()))

        mu_i = float((p * i).sum())
        mu_j = float((p * j).sum())
        sig_i = float(np.sqrt((p * (i - mu_i) ** 2).sum()))
        sig_j = float(np.sqrt((p * (j - mu_j) ** 2).sum()))
        if sig_i > 1e-12 and sig_j > 1e-12:
            corr = float((p * (i - mu_i) * (j - mu_j)).sum() / (sig_i * sig_j))
        else:
            corr = 0.0
        acc["correlation"].append(corr)

    return {k: (float(np.mean(v)) if v else 0.0) for k, v in acc.items()}


def texture_features(img_bgr, mask):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    x, y, w, h = cv2.boundingRect(mask)
    if w == 0 or h == 0:
        x, y, w, h = 0, 0, gray.shape[1], gray.shape[0]

    roi = gray[y : y + h, x : x + w]
    roi_mask = mask[y : y + h, x : x + w]
    roi_q = np.clip((roi.astype(np.float32) / 8).astype(np.uint8), 0, 31)
    roi_q[roi_mask == 0] = 0

    feats = {f"glcm_{k}": v for k, v in glcm_props(roi_q, levels=32).items()}
    edges = cv2.Canny(gray, 100, 200)
    m = mask > 0
    feats["edge_density"] = float(edges[m].astype(bool).mean()) if np.any(m) else 0.0
    return feats


def empty_contour_feats(shape):
    h, w = shape[:2]
    return {
        "area_px": float(h * w),
        "area_frac": 1.0,
        "perimeter_px": float(2 * (h + w)),
        "circularity": np.nan,
        "aspect_ratio": w / h if h else 0.0,
        "extent": 1.0,
        "solidity": 1.0,
    }


def draw_overlay(img_bgr, contour):
    out = img_bgr.copy()
    cv2.drawContours(out, [contour], -1, (0, 255, 0), 2)
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return out


def extract_one(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    bgr = resize_long_side(bgr)
    mask, contour, fallback = segment_wound(bgr)

    row = {"fallback_segmentation": int(fallback)}
    if contour is not None:
        row.update(contour_features(contour, bgr.shape))
    else:
        row.update(empty_contour_feats(bgr.shape))
    row.update(color_features(bgr, mask))
    row.update(texture_features(bgr, mask))
    return row, bgr, contour


def mask_quality(feats):
    """Veci skor = uverljivija maska (heuristika za izbor primera)."""
    if feats.get("fallback_segmentation", 0):
        return -10.0
    af = float(feats["area_frac"])
    sol = float(feats["solidity"])
    circ = float(feats["circularity"]) if np.isfinite(feats["circularity"]) else 0.0

    score = sol
    if 0.04 <= af <= 0.40:
        score += 2.0
    elif 0.02 <= af <= 0.55:
        score += 0.5
    else:
        score -= 3.0 * abs(af - 0.22)

    if 0.15 <= circ <= 0.85:
        score += 0.5
    else:
        score -= 0.5
    return score


def show_contour_grid(overlays_rgb, labels, title="Klasicna segmentacija / konture (test)"):
    n = len(overlays_rgb)
    if n == 0:
        return
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, rgb, lab in zip(axes, overlays_rgb, labels):
        ax.imshow(rgb)
        ax.set_title(lab)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def show_good_vs_bad(good, bad):
    """good/bad: liste (rgb, label_text)."""
    n = max(len(good), len(bad), 1)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 7))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for j in range(n):
        if j < len(good):
            axes[0, j].imshow(good[j][0])
            axes[0, j].set_title(f"OK: {good[j][1]}")
        axes[0, j].axis("off")
        if j < len(bad):
            axes[1, j].imshow(bad[j][0])
            axes[1, j].set_title(f"Greska: {bad[j][1]}")
        axes[1, j].axis("off")

    fig.suptitle(
        "Konture: uspesni vs problematicni primeri\n"
        "(bojom vodjena maska ponekad uhvati lenjir/kozu umesto rane)"
    )
    fig.tight_layout()
    plt.show()


def show_feature_boxplots(df):
    cols = [
        ("area_frac", "Udeo povrsine ROI"),
        ("circularity", "Circularnost"),
        ("edge_density", "Gustina ivica (Canny)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    tick_labels = [c.replace("_", " ") for c in NPUAP]

    for ax, (col, title) in zip(axes, cols):
        data = [
            df.loc[df["label"] == c, col].dropna().to_numpy()
            for c in NPUAP
        ]
        ax.boxplot(data, tick_labels=tick_labels, showfliers=False)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("Deskriptori po NPUAP klasi (ceo Valid skup)")
    fig.tight_layout()
    plt.show()


def show_comparison_bars(results):
    names = ["Majority", "RF", "SVM RBF", "EfficientNet-B0"]
    keys = [
        "majority_baseline",
        "random_forest",
        "svm_rbf",
        "cnn_model_b_reference",
    ]
    acc = [results[k]["accuracy"] for k in keys]
    f1 = [results[k]["macro_f1"] for k in keys]

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, acc, w, label="Accuracy")
    ax.bar(x + w / 2, f1, w, label="Macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Klasicni CV vs Model B (isti group-aware test)")
    ax.legend()
    fig.tight_layout()
    plt.show()


def show_confusion(cm, title):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(NPUAP)))
    ax.set_yticks(range(len(NPUAP)))
    ax.set_xticklabels([c.replace("_", " ") for c in NPUAP], rotation=45, ha="right")
    ax.set_yticklabels([c.replace("_", " ") for c in NPUAP])
    ax.set_xlabel("Predikcija")
    ax.set_ylabel("Stvarno")
    ax.set_title(title)
    for i in range(len(NPUAP)):
        for j in range(len(NPUAP)):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    plt.show()


def train_baselines(df, cnn_acc, cnn_f1):
    meta = {"path", "label", "split", "fallback_segmentation"}
    feat_cols = [c for c in df.columns if c not in meta]

    train_df = df[df["split"].isin(["train", "val"])]
    test_df = df[df["split"] == "test"]

    X_tr = train_df[feat_cols].to_numpy(dtype=np.float64)
    y_tr = train_df["label"].to_numpy()
    X_te = test_df[feat_cols].to_numpy(dtype=np.float64)
    y_te = test_df["label"].to_numpy()

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(imputer.fit_transform(X_tr))
    X_te = scaler.transform(imputer.transform(X_te))

    results = {}

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)
    rf_pred = rf.predict(X_te)
    results["random_forest"] = {
        "accuracy": float(accuracy_score(y_te, rf_pred)),
        "macro_f1": float(f1_score(y_te, rf_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_te, rf_pred, labels=NPUAP),
        "feature_importance_top10": sorted(
            zip(feat_cols, rf.feature_importances_.tolist()),
            key=lambda t: t[1],
            reverse=True,
        )[:10],
    }

    svm = SVC(C=10, kernel="rbf", class_weight="balanced", random_state=SEED)
    svm.fit(X_tr, y_tr)
    svm_pred = svm.predict(X_te)
    results["svm_rbf"] = {
        "accuracy": float(accuracy_score(y_te, svm_pred)),
        "macro_f1": float(f1_score(y_te, svm_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_te, svm_pred, labels=NPUAP),
    }

    majority = Counter(y_tr).most_common(1)[0][0]
    maj_pred = np.full(len(y_te), majority)
    results["majority_baseline"] = {
        "accuracy": float(accuracy_score(y_te, maj_pred)),
        "macro_f1": float(f1_score(y_te, maj_pred, average="macro", zero_division=0)),
        "predicted_class": majority,
    }

    results["cnn_model_b_reference"] = {
        "accuracy": cnn_acc,
        "macro_f1": cnn_f1,
        "note": f"ucitano iz {METRICS_JSON.name}",
    }
    results["n_trainval"] = int(len(train_df))
    results["n_test"] = int(len(test_df))
    results["feature_cols"] = feat_cols

    print("\n=== Majority baseline (test) ===")
    print(
        f"acc={results['majority_baseline']['accuracy']:.4f}  "
        f"macro_f1={results['majority_baseline']['macro_f1']:.4f}"
    )

    print("\n=== Random Forest (test) ===")
    print(classification_report(y_te, rf_pred, labels=NPUAP, zero_division=0))

    print("\n=== SVM RBF (test) ===")
    print(classification_report(y_te, svm_pred, labels=NPUAP, zero_division=0))

    print("\n=== Poredjenje (isti group-aware test protokol) ===")
    print(f"{'Model':<28} {'Acc':>8} {'Macro-F1':>10}")
    print("-" * 48)
    for name, key in [
        ("Majority", "majority_baseline"),
        ("RF (klasicni CV)", "random_forest"),
        ("SVM RBF (klasicni CV)", "svm_rbf"),
        ("EfficientNet-B0 (Model B)", "cnn_model_b_reference"),
    ]:
        r = results[key]
        print(f"{name:<28} {r['accuracy']:>8.3f} {r['macro_f1']:>10.3f}")

    return results


def main():
    cnn_acc, cnn_f1 = load_cnn_metrics()
    print(f"Model B (iz {METRICS_JSON.name}): acc={cnn_acc:.3f}  macro_f1={cnn_f1:.3f}")

    paths, labels, skipped = collect_paths(DATA, NPUAP)
    print(f"Valid slike: {len(paths)}  (preskoceno <{MIN_KB} KB: {len(skipped)})")

    group_map = load_group_map()
    matched = sum(1 for p in paths if path_key(p) in group_map)
    print(f"phash grupe: {matched}/{len(paths)} putanja pogodjeno")

    (tr_p, tr_y), (va_p, va_y), (te_p, te_y) = group_split(
        paths, labels, group_map, seed=SEED, tag="B-classical"
    )

    split_of = {}
    for p, y in zip(tr_p, tr_y):
        split_of[path_key(p)] = ("train", y)
    for p, y in zip(va_p, va_y):
        split_of[path_key(p)] = ("val", y)
    for p, y in zip(te_p, te_y):
        split_of[path_key(p)] = ("test", y)

    all_items = []
    for p, y in zip(paths, labels):
        sp, _ = split_of[path_key(p)]
        all_items.append((p, y, sp))

    rows = []
    fallback_n = 0
    overlay_count = {c: 0 for c in NPUAP}
    overlay_rgb = []
    overlay_labels = []
    # (quality, rgb, label) sa testa — za dobre/lose primere
    scored = []

    print(f"Ekstrakcija obelezja: {len(all_items)} slika...")
    for i, (p, y, sp) in enumerate(all_items):
        out = extract_one(p)
        if out is None:
            print(f"  nije ucitano: {p}")
            continue
        feats, bgr, contour = out
        if feats["fallback_segmentation"]:
            fallback_n += 1

        rows.append({
            "path": Path(p).as_posix(),
            "label": y,
            "split": sp,
            **feats,
        })

        if contour is not None and sp == "test":
            rgb = cv2.cvtColor(draw_overlay(bgr, contour), cv2.COLOR_BGR2RGB)
            q = mask_quality(feats)
            scored.append((q, rgb, y.replace("_", " ")))

            if overlay_count[y] < OVERLAYS_PER_CLASS:
                overlay_rgb.append(rgb)
                overlay_labels.append(y.replace("_", " "))
                overlay_count[y] += 1

        if (i + 1) % 200 == 0 or (i + 1) == len(all_items):
            print(f"  {i + 1}/{len(all_items)}")

    if any(overlay_count[c] < OVERLAYS_PER_CLASS for c in NPUAP):
        for p, y, sp in all_items:
            if overlay_count[y] >= OVERLAYS_PER_CLASS or sp == "test":
                continue
            out = extract_one(p)
            if out is None:
                continue
            _, bgr, contour = out
            if contour is None:
                continue
            rgb = cv2.cvtColor(draw_overlay(bgr, contour), cv2.COLOR_BGR2RGB)
            overlay_rgb.append(rgb)
            overlay_labels.append(y.replace("_", " "))
            overlay_count[y] += 1
            if all(overlay_count[c] >= OVERLAYS_PER_CLASS for c in NPUAP):
                break

    df = pd.DataFrame(rows)
    print(
        f"\nObelezja: {len(df)} slika | "
        f"fallback={fallback_n} ({100 * fallback_n / max(len(df), 1):.2f}%)"
    )
    print(f"Split: {dict(df['split'].value_counts())}")

    results = train_baselines(df, cnn_acc, cnn_f1)

    print("\nRF top-10 feature importance:")
    for name, imp in results["random_forest"]["feature_importance_top10"]:
        print(f"  {name:20s}  {imp:.4f}")

    order = {c.replace("_", " "): i for i, c in enumerate(NPUAP)}
    paired = sorted(zip(overlay_rgb, overlay_labels), key=lambda t: order.get(t[1], 99))
    if paired:
        overlay_rgb, overlay_labels = zip(*paired)
        show_contour_grid(list(overlay_rgb), list(overlay_labels))

    if len(scored) >= 4:
        scored.sort(key=lambda t: t[0], reverse=True)
        good = [(rgb, lab) for _, rgb, lab in scored[:2]]
        bad = [(rgb, lab) for _, rgb, lab in scored[-2:]]
        print("\nPrimeri kontura (heuristika kvaliteta maske):")
        print(f"  OK:    {[g[1] for g in good]}  score={[round(scored[i][0], 2) for i in range(2)]}")
        print(f"  greska:{[b[1] for b in bad]}  score={[round(scored[-2 + i][0], 2) for i in range(2)]}")
        show_good_vs_bad(good, bad)

    show_feature_boxplots(df)
    show_comparison_bars(results)
    show_confusion(results["random_forest"]["confusion_matrix"], "RF - matrica konfuzije")
    show_confusion(results["svm_rbf"]["confusion_matrix"], "SVM RBF - matrica konfuzije")



if __name__ == "__main__":
    main()
