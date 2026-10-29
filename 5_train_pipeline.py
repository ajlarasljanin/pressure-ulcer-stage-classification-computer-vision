"""
Trening + Grad-CAM: klasifikacija dekubitusa (2-step)

A) Valid vs Invalid
B) NPUAP staging (6 klasa) + Grad-CAM

Predprocesiranje / overlay: OpenCV
Klasifikacija: PyTorch (EfficientNet-B0)
"""

from pathlib import Path
import json
import random
import copy
from collections import Counter, defaultdict

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

ROOT = Path(".")
DATA = ROOT / "Dataset - expanded" / "pressure_ulcers_expanded"
GROUPS_CSV = ROOT / "phash_groups.csv"
METRICS_JSON = ROOT / "cnn_test_metrics.json"

NPUAP = ["SDTI", "Stage_I", "Stage_II", "Stage_III", "Stage_IV", "Unstageable"]
ALL_DIRS = ["Invalid"] + NPUAP
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS_A = 10
EPOCHS_B = 15
LR = 1e-4
SEED = 42
MIN_KB = 2.0
N_GRADCAM = 2  # primera po klasi za Grad-CAM

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


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


def load_rgb_cv2(path, size=IMG_SIZE):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cv2.imread nije ucitao: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return rgb


def augment_cv2(rgb):
    if random.random() < 0.5:
        rgb = cv2.flip(rgb, 1)

    angle = random.uniform(-15, 15)
    center = (IMG_SIZE / 2.0, IMG_SIZE / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rgb = cv2.warpAffine(
        rgb, M, (IMG_SIZE, IMG_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    alpha = random.uniform(0.85, 1.15)
    beta = random.uniform(-15, 15)
    rgb = cv2.convertScaleAbs(rgb, alpha=alpha, beta=beta)
    return rgb


def to_tensor_norm(rgb):
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x.copy())


class ImageFolderList(Dataset):
    def __init__(self, paths, labels, class_to_idx, train=False):
        self.paths = paths
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        rgb = load_rgb_cv2(self.paths[i])
        if self.train:
            rgb = augment_cv2(rgb)
        x = to_tensor_norm(rgb)
        y = self.class_to_idx[self.labels[i]]
        return x, y


def path_key(p):
    return Path(p).as_posix().lower()


def load_group_map(csv_path=GROUPS_CSV):
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Nema {csv_path} — prvo pokreni 4_phash_groups.py")
    df = pd.read_csv(csv_path)
    return {path_key(p): g for p, g in zip(df["path"], df["group_id"])}


def group_split(paths, labels, group_map, seed=SEED, tag=""):
    """
    ~70/15/15 po phash grupama, stratifikovano PO LABELLI.
    Najveca grupa po klasi ide u train (megagrupa ne ulazi u test/val).
    """
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
        print(f"  [{tag}] split slike: "
              f"train={len(splits['train'][0])}  "
              f"val={len(splits['val'][0])}  "
              f"test={len(splits['test'][0])}")
        print(f"  [{tag}] split grupe: "
              f"train={len(train_g)}  val={len(val_g)}  test={len(test_g)}")
        for split_name in ("train", "val", "test"):
            _, ys = splits[split_name]
            print(f"  [{tag}] {split_name} po klasi: {dict(Counter(ys))}")
        test_paths, test_labs = splits["test"]
        for lab in sorted(set(test_labs)):
            idx = [i for i, y in enumerate(test_labs) if y == lab]
            if not idx:
                continue
            gcounts = Counter(
                group_map.get(path_key(test_paths[i]), path_key(test_paths[i]))
                for i in idx
            )
            top_g, top_n = gcounts.most_common(1)[0]
            print(
                f"  [{tag}] test {lab}: n={len(idx)}  "
                f"max_grupa_udeo={100 * top_n / len(idx):.1f}% ({top_g})"
            )

    return splits["train"], splits["val"], splits["test"]


def class_weights(labels, class_names):
    cnt = Counter(labels)
    n = len(labels)
    k = len(class_names)
    w = [n / (k * max(cnt[c], 1)) for c in class_names]
    return torch.tensor(w, dtype=torch.float32)


def build_model(n_classes):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
    in_f = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_f, n_classes)
    return model.to(DEVICE)


def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, n = 0.0, 0
    ys, ps = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.set_grad_enabled(train):
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        pred = out.argmax(1)
        ys.extend(y.cpu().tolist())
        ps.extend(pred.cpu().tolist())
    return total_loss / max(n, 1), ys, ps


def train_model(model, train_loader, val_loader, criterion, epochs, tag):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best_f1, best_state, best_epoch = -1.0, None, -1
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}

    for ep in range(1, epochs + 1):
        tr_loss, tr_y, tr_p = run_epoch(model, train_loader, criterion, opt)
        va_loss, va_y, va_p = run_epoch(model, val_loader, criterion, None)
        tr_f1 = f1_score(tr_y, tr_p, average="macro", zero_division=0)
        va_f1 = f1_score(va_y, va_p, average="macro", zero_division=0)
        print(
            f"[{tag}] epoch {ep:02d}/{epochs}  "
            f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f}  "
            f"train_f1={tr_f1:.3f} val_f1={va_f1:.3f}"
        )
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_f1"].append(tr_f1)
        history["val_f1"].append(va_f1)
        if va_f1 > best_f1:
            best_f1 = va_f1
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = ep

    if best_state is not None:
        model.load_state_dict(best_state)
    plot_training_curves(history, best_epoch, tag)
    return model

def plot_training_curves(history, best_epoch, tag):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(epochs, history["train_loss"], marker="o", ms=4, label="Trening", color="#1b6ca8")
    axes[0].plot(epochs, history["val_loss"], marker="o", ms=4, label="Validacija", color="#d95f02")
    axes[0].axvline(best_epoch, color="gray", ls="--", lw=1, label=f"Izabrana epoha ({best_epoch})")
    axes[0].set_xlabel("Epoha")
    axes[0].set_ylabel("Loss (CrossEntropy)")
    axes[0].set_title(f"{tag} — funkcija gubitka")
    axes[0].set_xticks(list(epochs))
    axes[0].legend(fontsize=8)

    axes[1].plot(epochs, history["train_f1"], marker="o", ms=4, label="Trening", color="#1b6ca8")
    axes[1].plot(epochs, history["val_f1"], marker="o", ms=4, label="Validacija", color="#d95f02")
    axes[1].axvline(best_epoch, color="gray", ls="--", lw=1, label=f"Izabrana epoha ({best_epoch})")
    axes[1].set_xlabel("Epoha")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_title(f"{tag} — Macro-F1")
    axes[1].set_xticks(list(epochs))
    axes[1].legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    plt.show()

def evaluate(model, loader, class_names, tag):
    model.eval()
    _, ys, ps = run_epoch(model, loader, nn.CrossEntropyLoss(), None)
    acc = accuracy_score(ys, ps)
    f1 = f1_score(ys, ps, average="macro", zero_division=0)
    report = classification_report(
        ys, ps, target_names=class_names, digits=3, zero_division=0
    )
    cm = confusion_matrix(ys, ps)

    print(f"\n[{tag}] test accuracy={acc:.3f}  macro-F1={f1:.3f}")
    print(report)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predikcija")
    ax.set_ylabel("Stvarna klasa")
    ax.set_title(f"Matrica konfuzije — {tag}")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    plt.show()
    return acc, f1


def make_loaders(paths_labels, class_names):
    (p_tr, y_tr), (p_va, y_va), (p_te, y_te) = paths_labels
    c2i = {c: i for i, c in enumerate(class_names)}
    ds_tr = ImageFolderList(p_tr, y_tr, c2i, train=True)
    ds_va = ImageFolderList(p_va, y_va, c2i, train=False)
    ds_te = ImageFolderList(p_te, y_te, c2i, train=False)
    ld_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    ld_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    ld_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return ld_tr, ld_va, ld_te, y_tr


# ---------- Grad-CAM ----------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, module, inp, out):
        self.activations = out.detach()

    def _bwd(self, module, gin, gout):
        self.gradients = gout[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        score = logits[0, class_idx]
        score.backward()

        grads = self.gradients[0]
        acts = self.activations[0]
        weights = grads.mean(dim=(1, 2))
        cam = torch.zeros(acts.shape[1:], device=acts.device)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = F.interpolate(
            cam[None, None],
            size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear",
            align_corners=False,
        )[0, 0].cpu().numpy()
        probs = torch.softmax(logits, 1)[0].detach().cpu().numpy()
        return cam, class_idx, probs


def make_heatmap(cam):
    cam_u8 = np.clip(cam * 255.0, 0, 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)


def overlay_cam(rgb, cam, alpha=0.45):
    heat = make_heatmap(cam)
    return cv2.addWeighted(rgb, 1.0 - alpha, heat, alpha, 0)


def panel_three(rgb, cam, blend):
    return cv2.hconcat([rgb, make_heatmap(cam), blend])


def run_gradcam(model, paths_list, labels_list, class_names=None, n_per_class=N_GRADCAM):
    """Grad-CAM nad datim skupom (ocekivano: test)."""
    if class_names is None:
        class_names = NPUAP
    model.eval()
    cam_fn = GradCAM(model, model.features[-1])

    by_cls = {c: [] for c in class_names}
    for p, c in zip(paths_list, labels_list):
        if c in by_cls:
            by_cls[c].append(p)

    selected = []
    for c in class_names:
        files = by_cls[c][:]
        random.shuffle(files)
        selected.extend((p, c) for p in files[:n_per_class])

    print(f"\nGrad-CAM (test): {len(selected)} primera")
    for path, true_cls in selected:
        rgb = load_rgb_cv2(path)
        x = to_tensor_norm(rgb).unsqueeze(0).to(DEVICE)
        cam, pred_idx, probs = cam_fn(x)
        pred_cls = class_names[pred_idx]
        conf = float(probs[pred_idx])
        panel = panel_three(rgb, cam, overlay_cam(rgb, cam))

        fig, ax = plt.subplots(figsize=(10, 3.2))
        ax.imshow(panel)
        ax.axis("off")
        ax.set_title(
            f"true={true_cls} | pred={pred_cls} ({conf:.2f}) | {path.name}",
            fontsize=10,
        )
        fig.tight_layout()
        plt.show()


# ========================= main =========================
paths, labels, skipped = collect_paths(DATA, ALL_DIRS, min_kb=MIN_KB)
print(f"Ucitano: {len(paths)}")
print(f"Preskoceno (< {MIN_KB} KB): {len(skipped)}")
print("Raspodela:", Counter(labels))

group_map = load_group_map()
matched = sum(1 for p in paths if path_key(p) in group_map)
print(f"phash grupe: {matched}/{len(paths)} putanja pogodjeno")

# --- MODEL A ---
labels_a = ["Invalid" if y == "Invalid" else "Valid" for y in labels]
classes_a = ["Invalid", "Valid"]

print("\nMODEL A: Valid vs Invalid")
print(Counter(labels_a))
split_a = group_split(paths, labels_a, group_map, tag="A")
ld_tr, ld_va, ld_te, y_tr_a = make_loaders(split_a, classes_a)

model_a = build_model(len(classes_a))
crit_a = nn.CrossEntropyLoss(weight=class_weights(y_tr_a, classes_a).to(DEVICE))
model_a = train_model(model_a, ld_tr, ld_va, crit_a, EPOCHS_A, "modelA_triage")
acc_a, f1_a = evaluate(model_a, ld_te, classes_a, "modelA_triage")

# --- MODEL B ---
paths_b = [p for p, y in zip(paths, labels) if y != "Invalid"]
labels_b = [y for y in labels if y != "Invalid"]

print("\nMODEL B: NPUAP staging (6 klasa)")
print(Counter(labels_b))
split_b = group_split(paths_b, labels_b, group_map, tag="B")
ld_tr, ld_va, ld_te, y_tr_b = make_loaders(split_b, NPUAP)
p_te_b, y_te_b = split_b[2]

model_b = build_model(len(NPUAP))
crit_b = nn.CrossEntropyLoss(weight=class_weights(y_tr_b, NPUAP).to(DEVICE))
model_b = train_model(model_b, ld_tr, ld_va, crit_b, EPOCHS_B, "modelB_staging")
acc_b, f1_b = evaluate(model_b, ld_te, NPUAP, "modelB_staging")

metrics = {
    "model_a": {
        "tag": "modelA_triage",
        "accuracy": float(acc_a),
        "macro_f1": float(f1_a),
        "n_test": int(len(split_a[2][1])),
    },
    "model_b": {
        "tag": "modelB_staging",
        "accuracy": float(acc_b),
        "macro_f1": float(f1_b),
        "n_test": int(len(y_te_b)),
    },
    "seed": SEED,
    "group_split": "phash 70/15/15",
}

with open(METRICS_JSON, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrike sacuvane: {METRICS_JSON}")

# Grad-CAM samo na test skupu modela B
run_gradcam(model_b, p_te_b, y_te_b, class_names=NPUAP)

print("\nGotovo.")
