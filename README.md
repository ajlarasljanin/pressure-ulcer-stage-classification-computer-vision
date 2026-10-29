# Pressure Ulcer Stage Classification — Computer Vision

**Klasifikacija stadijuma dekubitusa primenom dubokog učenja nad RGB fotografijama rana**

Seminar project · Master studies in Artificial Intelligence, University of Kragujevac · Course: Computer Vision
Student: Ajla Rašljanin · Mentor: Bogdan Milicevic

> 📄 The accompanying seminar paper is written in **Serbian** (Latin script) and is not included in this repository. Code comments are in Serbian.

## Overview

A two-stage system for RGB wound photographs:

- **Model A (triage):** `Invalid` vs `Valid`. Filters out images that are not usable wound photos.
- **Model B (staging):** NPUAP stage classification into 6 classes: `SDTI`, `Stage_I`, `Stage_II`, `Stage_III`, `Stage_IV`, `Unstageable`.

Both models are **EfficientNet-B0** (ImageNet weights, PyTorch) fine-tuned on 224×224 images, with OpenCV preprocessing and **Grad-CAM** to show which regions drive each prediction.
A **classical computer vision baseline** (wound segmentation, contour-based shape descriptors, Random Forest / SVM) is trained on the same split for comparison.

### Avoiding data leakage

The expanded dataset contains augmented variants of the same original photos. A plain random split by file would put near-identical images in both train and test sets and inflate the scores.
Images are therefore grouped by **perceptual hash (pHash)** similarity (`4_phash_groups.py` → `phash_groups.csv`), and the data is split **by group** (70 / 15 / 15, seed 42). No group appears in more than one split.

## Results (test set, group-aware split)

| Model | Task | Accuracy | Macro F1 | n (test) |
|-------|------|---------:|---------:|---------:|
| Model A | Invalid vs Valid | 0.962 | 0.923 | 344 |
| Model B | NPUAP staging (6 classes) | 0.742 | 0.745 | 299 |

Training curves, confusion matrices and the comparison with the classical CV baseline are in [`imageslatex/`](imageslatex/).

## Repository structure

| Path | Description |
|------|-------------|
| `1_analyze_old_dataset.py` | EDA of the original dataset (258 images) |
| `2_compare_datasets.py` | Comparison of the original and expanded datasets (MD5 duplicates, distributions) |
| `3_analyze_expanded_dataset.py` | EDA of the expanded working dataset (~2300 images) |
| `4_phash_groups.py` | pHash grouping of near-duplicates → `phash_groups.csv` |
| `5_train_pipeline.py` | Training of Model A and Model B, evaluation, Grad-CAM |
| `6_classical_cv.py` | Classical CV baseline (segmentation, descriptors, RF/SVM) |
| `phash_groups.csv` | Precomputed pHash groups used for the split |
| `cnn_test_metrics.json` | CNN test metrics |
| `imageslatex/` | Figures from the seminar paper (plots only, see `NOTES.md`) |

## Dataset

The dataset is **not included** in this repository. Both sets are available on Kaggle:

| Set | Used as | Source |
|-----|---------|--------|
| Pressure Ulcer Stages | original set (`Dataset/`, 258 images) | [kaggle.com/datasets/sinemgokoz/pressure-ulcers-stages](https://www.kaggle.com/datasets/sinemgokoz/pressure-ulcers-stages) |
| Pressure Ulcers Detection | expanded working set (`Dataset - expanded/pressure_ulcers_expanded/`, ~2300 images) | [kaggle.com/datasets/divyanshmahajan2311/pressure-ulcers-detection](https://www.kaggle.com/datasets/divyanshmahajan2311/pressure-ulcers-detection) |

The scripts expect the following layout in the project root:

```
Dataset/                                   # original set, one folder per class
Dataset - expanded/pressure_ulcers_expanded/
    Invalid/  SDTI/  Stage_I/  Stage_II/  Stage_III/  Stage_IV/  Unstageable/
```

## How to run

Run the scripts from the project root (e.g. in Spyder, with the working directory set to the root):

```bash
pip install -r requirements.txt
python 4_phash_groups.py      # required before training
python 5_train_pipeline.py    # Model A, Model B, Grad-CAM
python 6_classical_cv.py      # classical CV baseline
```

Scripts `1`–`3` are optional (exploratory analysis).
Hyperparameters: image size 224, batch size 16, Adam with lr 1e-4, 10 epochs (Model A) / 15 epochs (Model B), class-weighted loss. A GPU is used when available.

## Disclaimer

This is an academic project. It is not a medical device and must not be used for clinical diagnosis.
