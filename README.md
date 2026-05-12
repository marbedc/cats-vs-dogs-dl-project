# Cats vs Dogs Deep Learning Project

## Overview

This project implements deep learning models to classify images as either **cats** or **dogs**.
We compare:

* A **custom CNN (baseline)**
* A **pretrained model (transfer learning)**

---

## Repository Structure

```
cats-vs-dogs-dl-project/
│
├── data/                # Processed dataset (not included in repo)
├── data_raw/            # Original dataset (not included in repo)
├── models/
│   ├── cnn_model.py
│   └── pretrained_model.py
├── utils/
│   ├── data.py
│   ├── train_eval.py
│   └── metrics.py
├── prepare_data.py      # Script to clean and split dataset
├── train_cnn.py         # Train custom CNN
├── train_pretrained.py  # Train pretrained model
├── evaluate.py          # Evaluate models
└── README.md
```

---

## Setup Instructions

### 1. Clone the repository

```
git clone <YOUR_REPO_URL>
cd cats-vs-dogs-dl-project
```

---

### 2. Install dependencies

```
pip install torch torchvision pillow matplotlib scikit-learn
```

---

## Dataset Setup

### Step 1: Download dataset

Download the **Cats vs Dogs dataset** from one of the following:
* Microsoft: https://www.microsoft.com/en-us/download/details.aspx?id=54765

---

### Step 2: Extract dataset

After downloading, extract it so that you have:

```
data_raw/
└── PetImages/
    ├── Cat/
    └── Dog/
```

---

### Step 3: Prepare dataset

Run the preprocessing script:

```
python prepare_data.py
```

This script will:

* Remove corrupted images
* Select **4000 images per class**
* Split data into:

  * 70% training
  * 15% validation
  * 15% testing
* Create the following structure:

```
data/
├── train/
├── val/
└── test/
```

---

## Important Notes (Reproducibility)

* The script uses a fixed random seed (`random.seed(42)`)
* This ensures **all team members get the exact same dataset split**
* Do **not modify** `prepare_data.py` before running

---

## Training

### Train the CNN (baseline)

```
python train_cnn.py
```

This will:

* Train the custom CNN
* Save the best model as `best_cnn.pth`

---

### Train the pretrained model

```
python train_pretrained.py
```
* Google Collab Link: https://colab.research.google.com/drive/1VLYem35bl6aFnp383L524-KEGpIyWyVP?usp=sharing
---

## Evaluation

Run:

```
python evaluate.py
```

This will compute:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

---

## GitHub Notes

* The dataset is **not included** in this repository
* Folders ignored by Git:

  * `data/`
  * `data_raw/`
  * `*.pth`

---

## Team Roles

* Person 1: Custom CNN
* Person 2: Pretrained model
* Person 3: Evaluation and analysis

---

## Summary

1. Clone repo
2. Download dataset
3. Run `prepare_data.py`
4. Train models
5. Evaluate results

---
