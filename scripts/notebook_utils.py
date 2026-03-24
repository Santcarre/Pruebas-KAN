from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms
except ModuleNotFoundError:  # pragma: no cover - allows light notebooks to import without torch
    torch = None
    Dataset = object
    transforms = None


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def project_root(cwd: Path | None = None) -> Path:
    cwd = (cwd or Path.cwd()).resolve()
    return cwd.parent if cwd.name == "notebooks" else cwd


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(prefer_cpu: bool = False):
    if torch is None:
        raise ModuleNotFoundError("torch is required to pick a training device")
    if prefer_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_manifest_splits(
    train_csv: Path,
    test_csv: Path,
    seed: int = 42,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    patients = train_df["patient_id"].dropna().unique()
    train_pat, val_pat = train_test_split(
        patients,
        test_size=test_size,
        random_state=seed,
    )
    tr_df = train_df[train_df["patient_id"].isin(train_pat)].copy()
    val_df = train_df[train_df["patient_id"].isin(val_pat)].copy()
    return tr_df, val_df, test_df


def default_transforms(
    img_size: int = 224,
    augment: bool = False,
    color_jitter: bool = False,
) -> "transforms.Compose":
    if transforms is None:
        raise ModuleNotFoundError("torchvision is required to build image transforms")
    steps: list[object] = [transforms.Resize((img_size, img_size))]
    if augment:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=12),
            ]
        )
        if color_jitter:
            steps.append(transforms.ColorJitter(brightness=0.05, contrast=0.05))
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(steps)


class MammographyDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform=None,
        return_path: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.return_path = return_path

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path_local"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(row["label"])
        if self.return_path:
            return image, label, row["image_path_local"]
        return image, label


def class_weights_from_labels(labels: Iterable[int], device="cpu"):
    if torch is None:
        raise ModuleNotFoundError("torch is required to compute class weights")
    counts = Counter(int(label) for label in labels)
    total = sum(counts.values())
    return torch.tensor(
        [total / (2 * counts[0]), total / (2 * counts[1])],
        dtype=torch.float32,
        device=device,
    )


def choose_threshold_by_recall(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    target_recall: float = 0.80,
) -> float:
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5
    recall_t = recall[:-1]
    precision_t = precision[:-1]
    valid_idx = np.where(recall_t >= target_recall)[0]
    if len(valid_idx) == 0:
        best_idx = int(np.argmax(recall_t))
    else:
        best_idx = int(valid_idx[np.argmax(precision_t[valid_idx])])
    return float(thresholds[best_idx])


def safe_binary_metrics(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    y_prob: np.ndarray | list[float] | None = None,
) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "recall_malignant": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_malignant": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_malignant": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        metrics["auc"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    return metrics
