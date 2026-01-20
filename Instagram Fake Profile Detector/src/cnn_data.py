from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .config import POST_LABEL_COL, POST_METADATA_PATH, VIDEO_FRAMES_DIR


class FrameDataset(Dataset):
    def __init__(
        self,
        frames_dir: Path,
        labels_map: dict[str, int],
        usernames: list[str],
        transform=None,
        max_frames_per_video: int = 0,
    ) -> None:
        self.items: list[tuple[Path, int]] = []
        self.transform = transform

        for username in usernames:
            label = labels_map.get(username)
            if label is None:
                continue
            user_dir = frames_dir / username
            if not user_dir.exists():
                continue
            for video_dir in sorted(user_dir.iterdir()):
                if not video_dir.is_dir():
                    continue
                frames = sorted(video_dir.glob("frame_*.jpg"))
                if max_frames_per_video > 0:
                    frames = frames[:max_frames_per_video]
                for frame_path in frames:
                    self.items.append((frame_path, int(label)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, float(label)


def load_username_labels(csv_path: Path | None = None) -> dict[str, int]:
    data_path = csv_path or POST_METADATA_PATH
    df = pd.read_csv(data_path)
    if "username" not in df.columns or POST_LABEL_COL not in df.columns:
        raise ValueError("ig_post_metadata.csv must contain username and label_fake columns")

    df = df.copy()
    df["username"] = (
        df["username"].astype(str).str.strip().str.lstrip("@").str.lower()
    )

    labels = df.groupby("username")[POST_LABEL_COL].first()
    return {username: int(label) for username, label in labels.items()}


def list_frames_for_user(
    frames_dir: Path,
    username: str,
    max_frames_per_video: int = 0,
) -> list[Path]:
    user_key = (username or "").strip().lstrip("@").lower()
    if not user_key:
        return []

    user_dir = frames_dir / user_key
    if not user_dir.exists():
        return []

    frames: list[Path] = []
    for video_dir in sorted(user_dir.iterdir()):
        if not video_dir.is_dir():
            continue
        frame_paths = sorted(video_dir.glob("frame_*.jpg"))
        if max_frames_per_video > 0:
            frame_paths = frame_paths[:max_frames_per_video]
        frames.extend(frame_paths)

    return frames


def split_usernames_for_training(
    labels_map: dict[str, int],
    frames_dir: Path | None = None,
    val_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    root = frames_dir or VIDEO_FRAMES_DIR
    usernames = [u for u in labels_map if (root / u).exists()]
    if not usernames:
        raise ValueError("No usernames with extracted frames found.")

    y = [labels_map[u] for u in usernames]
    try:
        train_users, val_users = train_test_split(
            usernames, test_size=val_size, random_state=seed, stratify=y
        )
    except ValueError:
        train_users, val_users = train_test_split(
            usernames, test_size=val_size, random_state=seed
        )

    return list(train_users), list(val_users)
