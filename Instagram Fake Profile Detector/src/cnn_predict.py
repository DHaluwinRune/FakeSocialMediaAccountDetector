from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .cnn_data import list_frames_for_user
from .cnn_model import build_transforms, load_model_checkpoint
from .config import CNN_MODEL_PATH, VIDEO_FRAMES_DIR


class _FrameInferenceDataset(Dataset):
    def __init__(self, frame_paths: list[Path], transform=None) -> None:
        self.frame_paths = frame_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, idx: int):
        path = self.frame_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def predict_username_from_frames(
    username: str,
    threshold: float = 0.5,
    frames_dir: Path | None = None,
    model_path: Path | None = None,
    batch_size: int = 32,
    max_frames_per_video: int = 0,
) -> dict:
    frames_root = frames_dir or VIDEO_FRAMES_DIR
    model_path = model_path or CNN_MODEL_PATH

    frame_paths = list_frames_for_user(frames_root, username, max_frames_per_video)
    if not frame_paths:
        raise ValueError("No frames found for this username. Run frame extraction first.")

    return predict_from_frame_paths(
        frame_paths,
        threshold=threshold,
        model_path=model_path,
        batch_size=batch_size,
    )


def predict_from_frame_paths(
    frame_paths: list[Path],
    threshold: float = 0.5,
    model_path: Path | None = None,
    batch_size: int = 32,
) -> dict:
    if not frame_paths:
        raise ValueError("No frames provided for inference.")

    model_path = model_path or CNN_MODEL_PATH

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, image_size = load_model_checkpoint(model_path, device=device, pretrained=False)

    transform = build_transforms(image_size, train=False)
    dataset = _FrameInferenceDataset(frame_paths, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    probs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            batch_probs = torch.sigmoid(logits).squeeze(1).cpu().tolist()
            probs.extend(batch_probs)

    if not probs:
        raise ValueError("No frames processed for inference.")

    mean_prob = float(sum(probs) / len(probs))
    prediction = "FAKE" if mean_prob >= threshold else "REAL"

    return {
        "prediction": prediction,
        "fake_probability": mean_prob,
        "threshold": float(threshold),
        "frame_count": len(probs),
    }


def predict_from_images(
    images: list[Image.Image],
    threshold: float = 0.5,
    model_path: Path | None = None,
    batch_size: int = 32,
) -> dict:
    if not images:
        raise ValueError("No images provided for inference.")

    model_path = model_path or CNN_MODEL_PATH
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, image_size = load_model_checkpoint(model_path, device=device, pretrained=False)

    transform = build_transforms(image_size, train=False)
    tensors = [transform(img.convert("RGB")) for img in images]

    probs: list[float] = []
    with torch.no_grad():
        for start in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[start : start + batch_size]).to(device)
            logits = model(batch)
            batch_probs = torch.sigmoid(logits).squeeze(1).cpu().tolist()
            probs.extend(batch_probs)

    mean_prob = float(sum(probs) / len(probs))
    prediction = "FAKE" if mean_prob >= threshold else "REAL"

    return {
        "prediction": prediction,
        "fake_probability": mean_prob,
        "threshold": float(threshold),
        "image_count": len(probs),
    }
