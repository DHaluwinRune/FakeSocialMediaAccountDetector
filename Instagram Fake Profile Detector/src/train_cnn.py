import argparse
from datetime import datetime
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .cnn_data import FrameDataset, load_username_labels, split_usernames_for_training
from .cnn_model import build_model, build_transforms
from .config import CNN_MODEL_PATH, VIDEO_FRAMES_DIR


def _accuracy(logits, labels) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    return (preds.eq(labels).float().mean().item())


def _run_epoch(model, loader, criterion, optimizer=None, device=None) -> tuple[float, float]:
    device = device or torch.device("cpu")
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_acc = 0.0
    steps = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).view(-1, 1)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item())
        total_acc += _accuracy(logits.detach(), labels)
        steps += 1

    if steps == 0:
        return 0.0, 0.0

    return total_loss / steps, total_acc / steps


def _collect_probs_labels(model, loader, device=None) -> tuple[list[float], list[int]]:
    device = device or torch.device("cpu")
    model.eval()
    probs: list[float] = []
    labels: list[int] = []

    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device)
            batch_labels = batch_labels.to(device).view(-1, 1)
            logits = model(images)
            batch_probs = torch.sigmoid(logits).view(-1).cpu().tolist()
            probs.extend(batch_probs)
            labels.extend([int(v) for v in batch_labels.view(-1).cpu().tolist()])

    return probs, labels


def _report_metrics(
    probs: list[float],
    labels: list[int],
    threshold: float = 0.5,
) -> dict:
    if not probs or not labels:
        print("Validation metrics: no samples to score.")
        return {}

    preds = [1 if p >= threshold else 0 for p in probs]
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    print(
        "Validation metrics - "
        f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | F1: {f1:.4f}"
    )
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN on Instagram video frames")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--max-frames-per-video", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--output", default=str(CNN_MODEL_PATH))
    parser.add_argument("--frames-dir", default=str(VIDEO_FRAMES_DIR))
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    if not frames_dir.exists():
        raise SystemExit(f"Frames directory not found: {frames_dir}")

    labels_map = load_username_labels()
    train_users, val_users = split_usernames_for_training(
        labels_map, frames_dir=frames_dir, val_size=args.val_split
    )

    train_transform = build_transforms(args.image_size, train=True)
    val_transform = build_transforms(args.image_size, train=False)

    train_dataset = FrameDataset(
        frames_dir,
        labels_map,
        train_users,
        transform=train_transform,
        max_frames_per_video=args.max_frames_per_video,
    )
    val_dataset = FrameDataset(
        frames_dir,
        labels_map,
        val_users,
        transform=val_transform,
        max_frames_per_video=args.max_frames_per_video,
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise SystemExit("No frames found for training/validation. Run frame extraction first.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=not args.no_pretrained)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer=optimizer, device=device
        )
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, device=device)
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    probs, labels = _collect_probs_labels(model, val_loader, device=device)
    metrics = _report_metrics(probs, labels)

    output_path = Path(args.output)
    if metrics:
        log_path = output_path.with_suffix(".metrics.txt")
        timestamp = datetime.now().isoformat(timespec="seconds")
        metrics_line = (
            f"{timestamp} | samples={len(labels)} | "
            f"accuracy={metrics['accuracy']:.4f} | "
            f"precision={metrics['precision']:.4f} | "
            f"recall={metrics['recall']:.4f} | "
            f"f1={metrics['f1']:.4f}"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(metrics_line + "\n")
        print(f"Metrics logged -> {log_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "image_size": args.image_size},
        output_path,
    )
    print(f"Saved CNN model -> {output_path}")


if __name__ == "__main__":
    main()
