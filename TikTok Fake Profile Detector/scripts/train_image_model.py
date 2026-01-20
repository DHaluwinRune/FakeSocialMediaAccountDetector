#!/usr/bin/env python
import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a CNN on TikTok profile images (real vs fake)."
    )
    parser.add_argument(
        "--data-dir",
        default="data/dataset",
        help="Directory containing class folders (e.g. real/ and fake/).",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory to write model artifacts.",
    )
    parser.add_argument("--image-size", type=int, default=224, help="Image size (square).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=12, help="Number of epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=0, help="Data loader workers.")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation.")
    parser.add_argument(
        "--device",
        default=None,
        help="Force device (cpu or cuda). Defaults to auto-detect.",
    )
    return parser.parse_args()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(image_size, augment):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if augment:
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )
    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, val_transform


def compute_class_weights(targets, num_classes):
    counts = torch.bincount(torch.tensor(targets), minlength=num_classes).float()
    if torch.any(counts == 0):
        return None
    weights = counts.sum() / (counts * num_classes)
    return weights


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy, y_true, y_pred


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}", file=sys.stderr)
        return 1

    seed_everything(args.random_state)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_transform, val_transform = build_transforms(
        args.image_size, augment=not args.no_augment
    )

    full_dataset = datasets.ImageFolder(root=str(data_dir), transform=train_transform)
    if len(full_dataset) == 0:
        print(f"No images found under: {data_dir}", file=sys.stderr)
        return 1

    num_classes = len(full_dataset.classes)
    if num_classes < 2:
        print(
            f"Expected at least 2 classes, found {num_classes}: {full_dataset.classes}",
            file=sys.stderr,
        )
        return 1

    val_size = int(len(full_dataset) * args.val_split)
    val_size = max(1, val_size)
    train_size = len(full_dataset) - val_size
    if train_size < 1:
        print("Validation split too large for dataset size.", file=sys.stderr)
        return 1

    generator = torch.Generator().manual_seed(args.random_state)
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )
    val_subset.dataset.transform = val_transform

    train_targets = [full_dataset.targets[i] for i in train_subset.indices]
    class_weights = compute_class_weights(train_targets, num_classes)

    pin_memory = device == "cuda"
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_state = None
    best_val_acc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"- train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f} "
            f"- val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
        )

    if best_state is None:
        print("Training failed to produce a model.", file=sys.stderr)
        return 1

    model.load_state_dict(best_state)
    val_loss, val_acc, y_true, y_pred = evaluate(
        model, val_loader, criterion, device
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=full_dataset.classes,
        output_dict=True,
        zero_division=0,
    )
    conf_matrix = confusion_matrix(y_true, y_pred).tolist()

    class_counts = {name: 0 for name in full_dataset.classes}
    for _, label in full_dataset.samples:
        class_counts[full_dataset.classes[label]] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "image_model.pth"
    meta_path = output_dir / "image_model_meta.json"
    metrics_path = output_dir / "image_model_metrics.json"

    torch.save(best_state, model_path)

    meta = {
        "architecture": "simple_cnn",
        "image_size": args.image_size,
        "classes": full_dataset.classes,
        "class_to_idx": full_dataset.class_to_idx,
        "counts_total": class_counts,
        "train_size": train_size,
        "val_size": val_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "val_split": args.val_split,
        "random_state": args.random_state,
        "best_val_accuracy": best_val_acc,
        "final_val_accuracy": val_acc,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    metrics = {
        "history": history,
        "classification_report": report,
        "confusion_matrix": conf_matrix,
    }

    write_json(meta_path, meta)
    write_json(metrics_path, metrics)

    print(f"Image model saved to: {model_path}")
    print(f"Meta saved to: {meta_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
