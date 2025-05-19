# 2_models/train_resnet.py

import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models
from dataset import MedDataset
from tqdm import tqdm

def train(train_csv, val_csv, img_dir, num_classes=2, epochs=10, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # === Load datasets ===
    train_ds = MedDataset(train_csv, img_dir)
    val_ds   = MedDataset(val_csv, img_dir)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=batch_size)

    # === Load ResNet-50 ===
    model = models.resnet50(weights="IMAGENET1K_V2")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # === Training loop ===
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_dl)
        acc = evaluate(model, val_dl, device)
        print(f"✅ Epoch {epoch+1}: Loss = {avg_loss:.4f}, Val Acc = {acc:.2f}%")

    # === Save weights ===
    os.makedirs("../2_models/weights", exist_ok=True)
    torch.save(model.state_dict(), "../2_models/weights/resnet50_med.pth")
    print("🎉 Model saved to weights/resnet50_med.pth")

def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return (correct / total) * 100

# === Run training ===
if __name__ == "__main__":
    train(
        train_csv="../1_data/train.csv",
        val_csv="../1_data/val.csv",
        img_dir="../1_data/images",
        num_classes=2,
        epochs=5,
        batch_size=32
    )
