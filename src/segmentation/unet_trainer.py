# src/segmentation/unet_trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.config import DEVICE, UNET_MODEL_PATH
from src.segmentation.unet_architecture import UNet

def train_unet(model: UNet, train_loader: DataLoader, val_loader: DataLoader,
               epochs: int = 100, patience: int = 10, lr: float = 1e-4):
    """
    Entrena un modelo UNet para la segmentación.

    Args:
        model (UNet): El modelo UNet a entrenar.
        train_loader (DataLoader): DataLoader para el conjunto de entrenamiento.
        val_loader (DataLoader): DataLoader para el conjunto de validación.
        epochs (int): Número máximo de épocas para entrenar.
        patience (int): Número de épocas sin mejora en la pérdida de validación
                        antes de detener el entrenamiento (early stopping).
        lr (float): Tasa de aprendizaje inicial para el optimizador.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    trigger = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, masks in tqdm(train_loader, desc=f"Época {epoch+1}/{epochs}"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()

        val_loss /= len(val_loader)
        print(f"\n🔁 Época {epoch+1} - Pérdida de Entrenamiento: {total_loss / len(train_loader):.4f} - Pérdida de Validación: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            trigger = 0
            torch.save(model.state_dict(), UNET_MODEL_PATH)
            print("✅ Modelo guardado.")
        else:
            trigger += 1
            if trigger >= patience:
                print("⏹️ Early stopping")
                break