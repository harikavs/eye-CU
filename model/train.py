import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

sys.path.append('../preprocessing')
from combined_dataset import get_combined_train, get_combined_test
from model import FERModel

BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 0.001
VAL_SPLIT = 0.1         # 10% of training set used for validation
RESULTS_DIR = '../results'
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, 'fer_model_best.pth')
FINAL_PATH = os.path.join(RESULTS_DIR, 'fer_model.pth')


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nLoading datasets...")
    full_train = get_combined_train()
    test_dataset = get_combined_test()

    # Split training set into train / val
    val_size = int(len(full_train) * VAL_SPLIT)
    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"\n  Train  : {len(train_dataset)}")
    print(f"  Val    : {len(val_dataset)}")
    print(f"  Test   : {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=(device.type == 'cuda')
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=(device.type == 'cuda')
    )

    model = FERModel().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # Cosine annealing gives smoother decay than step LR
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_val_acc = 0.0
    history = []

    print(f"\nTraining for {EPOCHS} epochs...\n")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        # --- Validate ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        scheduler.step()

        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        elapsed = time.time() - t0

        history.append({
            'epoch': epoch,
            'train_loss': train_loss / len(train_loader),
            'train_acc': train_acc,
            'val_loss': val_loss / len(val_loader),
            'val_acc': val_acc,
        })

        lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"train_loss={train_loss/len(train_loader):.4f} train_acc={train_acc:.2f}% | "
            f"val_loss={val_loss/len(val_loader):.4f} val_acc={val_acc:.2f}% | "
            f"lr={lr:.6f} | {elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  --> New best val acc: {best_val_acc:.2f}% — checkpoint saved")

    # --- Final evaluation on test set (best model) ---
    print(f"\nLoading best checkpoint (val_acc={best_val_acc:.2f}%)...")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    test_correct, test_total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_acc = 100. * test_correct / test_total
    print(f"Test accuracy: {test_acc:.2f}%")

    # Also save as the default model path used by evaluate.py
    torch.save(model.state_dict(), FINAL_PATH)
    print(f"Model saved to {FINAL_PATH}")

    # Save training history
    history_path = os.path.join(RESULTS_DIR, 'training_history.txt')
    with open(history_path, 'w') as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
        for h in history:
            f.write(f"{h['epoch']},{h['train_loss']:.4f},{h['train_acc']:.2f},{h['val_loss']:.4f},{h['val_acc']:.2f}\n")
    print(f"Training history saved to {history_path}")


if __name__ == '__main__':
    main()
