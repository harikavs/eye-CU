import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import FERModel

TEST_DIR = '../data/fer2013/test'
MODEL_PATH = '../results/fer_model.pth'
BATCH_SIZE = 64
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device('cpu')
model = FERModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

print("\n=== Classification Report ===")
print(classification_report(all_labels, all_preds, target_names=EMOTIONS))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(all_labels, all_preds))

overall_acc = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"\nOverall Test Accuracy: {overall_acc*100:.2f}%")