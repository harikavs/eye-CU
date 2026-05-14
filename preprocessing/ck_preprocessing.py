import os
import csv
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

class CKDataset(Dataset):
    def __init__(self, csv_path, split='Training', transform=None):
        self.transform = transform
        self.images = []
        self.labels = []

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)  # skip header row
            for row in reader:
                if len(row) < 3:
                    continue
                label = int(row[0])
                pixels = np.array(row[1].split(), dtype=np.uint8).reshape(48, 48)
                usage = row[2].strip()
                if usage == split and label <= 6:
                    self.images.append(pixels)
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.fromarray(self.images[index])
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label


# Test it loads correctly
if __name__ == "__main__":
    csv_path = '../data/ck+/ckextended.csv'

    train_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = CKDataset(csv_path, split='Training', transform=train_transform)
    print(f"CK+ training samples: {len(dataset)}")
    print(f"Sample shape: {dataset[0][0].shape}")
    print(f"Sample label: {dataset[0][1]} ({EMOTIONS[dataset[0][1]]})")