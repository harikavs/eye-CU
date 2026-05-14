import os
import csv
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


class CKDataset(Dataset):
    """Load CK+ from the raw CSV (pixel values)."""
    def __init__(self, csv_path, split='Training', transform=None):
        self.transform = transform
        self.images = []
        self.labels = []

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header row
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


class CKDatasetFromImages(Dataset):
    """Load preprocessed CK+ from an image-folder tree: root/<emotion>/*.png"""
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []

        for label, emotion in enumerate(EMOTIONS):
            emotion_dir = os.path.join(root_dir, emotion)
            if not os.path.isdir(emotion_dir):
                continue
            for fname in os.listdir(emotion_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(emotion_dir, fname))
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == "__main__":
    train_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    csv_path = '../data/ck+/ckextended.csv'
    dataset = CKDataset(csv_path, split='Training', transform=train_transform)
    print(f"CK+ (CSV) training samples: {len(dataset)}")

    processed_dir = '../data/ck+_processed/train'
    if os.path.exists(processed_dir):
        dataset2 = CKDatasetFromImages(processed_dir, transform=train_transform)
        print(f"CK+ (processed images) training samples: {len(dataset2)}")
    else:
        print("Processed CK+ not found — run ck_apply_opencv_preprocessing.py first")