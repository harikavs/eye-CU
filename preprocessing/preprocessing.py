import cv2
import os
import numpy as np 
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image 

# Paths
DATA_PATH = "../data/fer2013"
TRAIN_PATH = os.path.join(DATA_PATH, "train")
TEST_PATH = os.path.join(DATA_PATH, "test")

# Emotion labels 
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Basic transform
transform = transforms.Compose([
    transforms.Grayscale(), 
    transforms.Resize((48, 48)),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class FERDataset(Dataset): 
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform 
        self.images = []
        self.labels = []
        
        for label, emotion in enumerate(EMOTIONS): 
            emotion_path = os.path.join(root_dir, emotion)
            for img_file in os.listdir(emotion_path):
                self.images.append(os.path.join(emotion_path, img_file))
                self.labels.append(label)
                
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        label = self.labels[index]
        if self.transform: 
            image =  self.transform(image)
        return image, label
    
# Test it loads correctly 
if __name__ == "__main__":
    print("Script started")
    print(f"Train path: {TRAIN_PATH}")
    print(f"Path exists: {os.path.exists(TRAIN_PATH)}")
    
    try:
        dataset = FERDataset(TRAIN_PATH, transform=transform)
        print(f"Dataset size: {len(dataset)}")
        print(f"Sample shape: {dataset[0][0].shape}")
    except Exception as e:
        print(f"Error: {e}")