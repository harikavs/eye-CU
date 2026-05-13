from torch.utils.data import ConcatDataset
import torchvision.transforms as transforms
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import FERDataset
from ck_preprocessing import CKDataset

CK_CSV = '../data/ck+/ckextended.csv'
FER_TRAIN = '../data/fer2013/train'
FER_TEST = '../data/fer2013/test'

train_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def get_combined_train():
    fer_train = FERDataset(FER_TRAIN, transform=train_transform)
    ck_train = CKDataset(CK_CSV, split='Training', transform=train_transform)
    combined = ConcatDataset([fer_train, ck_train])
    print(f"FER2013 training samples: {len(fer_train)}")
    print(f"CK+ training samples: {len(ck_train)}")
    print(f"Combined total: {len(combined)}")
    return combined

def get_combined_test():
    fer_test = FERDataset(FER_TEST, transform=test_transform)
    ck_test = CKDataset(CK_CSV, split='PublicTest', transform=test_transform)
    combined = ConcatDataset([fer_test, ck_test])
    print(f"FER2013 test samples: {len(fer_test)}")
    print(f"CK+ test samples: {len(ck_test)}")
    print(f"Combined total: {len(combined)}")
    return combined

if __name__ == "__main__":
    print("=== Training set ===")
    train = get_combined_train()
    print("\n=== Test set ===")
    test = get_combined_test()