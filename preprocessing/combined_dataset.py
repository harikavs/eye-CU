import os
import sys
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import FERDataset
from ck_preprocessing import CKDataset, CKDatasetFromImages

# Prefer preprocessed folders; fall back to originals if not yet generated
FER_PROCESSED_TRAIN = '../data/fer2013_processed/train'
FER_PROCESSED_TEST = '../data/fer2013_processed/test'
FER_RAW_TRAIN = '../data/fer2013/train'
FER_RAW_TEST = '../data/fer2013/test'

CK_PROCESSED_TRAIN = '../data/ck+_processed/train'
CK_PROCESSED_TEST = '../data/ck+_processed/test'
CK_CSV = '../data/ck+/ckextended.csv'

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


def _fer_train():
    if os.path.isdir(FER_PROCESSED_TRAIN):
        print(f"  FER2013: using preprocessed data at {FER_PROCESSED_TRAIN}")
        return FERDataset(FER_PROCESSED_TRAIN, transform=train_transform)
    print(f"  FER2013: preprocessed not found, using raw data")
    return FERDataset(FER_RAW_TRAIN, transform=train_transform)


def _fer_test():
    if os.path.isdir(FER_PROCESSED_TEST):
        return FERDataset(FER_PROCESSED_TEST, transform=test_transform)
    return FERDataset(FER_RAW_TEST, transform=test_transform)


def _ck_train():
    if os.path.isdir(CK_PROCESSED_TRAIN):
        print(f"  CK+: using preprocessed images at {CK_PROCESSED_TRAIN}")
        return CKDatasetFromImages(CK_PROCESSED_TRAIN, transform=train_transform)
    print(f"  CK+: preprocessed not found, using raw CSV — run ck_apply_opencv_preprocessing.py to preprocess")
    return CKDataset(CK_CSV, split='Training', transform=train_transform)


def _ck_test():
    if os.path.isdir(CK_PROCESSED_TEST):
        return CKDatasetFromImages(CK_PROCESSED_TEST, transform=test_transform)
    return CKDataset(CK_CSV, split='PublicTest', transform=test_transform)


def get_combined_train():
    fer = _fer_train()
    ck = _ck_train()
    combined = ConcatDataset([fer, ck])
    print(f"  FER2013 train: {len(fer)}")
    print(f"  CK+ train    : {len(ck)}")
    print(f"  Combined     : {len(combined)}")
    return combined


def get_combined_test():
    fer = _fer_test()
    ck = _ck_test()
    combined = ConcatDataset([fer, ck])
    print(f"  FER2013 test : {len(fer)}")
    print(f"  CK+ test     : {len(ck)}")
    print(f"  Combined     : {len(combined)}")
    return combined


if __name__ == "__main__":
    print("=== Training set ===")
    train = get_combined_train()
    print("\n=== Test set ===")
    test = get_combined_test()
