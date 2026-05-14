import cv2
import dlib
import numpy as np
from PIL import Image
import os

# Load dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../models/shape_predictor_68_face_landmarks.dat')

def get_landmarks(image_gray):
    faces = detector(image_gray)
    if len(faces) == 0:
        return None, None
    face = faces[0]
    landmarks = predictor(image_gray, face)
    return face, landmarks

def get_eye_centers(landmarks):
    # Left eye: points 36-41, Right eye: points 42-47
    left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
    right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
    left_center = left_eye.mean(axis=0)
    right_center = right_eye.mean(axis=0)
    return left_center, right_center

def align_face(image, left_center, right_center):
    # Calculate angle between eyes and horizontal
    dx = right_center[0] - left_center[0]
    dy = right_center[1] - left_center[1]
    angle = np.degrees(np.arctan2(dy, dx))
    
    # Rotate image to align face
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv2.INTER_CUBIC)
    return aligned

def crop_face(image, face, padding=20):
    x1 = max(0, face.left() - padding)
    y1 = max(0, face.top() - padding)
    x2 = min(image.shape[1], face.right() + padding)
    y2 = min(image.shape[0], face.bottom() + padding)
    return image[y1:y2, x1:x2]

def preprocess_image(image_path, output_size=(48, 48)):
    # Load image as grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    
    # Detect face and landmarks
    face, landmarks = get_landmarks(image)
    if landmarks is None:
        # No face detected — fall back to simple resize
        return cv2.resize(image, output_size)
    
    # Get eye centers and align
    left_center, right_center = get_eye_centers(landmarks)
    aligned = align_face(image, left_center, right_center)
    
    # Re-detect face in aligned image for accurate crop
    face2, _ = get_landmarks(aligned)
    if face2 is not None:
        cropped = crop_face(aligned, face2)
    else:
        cropped = aligned
    
    # Resize to output size
    resized = cv2.resize(cropped, output_size)
    return resized

if __name__ == "__main__":
    # Test on a single image
    import sys
    TEST_DIR = '../data/fer2013/train/happy'
    test_image = os.path.join(TEST_DIR, os.listdir(TEST_DIR)[0])
    print(f"Testing on: {test_image}")
    
    result = preprocess_image(test_image)
    if result is not None:
        print(f"Output shape: {result.shape}")
        print("OpenCV preprocessing working correctly")
    else:
        print("Failed to process image")