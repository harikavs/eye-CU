import cv2
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from opencv_preprocessing import preprocess_image

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def process_dataset(input_dir, output_dir):
    total = 0
    failed = 0
    fallback = 0

    for emotion in EMOTIONS:
        input_emotion_dir = os.path.join(input_dir, emotion)
        output_emotion_dir = os.path.join(output_dir, emotion)
        os.makedirs(output_emotion_dir, exist_ok=True)

        if not os.path.exists(input_emotion_dir):
            print(f"Skipping {emotion} — folder not found")
            continue

        files = os.listdir(input_emotion_dir)
        print(f"Processing {emotion}: {len(files)} images...")

        for filename in files:
            input_path = os.path.join(input_emotion_dir, filename)
            output_path = os.path.join(output_emotion_dir, filename)

            result = preprocess_image(input_path)
            if result is not None:
                cv2.imwrite(output_path, result)
                total += 1
            else:
                failed += 1

    print(f"\nDone. Processed: {total}, Failed: {failed}, Fallback: {fallback}")

if __name__ == "__main__":
    print("Processing FER2013 training set...")
    process_dataset(
        '../data/fer2013/train',
        '../data/fer2013_processed/train'
    )

    print("\nProcessing FER2013 test set...")
    process_dataset(
        '../data/fer2013/test',
        '../data/fer2013_processed/test'
    )

    print("\nAll done! Processed dataset saved to data/fer2013_processed/")