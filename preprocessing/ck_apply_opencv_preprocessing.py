import csv
import os
import sys
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from opencv_preprocessing import detector, predictor, get_eye_centers, align_face, crop_face

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
CK_CSV = '../data/ck+/ckextended.csv'
OUTPUT_ROOT = '../data/ck+_processed'

UPSAMPLE_SIZE = 192  # upsample 48->192 so dlib face detector works reliably


def process_pixels(pixel_str, split_name, emotion_idx, idx):
    pixels = np.array(pixel_str.split(), dtype=np.uint8).reshape(48, 48)

    # Upsample for face detection
    big = cv2.resize(pixels, (UPSAMPLE_SIZE, UPSAMPLE_SIZE), interpolation=cv2.INTER_CUBIC)

    faces = detector(big)
    if len(faces) > 0:
        face = faces[0]
        landmarks = predictor(big, face)
        left_center, right_center = get_eye_centers(landmarks)
        aligned = align_face(big, left_center, right_center)

        faces2 = detector(aligned)
        face2 = faces2[0] if len(faces2) > 0 else None
        if face2 is not None:
            cropped = crop_face(aligned, face2, padding=10)
        else:
            cropped = aligned
        result = cv2.resize(cropped, (48, 48))
        used_alignment = True
    else:
        # No face detected — just use the original pixels
        result = pixels
        used_alignment = False

    emotion_name = EMOTIONS[emotion_idx]
    out_dir = os.path.join(OUTPUT_ROOT, split_name, emotion_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{split_name}_{emotion_name}_{idx:05d}.png')
    cv2.imwrite(out_path, result)
    return used_alignment


def main():
    split_map = {
        'Training': 'train',
        'PublicTest': 'test',
        'PrivateTest': 'test',
    }
    counters = {'total': 0, 'aligned': 0, 'fallback': 0, 'skipped': 0}

    with open(CK_CSV, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) < 3:
                continue
            emotion_idx = int(row[0])
            pixel_str = row[1]
            usage = row[2].strip()

            if emotion_idx > 6:
                counters['skipped'] += 1
                continue
            if usage not in split_map:
                counters['skipped'] += 1
                continue

            split_name = split_map[usage]
            idx = counters['total']
            aligned = process_pixels(pixel_str, split_name, emotion_idx, idx)

            counters['total'] += 1
            if aligned:
                counters['aligned'] += 1
            else:
                counters['fallback'] += 1

            if counters['total'] % 100 == 0:
                print(f"  {counters['total']} processed (aligned: {counters['aligned']}, fallback: {counters['fallback']})")

    print(f"\nDone.")
    print(f"  Total saved : {counters['total']}")
    print(f"  Aligned     : {counters['aligned']}")
    print(f"  Fallback    : {counters['fallback']}")
    print(f"  Skipped     : {counters['skipped']}")
    print(f"  Output      : {os.path.abspath(OUTPUT_ROOT)}")


if __name__ == '__main__':
    main()
