"""
Real-time cognitive strain detector.

Combines:
  - Facial expression recognition (FERModel CNN)
  - Eye-tracking metrics (blink rate, fixation stability, pupil area proxy)

Uses dlib 68-landmark face detection.  A MyGaze SDK hook is provided
at the bottom — swap in the real API when the hardware is connected.
"""

import sys
import os
import time
import collections
import cv2
import dlib
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from mygaze_hook import MyGazeHook

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))
from model import FERModel

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH = '../results/fer_model.pth'
PREDICTOR_PATH = '../models/shape_predictor_68_face_landmarks.dat'

# ── Labels ───────────────────────────────────────────────────────────────────
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Emotions that indicate cognitive strain — used to compute a strain score
STRAIN_EMOTIONS = {'Fear', 'Sad', 'Angry', 'Disgust'}

# ── Thresholds ────────────────────────────────────────────────────────────────
EAR_BLINK_THRESHOLD = 0.21   # Eye Aspect Ratio below this = blink
EAR_CONSEC_FRAMES = 2        # frames below threshold to count as a blink
BLINK_WINDOW_SECS = 30       # rolling window for blink-rate measurement
HIGH_BLINK_RATE = 25         # blinks/min above this = strain indicator
GAZE_INSTABILITY_THRESHOLD = 15  # px jitter over window = strain indicator
STRAIN_EMOTION_RATIO = 0.4   # fraction of recent frames with strain emotion

# ── CNN transform ─────────────────────────────────────────────────────────────
cnn_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


def _ear(eye_pts):
    """Eye Aspect Ratio — scalar distance ratio for blink detection."""
    A = np.linalg.norm(eye_pts[1] - eye_pts[5])
    B = np.linalg.norm(eye_pts[2] - eye_pts[4])
    C = np.linalg.norm(eye_pts[0] - eye_pts[3])
    return (A + B) / (2.0 * C)


def _eye_pts(landmarks, start, end):
    return np.array([(landmarks.part(i).x, landmarks.part(i).y)
                     for i in range(start, end)], dtype=np.float64)


def _gaze_center(landmarks):
    left = _eye_pts(landmarks, 36, 42).mean(axis=0)
    right = _eye_pts(landmarks, 42, 48).mean(axis=0)
    return (left + right) / 2.0


class CognitiveStrainDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = FERModel().to(self.device)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.eval()

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(PREDICTOR_PATH)

        # Rolling state
        self.blink_consec = 0
        self.blink_times = collections.deque()         # timestamps of blinks
        self.gaze_history = collections.deque(maxlen=30)
        self.emotion_history = collections.deque(maxlen=60)

    # ── Inference helpers ──────────────────────────────────────────────────────

    def _predict_emotion(self, face_roi_gray):
        img = Image.fromarray(face_roi_gray)
        tensor = cnn_transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return EMOTIONS[probs.argmax()], probs

    # ── Per-frame processing ───────────────────────────────────────────────────

    def process_frame(self, frame):
        """
        Process one BGR frame.  Returns annotated frame + metrics dict.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)

        metrics = {
            'emotion': None,
            'emotion_probs': None,
            'ear': None,
            'blink_rate': self._blink_rate(),
            'gaze_jitter': self._gaze_jitter(),
            'strain_score': 0.0,
            'strain_label': 'Unknown',
        }

        for face in faces:
            landmarks = self.predictor(gray, face)

            # ── Eye tracking ──────────────────────────────────────────────────
            left_pts = _eye_pts(landmarks, 36, 42)
            right_pts = _eye_pts(landmarks, 42, 48)
            ear = (_ear(left_pts) + _ear(right_pts)) / 2.0
            metrics['ear'] = ear

            # Blink detection
            if ear < EAR_BLINK_THRESHOLD:
                self.blink_consec += 1
            else:
                if self.blink_consec >= EAR_CONSEC_FRAMES:
                    self.blink_times.append(time.time())
                self.blink_consec = 0

            # Prune old blink timestamps
            cutoff = time.time() - BLINK_WINDOW_SECS
            while self.blink_times and self.blink_times[0] < cutoff:
                self.blink_times.popleft()

            # Gaze center
            gc = _gaze_center(landmarks)
            self.gaze_history.append(gc)

            # ── Emotion CNN ───────────────────────────────────────────────────
            x1, y1 = max(0, face.left()), max(0, face.top())
            x2, y2 = min(frame.shape[1], face.right()), min(frame.shape[0], face.bottom())
            face_roi = gray[y1:y2, x1:x2]
            if face_roi.size > 0:
                emotion, probs = self._predict_emotion(face_roi)
                metrics['emotion'] = emotion
                metrics['emotion_probs'] = probs
                self.emotion_history.append(emotion)

            # Only process first face
            break

        # ── Strain score ──────────────────────────────────────────────────────
        metrics['blink_rate'] = self._blink_rate()
        metrics['gaze_jitter'] = self._gaze_jitter()
        metrics['strain_score'], metrics['strain_label'] = self._strain_score(metrics)

        # ── Annotate frame ────────────────────────────────────────────────────
        self._draw(frame, metrics, faces)
        return frame, metrics

    # ── Strain scoring ─────────────────────────────────────────────────────────

    def _blink_rate(self):
        """Blinks per minute over the last BLINK_WINDOW_SECS seconds."""
        return len(self.blink_times) * (60.0 / BLINK_WINDOW_SECS)

    def _gaze_jitter(self):
        if len(self.gaze_history) < 5:
            return 0.0
        pts = np.array(self.gaze_history)
        return float(np.std(pts, axis=0).mean())

    def _emotion_strain_ratio(self):
        if not self.emotion_history:
            return 0.0
        n = len(self.emotion_history)
        strain_count = sum(1 for e in self.emotion_history if e in STRAIN_EMOTIONS)
        return strain_count / n

    def _strain_score(self, metrics):
        score = 0.0
        reasons = []

        br = metrics['blink_rate']
        if br > HIGH_BLINK_RATE:
            score += 0.33
            reasons.append('high blink rate')
        elif br < 8:                      # very low blink rate = focused/strained
            score += 0.2
            reasons.append('low blink rate')

        if metrics['gaze_jitter'] > GAZE_INSTABILITY_THRESHOLD:
            score += 0.33
            reasons.append('gaze instability')

        er = self._emotion_strain_ratio()
        if er > STRAIN_EMOTION_RATIO:
            score += 0.34
            reasons.append('strain emotion')

        score = min(score, 1.0)

        if score >= 0.66:
            label = 'HIGH STRAIN'
        elif score >= 0.33:
            label = 'MODERATE STRAIN'
        else:
            label = 'LOW STRAIN'

        return round(score, 2), label

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw(self, frame, metrics, faces):
        h, w = frame.shape[:2]

        for face in faces:
            cv2.rectangle(frame,
                          (face.left(), face.top()),
                          (face.right(), face.bottom()),
                          (0, 255, 0), 2)

        y = 30
        def put(text, color=(255, 255, 255)):
            nonlocal y
            cv2.putText(frame, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 25

        emotion = metrics['emotion'] or '—'
        put(f"Emotion : {emotion}")

        if metrics['ear'] is not None:
            put(f"EAR     : {metrics['ear']:.3f}")

        put(f"Blinks/m: {metrics['blink_rate']:.1f}")
        put(f"Gaze jit: {metrics['gaze_jitter']:.1f}px")

        strain_color = {
            'HIGH STRAIN': (0, 0, 255),
            'MODERATE STRAIN': (0, 165, 255),
            'LOW STRAIN': (0, 200, 0),
        }.get(metrics['strain_label'], (200, 200, 200))

        put(f"Strain  : {metrics['strain_label']} ({metrics['strain_score']:.2f})", strain_color)

def main():
    detector_system = CognitiveStrainDetector()
    mygaze = MyGazeHook()
    mygaze.connect()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Running — press Q to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, metrics = detector_system.process_frame(frame)

        # Optionally overlay MyGaze gaze point
        gaze = mygaze.get_gaze_point()
        if gaze:
            cv2.circle(annotated, (int(gaze[0]), int(gaze[1])), 8, (0, 255, 255), -1)

        cv2.imshow('Cognitive Strain Detector', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    mygaze.disconnect()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
