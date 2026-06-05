"""
Real-time cognitive strain detector.

Combines:
  - Facial expression recognition (FERModel CNN)
  - Eye-tracking metrics (blink rate, fixation stability)

Uses dlib 68-landmark face detection.
"""

import csv
import datetime
import sys
import os
import time
import collections
import cv2
import dlib
import mediapipe as mp
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

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
CALIBRATION_SECS = 20        # duration of startup blink-rate calibration

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

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True, max_num_faces=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )

        # Calibration baseline (overwritten by calibrate())
        self.baseline_blink_rate = HIGH_BLINK_RATE

        # Rolling state
        self.blink_consec = 0
        self.blink_times = collections.deque()         # timestamps of blinks
        self.gaze_history = collections.deque(maxlen=30)
        self.emotion_history = collections.deque(maxlen=60)
        self.iris_diameter_history = collections.deque(maxlen=60)

        # Session CSV log
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        session_ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_path = os.path.join(log_dir, f'session_{session_ts}.csv')
        self._csv_file = open(log_path, 'w', newline='', buffering=1)
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow([
            'timestamp', 'strain_score', 'strain_label',
            'emotion', 'blink_rate', 'ear',
            'gaze_jitter', 'gaze_direction', 'iris_diameter',
        ])
        print(f"Logging session to {log_path}")

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
            'gaze_direction': None,
            'iris_diameter': None,
        }

        dlib_gaze = None

        for face in faces:
            landmarks = self.predictor(gray, face)

            # ── Eye tracking (EAR / blink) ────────────────────────────────────
            left_pts = _eye_pts(landmarks, 36, 42)
            right_pts = _eye_pts(landmarks, 42, 48)
            ear = (_ear(left_pts) + _ear(right_pts)) / 2.0
            metrics['ear'] = ear

            if ear < EAR_BLINK_THRESHOLD:
                self.blink_consec += 1
            else:
                if self.blink_consec >= EAR_CONSEC_FRAMES:
                    self.blink_times.append(time.time())
                self.blink_consec = 0

            cutoff = time.time() - BLINK_WINDOW_SECS
            while self.blink_times and self.blink_times[0] < cutoff:
                self.blink_times.popleft()

            dlib_gaze = _gaze_center(landmarks)

            # ── Emotion CNN ───────────────────────────────────────────────────
            x1, y1 = max(0, face.left()), max(0, face.top())
            x2, y2 = min(frame.shape[1], face.right()), min(frame.shape[0], face.bottom())
            face_roi = gray[y1:y2, x1:x2]
            if face_roi.size > 0:
                emotion, probs = self._predict_emotion(face_roi)
                metrics['emotion'] = emotion
                metrics['emotion_probs'] = probs
                self.emotion_history.append(emotion)

            break

        # ── MediaPipe Iris ────────────────────────────────────────────────────
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_result = self.face_mesh.process(rgb)

        if mp_result.multi_face_landmarks:
            lm = mp_result.multi_face_landmarks[0].landmark

            # Iris centers in pixels
            left_iris_px = np.array([lm[468].x * w, lm[468].y * h])
            right_iris_px = np.array([lm[473].x * w, lm[473].y * h])
            self.gaze_history.append((left_iris_px + right_iris_px) / 2.0)

            # Iris diameter: horizontal span of the 4 edge points per eye
            left_edge = np.array([(lm[i].x * w, lm[i].y * h) for i in range(469, 473)])
            right_edge = np.array([(lm[i].x * w, lm[i].y * h) for i in range(474, 478)])
            left_diam = float(np.max(left_edge[:, 0]) - np.min(left_edge[:, 0]))
            right_diam = float(np.max(right_edge[:, 0]) - np.min(right_edge[:, 0]))
            iris_diameter = (left_diam + right_diam) / 2.0
            self.iris_diameter_history.append(iris_diameter)
            metrics['iris_diameter'] = iris_diameter

            # Gaze direction: iris position ratio within each eye
            left_ratio = (lm[468].x - lm[33].x) / max(lm[133].x - lm[33].x, 1e-6)
            right_ratio = (lm[473].x - lm[362].x) / max(lm[263].x - lm[362].x, 1e-6)
            avg_ratio = (left_ratio + right_ratio) / 2.0
            if avg_ratio < 0.42:
                metrics['gaze_direction'] = 'left'
            elif avg_ratio > 0.58:
                metrics['gaze_direction'] = 'right'
            else:
                metrics['gaze_direction'] = 'center'
        elif dlib_gaze is not None:
            self.gaze_history.append(dlib_gaze)

        # ── Strain score ──────────────────────────────────────────────────────
        metrics['blink_rate'] = self._blink_rate()
        metrics['gaze_jitter'] = self._gaze_jitter()
        metrics['strain_score'], metrics['strain_label'] = self._strain_score(metrics)

        # ── Log to CSV ────────────────────────────────────────────────────────
        self._csv.writerow([
            round(time.time(), 3),
            metrics['strain_score'],
            metrics['strain_label'],
            metrics['emotion'] or '',
            round(metrics['blink_rate'], 2),
            round(metrics['ear'], 3) if metrics['ear'] is not None else '',
            round(metrics['gaze_jitter'], 2),
            metrics['gaze_direction'] or '',
            round(metrics['iris_diameter'], 2) if metrics['iris_diameter'] is not None else '',
        ])

        # ── Annotate frame ────────────────────────────────────────────────────
        self._draw(frame, metrics, faces)
        return frame, metrics

    # ── Calibration ───────────────────────────────────────────────────────────

    def calibrate(self, cap, duration=CALIBRATION_SECS):
        """Run a timed blink-collection phase to set a personal blink-rate baseline."""
        print(f"Calibrating for {duration}s — look naturally at the screen...")
        end_time = time.time() + duration
        cal_blinks = 0
        cal_consec = 0

        while time.time() < end_time:
            ret, frame = cap.read()
            if not ret:
                break

            remaining = max(0, int(end_time - time.time()) + 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            for face in self.detector(gray, 0):
                landmarks = self.predictor(gray, face)
                left_pts = _eye_pts(landmarks, 36, 42)
                right_pts = _eye_pts(landmarks, 42, 48)
                ear = (_ear(left_pts) + _ear(right_pts)) / 2.0
                if ear < EAR_BLINK_THRESHOLD:
                    cal_consec += 1
                else:
                    if cal_consec >= EAR_CONSEC_FRAMES:
                        cal_blinks += 1
                    cal_consec = 0
                break

            # Overlay
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 75), (0, 0, 0), -1)
            cv2.putText(frame, 'CALIBRATING — look naturally at the screen',
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(frame, f'Time remaining: {remaining}s    Blinks detected: {cal_blinks}',
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.imshow('Cognitive Strain Detector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.baseline_blink_rate = max(cal_blinks * (60.0 / duration), 8.0)
        print(f"Calibration done. Baseline: {self.baseline_blink_rate:.1f} blinks/min")
        return self.baseline_blink_rate

    def close(self):
        self._csv_file.close()
        self.face_mesh.close()

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
        high_thresh = max(self.baseline_blink_rate * 1.5, self.baseline_blink_rate + 8)
        low_thresh = max(self.baseline_blink_rate * 0.5, 4.0)
        if br > high_thresh:
            score += 0.33
            reasons.append('high blink rate')
        elif br < low_thresh:
            score += 0.2
            reasons.append('low blink rate')

        if metrics['gaze_jitter'] > GAZE_INSTABILITY_THRESHOLD:
            score += 0.33
            reasons.append('gaze instability')

        er = self._emotion_strain_ratio()
        if er > STRAIN_EMOTION_RATIO:
            score += 0.34
            reasons.append('strain emotion')

        if (metrics.get('iris_diameter') is not None
                and len(self.iris_diameter_history) >= 10):
            rolling_mean = np.mean(self.iris_diameter_history)
            if rolling_mean > 0 and metrics['iris_diameter'] > 1.15 * rolling_mean:
                score += 0.2
                reasons.append('iris dilation')

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
        put(f"Gaze    : {metrics['gaze_direction'] or '—'}")

        iris_d = metrics['iris_diameter']
        if iris_d is not None:
            put(f"Iris dia: {iris_d:.1f}px")

        strain_color = {
            'HIGH STRAIN': (0, 0, 255),
            'MODERATE STRAIN': (0, 165, 255),
            'LOW STRAIN': (0, 200, 0),
        }.get(metrics['strain_label'], (200, 200, 200))

        put(f"Strain  : {metrics['strain_label']} ({metrics['strain_score']:.2f})", strain_color)


def main():
    detector_system = CognitiveStrainDetector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam")
        detector_system.close()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        detector_system.calibrate(cap)

        print("Running — press Q to quit")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated, _ = detector_system.process_frame(frame)

            cv2.imshow('Cognitive Strain Detector', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector_system.close()


if __name__ == '__main__':
    main()