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
EAR_BLINK_THRESHOLD = 0.21   # fallback only — overridden by personal calibration
EAR_BLINK_RATIO    = 0.72    # blink threshold = open-eye EAR × this ratio
EAR_PERCLOS_RATIO  = 0.80    # PERCLOS closure threshold = open-eye EAR × this ratio
EAR_CONSEC_FRAMES  = 1       # frames below threshold to register a blink
PERCLOS_WINDOW     = 900     # rolling frame window for PERCLOS (~30 s at 30 fps)
PERCLOS_THRESHOLD  = 0.15    # PERCLOS fraction above this = strain indicator
BLINK_WINDOW_SECS = 30       # rolling window for blink-rate measurement
HIGH_BLINK_RATE = 25         # blinks/min above this = strain indicator
GAZE_INSTABILITY_THRESHOLD = 15  # px jitter over window = strain indicator
STRAIN_EMOTION_RATIO = 0.4   # fraction of recent frames with strain emotion
CALIBRATION_SECS  = 20       # total calibration duration
BASELINE_SECS     = 3        # first N seconds: eyes-open EAR measurement

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


def _wait_for_key(cap, title, subtitle='', key=32):
    """Overlay a prompt on the live feed and block until SPACE (or Q to abort).
    Returns True when the user continues, False if they quit."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ret, frame = cap.read()
        if not ret:
            return False
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h // 2 - 80), (w, h // 2 + 85), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        for text, y, scale, color, thick in [
            (title,                         h // 2 - 30, 0.75, (255, 255, 255), 2),
            (subtitle,                      h // 2 + 12, 0.55, (180, 180, 180), 1),
            ('SPACE to continue  |  Q to quit', h // 2 + 58, 0.48, (100, 210, 100), 1),
        ]:
            if text:
                (tw, _), _ = cv2.getTextSize(text, font, scale, thick)
                cv2.putText(frame, text, ((w - tw) // 2, y), font, scale, color, thick)
        cv2.imshow('Cognitive Strain Detector', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 32:
            return True
        if k == ord('q'):
            return False


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

        # Calibration baselines (overwritten by calibrate())
        self.baseline_blink_rate = HIGH_BLINK_RATE
        self.ear_threshold = EAR_BLINK_THRESHOLD
        self.perclos_threshold = EAR_BLINK_THRESHOLD / EAR_BLINK_RATIO * EAR_PERCLOS_RATIO

        # Rolling state
        self.blink_consec = 0
        self.blink_times = collections.deque()
        self.gaze_history = collections.deque(maxlen=30)
        self.emotion_history = collections.deque(maxlen=60)
        self.iris_diameter_history = collections.deque(maxlen=60)
        self.ear_history = collections.deque(maxlen=PERCLOS_WINDOW)
        self.no_face_frames = 0

        # CSV log — opened by start_recording() once the user starts the session
        self._csv_file = None
        self._csv = None

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

            self.ear_history.append(ear)
            if ear < self.ear_threshold:
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

        # ── Face-presence tracking ────────────────────────────────────────────
        face_found = bool(faces) or (mp_result.multi_face_landmarks is not None)
        self.no_face_frames = 0 if face_found else self.no_face_frames + 1

        # ── Strain score ──────────────────────────────────────────────────────
        metrics['blink_rate'] = self._blink_rate()
        metrics['gaze_jitter'] = self._gaze_jitter()
        metrics['perclos'] = self._perclos()
        metrics['strain_score'], metrics['strain_label'] = self._strain_score(metrics)

        # ── Log to CSV (only while recording and face is visible) ─────────────
        if self._csv is not None and face_found:
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
                round(metrics['perclos'], 4),
            ])

        # ── Annotate frame ────────────────────────────────────────────────────
        self._draw(frame, metrics, faces)
        return frame, metrics

    # ── Calibration ───────────────────────────────────────────────────────────

    def calibrate(self, cap, duration=CALIBRATION_SECS):
        """Two-phase calibration:
        Phase A — measure open-eye EAR to set a personal blink threshold.
        Phase B — count blinks with that threshold to set a personal blink-rate baseline.
        """
        if not _wait_for_key(cap, 'Step 1 of 2: Calibration',
                             f'Keep eyes open for {BASELINE_SECS}s, then blink normally'):
            return

        font = cv2.FONT_HERSHEY_SIMPLEX

        # ── Phase A: open-eye EAR baseline ───────────────────────────────────
        print(f"Phase A: measuring open-eye EAR for {BASELINE_SECS}s — keep eyes open...")
        open_ears = []
        end_a = time.time() + BASELINE_SECS

        while time.time() < end_a:
            ret, frame = cap.read()
            if not ret:
                break
            remaining = max(0, int(end_a - time.time()) + 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_ear = None

            for face in self.detector(gray, 0):
                landmarks = self.predictor(gray, face)
                left_pts  = _eye_pts(landmarks, 36, 42)
                right_pts = _eye_pts(landmarks, 42, 48)
                ear = (_ear(left_pts) + _ear(right_pts)) / 2.0
                open_ears.append(ear)
                current_ear = ear
                break

            w = frame.shape[1]
            cv2.rectangle(frame, (0, 0), (w, 80), (0, 60, 0), -1)
            cv2.putText(frame, 'PHASE 1 — Keep your eyes open naturally',
                        (10, 28), font, 0.65, (0, 255, 150), 2)
            ear_txt = f'EAR: {current_ear:.3f}' if current_ear else 'No face detected'
            cv2.putText(frame,
                        f'Time remaining: {remaining}s    {ear_txt}    Samples: {len(open_ears)}',
                        (10, 60), font, 0.52, (200, 200, 200), 1)
            cv2.imshow('Cognitive Strain Detector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

        # Derive personal threshold from the 80th-percentile open-eye EAR
        if len(open_ears) >= 5:
            open_eye_ear = float(np.percentile(open_ears, 80))
            self.ear_threshold    = open_eye_ear * EAR_BLINK_RATIO
            self.perclos_threshold = open_eye_ear * EAR_PERCLOS_RATIO
        else:
            open_eye_ear = EAR_BLINK_THRESHOLD / EAR_BLINK_RATIO
            self.ear_threshold    = EAR_BLINK_THRESHOLD
            self.perclos_threshold = open_eye_ear * EAR_PERCLOS_RATIO
        print(f"Open-eye EAR: {open_eye_ear:.3f}  →  "
              f"blink thr: {self.ear_threshold:.3f}  "
              f"PERCLOS thr: {self.perclos_threshold:.3f}")

        # ── Phase B: blink counting ───────────────────────────────────────────
        blink_secs = duration - BASELINE_SECS
        print(f"Phase B: counting blinks for {blink_secs}s — blink normally...")
        cal_blinks = 0
        cal_consec = 0
        end_b = time.time() + blink_secs

        while time.time() < end_b:
            ret, frame = cap.read()
            if not ret:
                break
            remaining = max(0, int(end_b - time.time()) + 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            for face in self.detector(gray, 0):
                landmarks = self.predictor(gray, face)
                left_pts  = _eye_pts(landmarks, 36, 42)
                right_pts = _eye_pts(landmarks, 42, 48)
                ear = (_ear(left_pts) + _ear(right_pts)) / 2.0
                if ear < self.ear_threshold:
                    cal_consec += 1
                else:
                    if cal_consec >= EAR_CONSEC_FRAMES:
                        cal_blinks += 1
                    cal_consec = 0
                break

            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
            cv2.putText(frame, 'PHASE 2 — Blink normally',
                        (10, 28), font, 0.65, (0, 255, 255), 2)
            cv2.putText(frame,
                        f'Time remaining: {remaining}s    '
                        f'Blinks: {cal_blinks}    '
                        f'Threshold EAR: {self.ear_threshold:.3f}',
                        (10, 60), font, 0.52, (200, 200, 200), 1)
            cv2.imshow('Cognitive Strain Detector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.baseline_blink_rate = max(cal_blinks * (60.0 / blink_secs), 8.0)
        print(f"Calibration done. "
              f"Blink threshold: {self.ear_threshold:.3f}  "
              f"Baseline rate: {self.baseline_blink_rate:.1f} blinks/min")
        return self.baseline_blink_rate

    def start_recording(self, participant_id='unknown'):
        """Open the session CSV. Call this only when the user commits to recording."""
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        session_ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_path = os.path.join(log_dir, f'session_{participant_id}_{session_ts}.csv')
        self._csv_file = open(log_path, 'w', newline='', buffering=1)
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow([
            'timestamp', 'strain_score', 'strain_label',
            'emotion', 'blink_rate', 'ear',
            'gaze_jitter', 'gaze_direction', 'iris_diameter', 'perclos',
        ])
        print(f"Recording to {log_path}")

    def close(self):
        if self._csv_file is not None:
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

    def _perclos(self):
        """Fraction of recent frames where EAR < perclos_threshold (eye ≥ 80 % closed)."""
        if len(self.ear_history) < 30:
            return 0.0
        arr = np.array(self.ear_history)
        return float(np.mean(arr < self.perclos_threshold))

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

        if metrics.get('perclos', 0.0) > PERCLOS_THRESHOLD:
            score += 0.25
            reasons.append('sustained eye closure')

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
        put(f"PERCLOS : {metrics['perclos']*100:.1f}%")
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

        # ── Face-not-detected warning ─────────────────────────────────────────
        nf = self.no_face_frames
        if nf >= 90:
            # Prominent red tint + centred message (face gone ~3 s+)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 160), -1)
            cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)
            secs = nf // 30
            for text, fy, scale, color, thick in [
                ('FACE NOT DETECTED',               h // 2 - 20, 1.1,  (255, 255, 255), 3),
                ('Please look at the camera',       h // 2 + 28, 0.65, (220, 220, 220), 1),
                (f'({secs}s without detection)',     h // 2 + 60, 0.52, (180, 180, 180), 1),
            ]:
                (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
                cv2.putText(frame, text, ((w - tw) // 2, fy),
                            cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)
        elif nf >= 30:
            # Subtle amber strip at the bottom (~1–3 s)
            cv2.rectangle(frame, (0, h - 36), (w, h), (0, 140, 220), -1)
            msg = 'Face not visible — please look at the camera'
            (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.putText(frame, msg, ((w - tw) // 2, h - 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def main():
    participant_id = input("Enter participant ID: ").strip() or "unknown"

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

        if not _wait_for_key(cap, 'Step 2 of 2: Recording',
                             'Session data will be saved when you press SPACE'):
            return

        detector_system.start_recording(participant_id)
        print("Recording — press Q to quit")

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