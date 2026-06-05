# eye-CU — Real-Time Cognitive Strain Detector

A webcam-based system that measures cognitive strain in real time by combining facial expression recognition, eye-tracking, and iris analysis. Designed for research use — every session is logged to CSV so results can be analysed and included directly in a thesis.

---

## How It Works

The pipeline runs three signal sources in parallel on every frame:

### 1. Facial Expression (FER CNN)
A custom convolutional neural network trained on FER2013 and CK+ classifies the face crop into one of seven emotions: *Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise*. Strain-associated emotions (Fear, Sad, Angry, Disgust) contribute to the strain score when they dominate a rolling 60-frame window.

### 2. Eye Tracking — dlib 68-landmark detector
- **Eye Aspect Ratio (EAR):** ratio of vertical to horizontal eye distances. Drops sharply during a blink.
- **Blink detection:** a blink is counted when EAR stays below 0.21 for ≥ 2 consecutive frames.
- **Blink rate:** blinks per minute over a 30-second rolling window, compared against the user's personal calibrated baseline.

### 3. Iris Analysis — MediaPipe FaceMesh (478 landmarks)
- **Iris diameter:** horizontal span of the four iris-edge landmarks per eye, averaged across both eyes.
- **Gaze direction:** position of the iris centre relative to the eye corners → *left / centre / right*.
- **Gaze jitter:** standard deviation of the iris centre position over 30 frames — a proxy for fixation stability.

### Strain Score
Each frame produces a score in [0, 1] from up to four signals:

| Signal | Condition | Contribution |
|---|---|---|
| Blink rate | > 1.5 × personal baseline | +0.33 |
| Blink rate | < 0.5 × personal baseline | +0.20 |
| Gaze jitter | > 15 px | +0.33 |
| Emotion | strain emotion in > 40 % of recent frames | +0.34 |
| Iris diameter | > 15 % above its 60-frame rolling mean | +0.20 |

Score → label: `LOW STRAIN` (< 0.33), `MODERATE STRAIN` (0.33 – 0.66), `HIGH STRAIN` (≥ 0.66).

---

## Project Structure

```
eye-CU/
├── pipeline/
│   └── cognitive_strain_detector.py   ← main program
├── model/
│   ├── model.py                        ← FER CNN architecture
│   ├── train.py                        ← training script
│   └── evaluate.py                     ← evaluation script
├── models/
│   └── shape_predictor_68_face_landmarks.dat   ← dlib predictor (95 MB)
├── preprocessing/
│   ├── preprocessing.py                ← FER2013 dataset loader
│   ├── ck_preprocessing.py             ← CK+ dataset loader
│   ├── combined_dataset.py             ← merge datasets for training
│   ├── opencv_preprocessing.py         ← face alignment utilities
│   ├── apply_opencv_preprocessing.py   ← apply preprocessing to FER2013
│   └── ck_apply_opencv_preprocessing.py
├── results/
│   ├── fer_model.pth                   ← trained model weights
│   ├── training_history.txt
│   └── evaluation_report.txt
├── analysis/
│   └── analyse_session.py              ← post-session analysis script
├── logs/                               ← session CSVs written here at runtime
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.9 – 3.11
- A webcam
- PyTorch (install separately before the requirements):

```bash
# CPU — works on any machine
pip install torch torchvision

# Apple Silicon (MPS acceleration)
pip install torch torchvision
```

For GPU builds see [pytorch.org/get-started](https://pytorch.org/get-started).

### Install dependencies

```bash
cd eye-CU
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

---

## Running the Detector

```bash
cd eye-CU/pipeline
python cognitive_strain_detector.py
```

### Startup sequence

The session is split into three phases, each triggered by pressing **SPACE**:

```
┌─────────────────────────────────────────────────┐
│  Step 1 of 2: Calibration                       │
│  Sit naturally and blink normally for 20 s       │
│                                                  │
│         SPACE to continue  |  Q to quit          │
└─────────────────────────────────────────────────┘
         ↓ SPACE
┌─────────────────────────────────────────────────┐
│  ██ CALIBRATING — look naturally at the screen  │
│  Time remaining: 14 s    Blinks detected: 3      │
└─────────────────────────────────────────────────┘
         ↓ (automatic after 20 s)
┌─────────────────────────────────────────────────┐
│  Step 2 of 2: Recording                          │
│  Session data will be saved when you press SPACE │
│                                                  │
│         SPACE to continue  |  Q to quit          │
└─────────────────────────────────────────────────┘
         ↓ SPACE  →  recording starts, CSV opens
```

**Calibration** measures your natural blink rate for 20 seconds. This sets a personal baseline so the strain thresholds adapt to you rather than using a generic value.

**Recording** only begins after the second SPACE press, so calibration frames are never included in the session data.

### Controls during recording

| Key | Action |
|---|---|
| `Q` | Stop recording and save |

### Face-not-detected warnings

If the face disappears (hand covering face, looking away, poor lighting):

| Duration | On-screen feedback |
|---|---|
| < 1 s | No warning — brief glances are ignored |
| 1 – 3 s | Amber strip at the bottom of the frame |
| > 3 s | Red tint over the whole frame + elapsed time counter |

Frames without a detected face are **not written** to the CSV, keeping the session data clean.

---

## Session Data

Each session is saved to `logs/session_YYYY-MM-DD_HH-MM-SS.csv`:

| Column | Description |
|---|---|
| `timestamp` | Unix timestamp (seconds) |
| `strain_score` | Composite strain score [0, 1] |
| `strain_label` | LOW / MODERATE / HIGH STRAIN |
| `emotion` | Detected facial emotion |
| `blink_rate` | Blinks per minute (30 s rolling window) |
| `ear` | Eye Aspect Ratio |
| `gaze_jitter` | Iris position std dev over 30 frames (px) |
| `gaze_direction` | left / centre / right |
| `iris_diameter` | Average iris diameter in pixels |

---

## Analysing a Session

```bash
cd eye-CU/analysis

# Analyse the most recent session automatically
python analyse_session.py

# Analyse a specific file
python analyse_session.py ../logs/session_2026-06-05_12-25-18.csv
```

This produces:

1. **A 7-panel PNG** saved next to the CSV (`session_..._analysis.png`), ready to drop into a thesis:
   - Strain score over time (with rolling mean and threshold lines)
   - Blink rate, gaze jitter, EAR over time
   - Iris diameter over time
   - Emotion distribution (bar chart)
   - Strain label breakdown (donut chart)

2. **A text summary** printed to the terminal — session duration, mean/peak strain, blink statistics, dominant emotion, gaze direction split.

---

## Retraining the Model (optional)

The pre-trained weights in `results/fer_model.pth` are ready to use out of the box. If you want to retrain:

1. Download the datasets and place them under `data/`:
   - **FER2013** — [kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)
   - **CK+** — available from the CMU/Pittsburgh FTP (request access from the authors)

2. Preprocess:
   ```bash
   cd preprocessing
   python apply_opencv_preprocessing.py
   python ck_apply_opencv_preprocessing.py
   ```

3. Train:
   ```bash
   cd model
   python train.py
   ```

4. Evaluate:
   ```bash
   python evaluate.py
   ```

The best checkpoint is saved to `results/fer_model_best.pth` and the final epoch to `results/fer_model.pth`. The pipeline loads `fer_model.pth` by default.
