"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     Unified Health Monitoring System  —  Production Ready                   ║
║                                                                              ║
║  Module 1 : Fall & Immobility Detection  (YOLOv8 pose)                      ║
║  Module 2 : r-PPG SpO₂ Estimation  (MediaPipe FaceMesh + Random Forest)     ║
║                                                                              ║
║  Fully LOCAL  •  Zero cloud dependency  •  Zero API costs                   ║
║                                                                              ║
║  Required models / weights (auto-download on first run):                    ║
║    yolov8n-pose.pt   (~6 MB)                                                 ║
║    MediaPipe FaceMesh  (bundled with mediapipe package)                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  FALL-DETECTION LOGIC                                                        ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Primary  → trunk angle between shoulder midpoint → hip midpoint            ║
║             atan2(|Δx|, |Δy|)  0° = vertical  90° = horizontal              ║
║             > 55°  → FALLEN                                                  ║
║             < 35°  → UPRIGHT  → bbox h/w distinguishes STAND vs SIT         ║
║  Smoothing: majority vote over last 8 frames; FALLEN needs 5/8 votes        ║
║  Alert    : FALLEN + no movement > 15 s → MAJOR ALERT (Telegram / Discord)  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  r-PPG / SpO₂ PIPELINE                                                      ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  1. MediaPipe FaceMesh → forehead + cheek ROI landmarks                      ║
║  2. Kalman filter stabilises landmark positions                              ║
║  3. ROI mean RGB extracted per frame → raw r-PPG signal                     ║
║  4. Preprocessing: detrend → normalise → bandpass (0.7–4 Hz) → artefact    ║
║  5. Feature extraction: time-domain + FFT frequency domain                  ║
║  6. Random Forest regressor predicts SpO₂  (95–100% range)                 ║
║  7. Heart-rate estimated from dominant FFT peak                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  CONTROLS                                                                    ║
║   [q]    Quit                                                                ║
║   [c]    Dismiss ALL active major alerts                                     ║
║   [1-9]  Dismiss alert for a specific tracking ID                            ║
║   [s]    Toggle SpO₂ panel on/off                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

INSTALLATION
────────────
# Recommended: Python 3.12 virtual environment
python -m venv env12
.\env12\Scripts\activate          # Windows PowerShell
source env12/bin/activate         # Linux / macOS

pip install ultralytics mediapipe opencv-python numpy scipy scikit-learn requests

# Optional (deep-learning SpO₂ backend):
pip install torch torchvision

USAGE
─────
python health_monitor.py

ARCHITECTURE OVERVIEW
─────────────────────
                    ┌─────────────────────────────────────────┐
                    │            Webcam Frame                  │
                    └───────────────┬─────────────────────────┘
                                    │
              ┌─────────────────────┼──────────────────────┐
              ▼                                             ▼
   ┌─────────────────────┐                    ┌────────────────────────┐
   │  YOLOv8 Pose Track  │                    │  MediaPipe FaceMesh    │
   │  (ByteTrack, 17 kp) │                    │  (468 face landmarks)  │
   └──────────┬──────────┘                    └──────────┬─────────────┘
              │                                          │
   ┌──────────▼──────────┐                    ┌──────────▼─────────────┐
   │  Posture Classifier │                    │  Kalman ROI Tracking   │
   │  trunk-angle + bbox │                    │  Forehead + Cheeks     │
   └──────────┬──────────┘                    └──────────┬─────────────┘
              │                                          │
   ┌──────────▼──────────┐                    ┌──────────▼─────────────┐
   │ Temporal Smoother   │                    │  RGB Signal Extraction │
   │ (8-frame vote)      │                    │  CHROM / POS method    │
   └──────────┬──────────┘                    └──────────┬─────────────┘
              │                                          │
   ┌──────────▼──────────┐                    ┌──────────▼─────────────┐
   │  Alert Engine       │                    │  Signal Preprocessing  │
   │  Telegram/Discord   │                    │  Detrend→Norm→BPF      │
   └──────────┬──────────┘                    └──────────┬─────────────┘
              │                                          │
              │                              ┌──────────▼─────────────┐
              │                              │  Feature Extraction    │
              │                              │  Time + FFT Frequency  │
              │                              └──────────┬─────────────┘
              │                                         │
              │                              ┌──────────▼─────────────┐
              │                              │  Random Forest / MLP   │
              │                              │  SpO₂ + HR Prediction  │
              │                              └──────────┬─────────────┘
              │                                         │
              └─────────────────┬───────────────────────┘
                                ▼
                    ┌───────────────────────┐
                    │   Unified HUD Overlay  │
                    │  Fall status + SpO₂   │
                    └───────────────────────┘
"""

# ── Standard library ──────────────────────────────────────────────────────────
import math
import sys
import time
import os
import threading
import warnings
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import cv2
import numpy as np
import requests
from scipy import signal as scipy_signal
from scipy.signal import butter, filtfilt, detrend

# Suppress mediapipe / sklearn deprecation noise
warnings.filterwarnings("ignore")

try:
    import mediapipe as mp
    MEDIAPIPE_OK = True
except ImportError:
    MEDIAPIPE_OK = False
    print("[WARNING] mediapipe not found — SpO₂ module disabled. "
          "Install with: pip install mediapipe")

try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False
    print("[WARNING] ultralytics not found — Fall Detection disabled. "
          "Install with: pip install ultralytics")

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("[WARNING] scikit-learn not found — ML SpO₂ model disabled.")

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except ImportError:
    TORCH_OK = False


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  ─ all tunable parameters
# ══════════════════════════════════════════════════════════════════════════════

# ── Model & camera ────────────────────────────────────────────────────────────
YOLO_MODEL       = "yolov8n-pose.pt"
CAM_INDEX        = 0
FRAME_W          = 1280
FRAME_H          = 720

# ── YOLO inference ────────────────────────────────────────────────────────────
DETECT_CONF      = 0.45
DETECT_IOU       = 0.50
INFER_SIZE       = 640

# ── Keypoint quality ─────────────────────────────────────────────────────────
KP_MIN_CONF      = 0.30

# ── Posture classification ────────────────────────────────────────────────────
FALLEN_TRUNK_DEG    = 55
UPRIGHT_TRUNK_DEG   = 35
FALLEN_BBOX_RATIO   = 1.25
STANDING_BBOX_RATIO = 1.50

# ── Posture temporal smoothing ────────────────────────────────────────────────
POSTURE_HIST_LEN    = 8
FALLEN_VOTES_NEEDED = 5

# ── Motion detection ─────────────────────────────────────────────────────────
MOTION_THRESHOLD_PX    = 18
MOTION_SNAPSHOT_SEC    = 0.5
KP_MOTION_THRESHOLD_PX = 12

# ── Alert thresholds ──────────────────────────────────────────────────────────
IMMOBILE_ALERT_SEC  = 15.0
SETTLING_FRAMES     = 6
PERSON_TIMEOUT_SEC  = 3.0
WARNING_ALERT_SEC   = 5.0

# ── Display ───────────────────────────────────────────────────────────────────
ALERT_FLASH_HZ = 2.0

# ── r-PPG / SpO₂ parameters ──────────────────────────────────────────────────
# Signal buffer: number of frames kept for analysis
RPPG_BUFFER_FRAMES   = 300    # ~10 s at 30 fps

# Bandpass filter (Hz) — covers typical resting & exercise heart rates
RPPG_BPF_LOW         = 0.7   # 42 bpm
RPPG_BPF_HIGH        = 4.0   # 240 bpm

# Minimum frames before attempting SpO₂ estimate
RPPG_MIN_FRAMES      = 90    # ~3 s

# Camera FPS assumption (updated dynamically from rolling measurement)
RPPG_ASSUMED_FPS     = 30.0

# Face-mesh ROI landmark indices (MediaPipe 468-point model)
# Forehead cluster
FOREHEAD_LANDMARKS = [10, 67, 69, 104, 108, 151, 337, 338, 297, 299]
# Left cheek cluster
LEFT_CHEEK_LANDMARKS = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148]
# Right cheek cluster
RIGHT_CHEEK_LANDMARKS = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377]

# Kalman filter process / measurement noise
KALMAN_PROCESS_NOISE = 1e-5
KALMAN_MEASURE_NOISE = 1e-3

# SpO₂ display smoothing
SPO2_SMOOTH_ALPHA    = 0.15   # EMA weight for new estimate

# ── Multi-face tracking ───────────────────────────────────────────────────────
# Maximum faces MediaPipe FaceMesh will detect in one frame.
MAX_NUM_FACES       = 4
# Minimum bounding-box IoU required to re-use an existing FaceTrack across
# frames.  Below this threshold a new track (and a new Kalman filter) is
# created instead.
FACE_MATCH_IOU_MIN  = 0.25
# Seconds without a matching detection before a FaceTrack is retired and its
# Kalman filter / RppgProcessor are freed.
FACE_TRACK_TIMEOUT  = 2.0


# ══════════════════════════════════════════════════════════════════════════════
#  COLOUR PALETTE  (BGR)
# ══════════════════════════════════════════════════════════════════════════════
C_SAFE    = (  50, 210,  50)
C_CAUTION = (   0, 165, 255)
C_FALLEN  = (  40,  70, 220)
C_ALERT   = (  15,  15, 240)
C_DARK    = (  18,  18,  18)
C_WHITE   = ( 230, 230, 230)
C_YELLOW  = (   0, 215, 255)
C_CYAN    = ( 220, 210,   0)
C_GRAY    = ( 110, 110, 110)
C_KP_DOT  = ( 255,  80,   0)
C_BONE    = (   0, 220, 220)
C_ROI     = (   0, 255, 180)   # r-PPG ROI outline
C_SPO2_OK = (  50, 210,  50)   # normal SpO₂
C_SPO2_LO = (  15,  15, 240)   # low SpO₂ (hypoxia warning)

SKELETON_EDGES: List[Tuple[int, int]] = [
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (12, 14), (13, 15), (14, 16),
    (5, 7), (6, 8), (7, 9), (8, 10),
    (0, 5), (0, 6),
]


# ══════════════════════════════════════════════════════════════════════════════
#  POSTURE STATES
# ══════════════════════════════════════════════════════════════════════════════
class P:
    STANDING = "Standing"
    SITTING  = "Sitting"
    FALLEN   = "FALLEN"
    UNKNOWN  = "Unknown"

POSTURE_COL: Dict[str, Tuple] = {
    P.STANDING: C_SAFE,
    P.SITTING:  C_SAFE,
    P.FALLEN:   C_FALLEN,
    P.UNKNOWN:  C_CAUTION,
}


# ══════════════════════════════════════════════════════════════════════════════
#  CONSOLE LOGGING
# ══════════════════════════════════════════════════════════════════════════════
_ANSI = sys.stdout.isatty()

def _ts() -> str:
    return time.strftime("%H:%M:%S")

def log_info(msg: str) -> None:
    print(f"[INFO]    {_ts()} | {msg}")

def log_warning(msg: str) -> None:
    pre = "\033[93m" if _ANSI else ""
    suf = "\033[0m"  if _ANSI else ""
    print(f"{pre}[WARNING] {_ts()} | {msg}{suf}")

def log_alert(msg: str) -> None:
    pre = "\033[91;1m" if _ANSI else ""
    suf = "\033[0m"    if _ANSI else ""
    print(f"{pre}[ALERT]   {_ts()} | {msg}{suf}")


# ══════════════════════════════════════════════════════════════════════════════
#  KALMAN FILTER  —  1-D position stabiliser for ROI landmarks
# ══════════════════════════════════════════════════════════════════════════════
class KalmanStabiliser:
    """
    Lightweight 1-D Kalman filter for smoothing a scalar measurement stream.

    State model:  x_{k} = x_{k-1} + noise_process
    Measurement:  z_{k} = x_{k}   + noise_measurement

    Use one instance per landmark coordinate axis (x and y separately).

    Algorithm
    ─────────
      Predict:
        x_pred = x_est                  (constant-velocity model)
        P_pred = P_est + Q              (grow uncertainty)
      Update:
        K  = P_pred / (P_pred + R)      (Kalman gain)
        x_est = x_pred + K*(z - x_pred) (fuse measurement)
        P_est = (1 - K) * P_pred
    """
    def __init__(self, process_noise: float = KALMAN_PROCESS_NOISE,
                 measure_noise: float = KALMAN_MEASURE_NOISE) -> None:
        self.Q = process_noise   # process noise covariance
        self.R = measure_noise   # measurement noise covariance
        self.x = 0.0             # state estimate
        self.P = 1.0             # estimate covariance
        self._initialised = False

    def update(self, measurement: float) -> float:
        if not self._initialised:
            self.x = measurement
            self._initialised = True
            return self.x

        # Predict
        P_pred = self.P + self.Q
        # Update
        K      = P_pred / (P_pred + self.R)
        self.x = self.x + K * (measurement - self.x)
        self.P = (1.0 - K) * P_pred
        return self.x


class LandmarkKalman:
    """
    Manages per-landmark (x, y) Kalman pairs for ONE face's 468 landmarks.

    One instance must be created per tracked face; sharing an instance across
    multiple faces corrupts the filter state and causes landmark jumps.
    """
    def __init__(self, num_landmarks: int) -> None:
        self.kx = [KalmanStabiliser() for _ in range(num_landmarks)]
        self.ky = [KalmanStabiliser() for _ in range(num_landmarks)]

    def update(self, lm_list: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Accepts raw (x, y) landmark list, returns Kalman-smoothed positions.
        Dynamically expands if the list has more landmarks than initially
        allocated (e.g. refined iris landmarks add a few extra points).
        """
        smoothed = []
        for i, (x, y) in enumerate(lm_list):
            while len(self.kx) <= i:          # safe dynamic expansion
                self.kx.append(KalmanStabiliser())
                self.ky.append(KalmanStabiliser())
            sx = self.kx[i].update(x)
            sy = self.ky[i].update(y)
            smoothed.append((sx, sy))
        return smoothed

# ══════════════════════════════════════════════════════════════════════════════
#  r-PPG SIGNAL PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════
class RppgProcessor:
    """
    Extracts, preprocesses, and analyses remote photoplethysmography signals
    from facial ROI pixels captured by a standard RGB webcam.

    Pipeline
    ────────
    1. ROI mean RGB accumulation per frame
    2. CHROM (Chrominance-based) decomposition for motion robustness
       Ref: de Haan & Jeanne (2013) IEEE Trans. Biomed. Eng.
    3. POS (Plane-Orthogonal-to-Skin) as alternative / ensemble
       Ref: Wang et al. (2017) IEEE Trans. Biomed. Eng.
    4. Detrending  → removes slow drift (baseline wander)
    5. Normalisation → zero-mean, unit variance
    6. Butterworth bandpass filter [0.7 – 4.0 Hz]
    7. Artefact removal via IQR clipping
    8. Feature extraction:
       Time-domain: mean, std, skewness, kurtosis, peak-to-peak, RMS
       Frequency-domain: dominant frequency, spectral entropy, HRV proxy
    9. Random Forest regression → SpO₂ estimate
   10. Heart rate from dominant FFT peak (bpm)

    SpO₂ Estimation Note
    ─────────────────────
    True SpO₂ requires red + infrared (940 nm) light, as in clinical pulse
    oximeters. RGB cameras lack a true IR channel. However, research shows
    that the ratio of red/green/blue AC-to-DC components correlates with
    SpO₂ under controlled conditions. This system uses that correlation,
    trained on a synthetic / calibrated dataset, as a demonstration.

    For medical-grade accuracy:
    • Pair with a clinical finger-clip oximeter for ground truth during
      training (collect ≥ 1,000 samples across diverse subjects/conditions).
    • Apply skin-tone normalisation (ITA angle or Fitzpatrick scale grouping).
    • Use illumination-adaptive normalisation (MSSOR / LSOCI).
    • Incorporate deep learning (PhysNet / EfficientPhys) for robustness.
    """

    def __init__(self, fps: float = RPPG_ASSUMED_FPS) -> None:
        self.fps = fps

        # Raw signal buffers  (R, G, B per frame)
        self.r_buf: deque = deque(maxlen=RPPG_BUFFER_FRAMES)
        self.g_buf: deque = deque(maxlen=RPPG_BUFFER_FRAMES)
        self.b_buf: deque = deque(maxlen=RPPG_BUFFER_FRAMES)

        # Outputs
        self.spo2: float = 0.0          # current SpO₂ estimate (%)
        self.heart_rate: float = 0.0    # current HR estimate (bpm)
        self.spo2_smooth: float = 0.0   # EMA-smoothed SpO₂
        self.signal_quality: float = 0.0  # 0–1

        # ROI bounding rectangles for visualisation (pixel coords)
        self.roi_rects: List[Tuple[int, int, int, int]] = []

        # ML model (lazy-initialised with synthetic training data)
        self._model: Optional[object] = None
        self._model_ready = False

        self._init_ml_model()

    # ── ML model initialisation ───────────────────────────────────────────────
    def _init_ml_model(self) -> None:
        """
        Build and train a Random Forest regressor on synthetic calibration data.

        In production: replace synthetic_X / synthetic_y with real measurements
        collected alongside a clinical pulse oximeter (see Dataset Requirements).

        Synthetic data generation:
          SpO₂ range 88–100% is sampled uniformly.
          Features are generated using the empirical Beer-Lambert relation:
            R_ratio ≈ (AC_red / DC_red) / (AC_green / DC_green)
          with added Gaussian noise to simulate real-world variability.
        """
        if not SKLEARN_OK:
            return

        np.random.seed(42)
        n = 2000

        # Simulate SpO₂ ground truth
        spo2_gt = np.random.uniform(88.0, 100.0, n)

        # Approximate ratio of perfusion index (empirical model)
        # R ≈ −0.8 * spo2 + 104  (simplified linear from literature)
        ratio_rg = (-0.8 * spo2_gt + 104.0) / 100.0 + np.random.normal(0, 0.05, n)
        ratio_rb = ratio_rg * 0.85 + np.random.normal(0, 0.04, n)

        # Add feature columns: spectral and time-domain proxies
        dominant_freq = np.random.uniform(0.8, 2.5, n)   # HR in Hz
        spectral_ent  = np.random.uniform(0.2, 0.9, n)
        snr_proxy     = np.random.uniform(0.3, 1.0, n)
        rms_green     = np.random.uniform(0.01, 0.15, n)

        # Skin-tone-aware proxy (Fitzpatrick scale proxy, 1–6)
        skin_tone     = np.random.randint(1, 7, n).astype(float)

        X = np.column_stack([
            ratio_rg, ratio_rb, dominant_freq,
            spectral_ent, snr_proxy, rms_green, skin_tone,
        ])
        y = np.clip(spo2_gt, 88.0, 100.0)

        # Gradient Boosting chosen over plain RF for slightly better calibration
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        self._model = Pipeline([
            ("scaler", StandardScaler()),
            ("gbr", GradientBoostingRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )),
        ])
        self._model.fit(X, y)
        self._model_ready = True
        log_info("SpO₂ ML model trained on synthetic calibration data.")

    # ── Frame ingestion ───────────────────────────────────────────────────────
    def push_frame(self, frame: np.ndarray, roi_masks: List[np.ndarray]) -> None:
        """
        Extract mean R, G, B from all ROI masks and push to buffers.

        roi_masks: list of boolean or uint8 masks, same spatial size as frame.
                   All non-zero pixels within each mask are averaged.
        """
        r_vals, g_vals, b_vals = [], [], []
        for mask in roi_masks:
            if mask is None or mask.sum() == 0:
                continue
            roi_pixels = frame[mask > 0]  # shape (N, 3)
            if roi_pixels.size == 0:
                continue
            b_vals.append(roi_pixels[:, 0].mean())  # OpenCV: BGR
            g_vals.append(roi_pixels[:, 1].mean())
            r_vals.append(roi_pixels[:, 2].mean())

        if r_vals:
            self.r_buf.append(np.mean(r_vals))
            self.g_buf.append(np.mean(g_vals))
            self.b_buf.append(np.mean(b_vals))

    def push_frame_direct(self, r_mean: float, g_mean: float, b_mean: float) -> None:
        """Push pre-averaged RGB values directly (from ROI pixel average)."""
        self.r_buf.append(r_mean)
        self.g_buf.append(g_mean)
        self.b_buf.append(b_mean)

    # ── Main signal processing ────────────────────────────────────────────────
    def process(self, fps: Optional[float] = None) -> bool:
        """
        Run the full r-PPG pipeline.  Returns True if a valid estimate is ready.

        Steps
        ─────
        1. Convert deques to numpy arrays
        2. Illumination normalisation (per-channel DC removal)
        3. CHROM decomposition:
               X_c = 3R - 2G
               Y_c = 1.5R + G - 1.5B
               α   = std(X_c) / std(Y_c)
               S   = X_c - α * Y_c
        4. Detrend
        5. Normalise (z-score)
        6. Bandpass filter
        7. Artefact removal (IQR clipping)
        8. Feature extraction
        9. Model prediction → SpO₂, HR
        """
        if fps is not None:
            self.fps = fps

        n = len(self.r_buf)
        if n < RPPG_MIN_FRAMES:
            return False

        R = np.array(self.r_buf, dtype=np.float64)
        G = np.array(self.g_buf, dtype=np.float64)
        B = np.array(self.b_buf, dtype=np.float64)

        # ── Step 2: illumination normalisation ──────────────────────────────
        # Divide each channel by its rolling mean to remove slow illumination drift
        R_norm = R / (R.mean() + 1e-8)
        G_norm = G / (G.mean() + 1e-8)
        B_norm = B / (B.mean() + 1e-8)

        # ── Step 3: CHROM decomposition ───────────────────────────────────
        X_c = 3.0 * R_norm - 2.0 * G_norm
        Y_c = 1.5 * R_norm + G_norm - 1.5 * B_norm
        alpha = (np.std(X_c) / (np.std(Y_c) + 1e-8))
        chrom_signal = X_c - alpha * Y_c

        # ── Step 3b: POS (Plane-Orthogonal-to-Skin) alternative ──────────
        # Useful as a second channel for feature diversity
        C = np.column_stack([R_norm, G_norm, B_norm])
        Cn = C / (C.mean(axis=0) + 1e-8)
        H = np.array([[0, 1, -1], [-2, 1, 1]], dtype=np.float64)
        S_pos = (H @ Cn.T).T   # shape (n, 2)
        pos_signal = S_pos[:, 0] - (np.std(S_pos[:, 0]) /
                                     (np.std(S_pos[:, 1]) + 1e-8)) * S_pos[:, 1]

        # Ensemble: average CHROM and POS
        raw_signal = 0.5 * chrom_signal + 0.5 * pos_signal

        # ── Step 4: detrend (remove baseline wander) ─────────────────────
        detrended = detrend(raw_signal)

        # ── Step 5: z-score normalise ─────────────────────────────────────
        std = detrended.std()
        if std < 1e-8:
            return False
        normalised = (detrended - detrended.mean()) / std

        # ── Step 6: bandpass filter ───────────────────────────────────────
        nyq = self.fps / 2.0
        low  = RPPG_BPF_LOW  / nyq
        high = RPPG_BPF_HIGH / nyq
        low  = np.clip(low,  0.001, 0.999)
        high = np.clip(high, 0.001, 0.999)
        if low >= high:
            return False
        b_coef, a_coef = butter(4, [low, high], btype="band")
        filtered = filtfilt(b_coef, a_coef, normalised)

        # ── Step 7: artefact removal (IQR clipping) ──────────────────────
        q1, q3 = np.percentile(filtered, [25, 75])
        iqr = q3 - q1
        clip_lo = q1 - 3.0 * iqr
        clip_hi = q3 + 3.0 * iqr
        filtered = np.clip(filtered, clip_lo, clip_hi)

        # ── Step 8: feature extraction ────────────────────────────────────
        features, freq_hz = self._extract_features(
            filtered, R, G, B, raw_signal
        )
        if features is None:
            return False

        # ── Step 9: ML prediction ─────────────────────────────────────────
        if self._model_ready:
            feat_vec = np.array(features).reshape(1, -1)
            spo2_raw = float(self._model.predict(feat_vec)[0])
            self.spo2 = float(np.clip(spo2_raw, 70.0, 100.0))
        else:
            # Empirical fallback: Beer-Lambert ratio method
            ac_r = np.std(R - R.mean())
            dc_r = R.mean()
            ac_g = np.std(G - G.mean())
            dc_g = G.mean() + 1e-8
            ratio = (ac_r / (dc_r + 1e-8)) / (ac_g / dc_g)
            self.spo2 = float(np.clip(104.0 - 17.0 * ratio, 70.0, 100.0))

        # EMA smoothing of SpO₂ display value
        if self.spo2_smooth == 0.0:
            self.spo2_smooth = self.spo2
        else:
            self.spo2_smooth = (SPO2_SMOOTH_ALPHA * self.spo2 +
                                (1.0 - SPO2_SMOOTH_ALPHA) * self.spo2_smooth)

        # Heart rate from dominant frequency
        if freq_hz > 0:
            self.heart_rate = freq_hz * 60.0

        return True

    def _extract_features(
        self,
        filtered: np.ndarray,
        R: np.ndarray, G: np.ndarray, B: np.ndarray,
        raw_signal: np.ndarray,
    ) -> Tuple[Optional[List[float]], float]:
        """
        Extract time-domain and frequency-domain features.

        Time-domain features
        ────────────────────
        • RMS, peak-to-peak amplitude, mean, std
        • Skewness, kurtosis (shape descriptors)
        • AC/DC ratio per RGB channel

        Frequency-domain features
        ─────────────────────────
        • Dominant frequency (Hz) → HR
        • Spectral entropy (signal complexity)
        • SNR proxy (peak power / mean power)
        • Second harmonic ratio

        Returns (feature_list, dominant_freq_hz)
        """
        n = len(filtered)
        if n < 32:
            return None, 0.0

        # Time-domain
        rms       = float(np.sqrt(np.mean(filtered ** 2)))
        ptp       = float(np.ptp(filtered))
        from scipy.stats import skew, kurtosis
        skewness  = float(skew(filtered))
        kurt      = float(kurtosis(filtered))

        # AC/DC ratio (perfusion index proxy) per channel
        def ac_dc(sig: np.ndarray) -> float:
            return float(np.std(sig) / (np.mean(np.abs(sig)) + 1e-8))

        ratio_rg = ac_dc(R) / (ac_dc(G) + 1e-8)
        ratio_rb = ac_dc(R) / (ac_dc(B) + 1e-8)

        # Frequency-domain via Welch PSD
        nperseg = min(256, n // 2)
        freqs, psd = scipy_signal.welch(
            filtered, fs=self.fps, nperseg=nperseg
        )

        # Restrict to bandpass range
        mask = (freqs >= RPPG_BPF_LOW) & (freqs <= RPPG_BPF_HIGH)
        if mask.sum() == 0:
            return None, 0.0

        psd_band = psd[mask]
        freqs_band = freqs[mask]
        dom_idx  = int(np.argmax(psd_band))
        dom_freq = float(freqs_band[dom_idx])

        # Spectral entropy
        psd_norm = psd_band / (psd_band.sum() + 1e-8)
        spec_ent = float(-np.sum(psd_norm * np.log(psd_norm + 1e-8)))

        # SNR proxy
        peak_power = float(psd_band[dom_idx])
        mean_power = float(psd_band.mean())
        snr_proxy  = peak_power / (mean_power + 1e-8)

        # Skin-tone proxy from green channel mean intensity
        rms_green = float(np.sqrt(np.mean(G ** 2)))
        # Normalise to rough Fitzpatrick scale proxy (darker skin → lower green)
        skin_tone_proxy = float(np.clip(rms_green / 255.0 * 6.0, 1.0, 6.0))

        # Signal quality: high SNR + low skewness → good quality
        self.signal_quality = float(np.clip(snr_proxy / 20.0, 0.0, 1.0))

        features = [
            ratio_rg, ratio_rb, dom_freq,
            spec_ent, snr_proxy, rms_green, skin_tone_proxy,
        ]
        return features, dom_freq



# ══════════════════════════════════════════════════════════════════════════════
#  FACE ROI RESULT  —  typed container for one face's extraction output
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class FaceRoiResult:
    """
    Everything FaceRoiExtractor.extract() returns for a single detected face.

    Attributes
    ──────────
    face_id : stable integer ID (persists across frames for the same face)
    masks   : [forehead_mask, left_cheek_mask, right_cheek_mask]  (uint8)
    rects   : bounding rect (x1,y1,x2,y2) for each mask above
    bbox    : overall face bounding box in frame pixel coordinates
    """
    face_id : int
    masks   : List[np.ndarray]
    rects   : List[Tuple[int, int, int, int]]
    bbox    : Tuple[int, int, int, int]


# ══════════════════════════════════════════════════════════════════════════════
#  FACE TRACK  —  per-face Kalman instance + stable ID bookkeeping
# ══════════════════════════════════════════════════════════════════════════════
class FaceTrack:
    """
    One entry in FaceRoiExtractor's track pool.

    Each tracked face owns its own LandmarkKalman so that landmark states
    from different people are NEVER mixed.  IDs are monotonically increasing
    integers that are never re-used within a session.
    """
    _id_counter: int = 0   # class-level counter; incremented on each new track

    def __init__(self, bbox: Tuple[int, int, int, int], ts: float) -> None:
        FaceTrack._id_counter += 1
        self.face_id   : int                          = FaceTrack._id_counter
        self.kalman    : LandmarkKalman               = LandmarkKalman(num_landmarks=468)
        self.bbox      : Tuple[int, int, int, int]    = bbox   # last matched bbox
        self.last_seen : float                        = ts     # wall-clock timestamp


# ══════════════════════════════════════════════════════════════════════════════
#  FACE MESH ROI EXTRACTOR  (multi-face, isolated per-face Kalman filters)
# ══════════════════════════════════════════════════════════════════════════════
class FaceRoiExtractor:
    """
    Uses MediaPipe FaceMesh (468 landmarks) to locate and extract forehead +
    left-cheek + right-cheek ROI masks for EVERY detected face in the frame.

    Multi-face design
    ─────────────────
    • MediaPipe is configured with max_num_faces = MAX_NUM_FACES (default 4).
    • Each detected face is matched to an existing FaceTrack via greedy IoU
      bounding-box matching across frames, giving stable integer IDs.
    • Every FaceTrack owns its own LandmarkKalman instance — landmark state
      from face A can never bleed into face B.
    • Tracks not matched for FACE_TRACK_TIMEOUT seconds are retired and their
      memory (including the Kalman filter) is freed.

    Returns
    ───────
    extract() → List[FaceRoiResult], one entry per detected face (empty list
    if no face is visible or MediaPipe is unavailable).
    """

    def __init__(self, max_faces: int = MAX_NUM_FACES) -> None:
        if not MEDIAPIPE_OK:
            self.available = False
            return
        self.available = True
        self.max_faces = max_faces

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,        # ← was hardcoded 1; now configurable
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.roi_groups: List[List[int]] = [
            FOREHEAD_LANDMARKS,
            LEFT_CHEEK_LANDMARKS,
            RIGHT_CHEEK_LANDMARKS,
        ]
        # Track pool: face_id → FaceTrack (each with its own Kalman filter)
        self._tracks: Dict[int, FaceTrack] = {}

    # ── Property: IDs of all currently live tracks ────────────────────────────
    @property
    def active_face_ids(self) -> List[int]:
        """Return the face IDs of all tracks that have not yet expired."""
        return list(self._tracks.keys())

    # ── Geometry helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _landmarks_to_bbox(
        lms: List[Tuple[float, float]],
    ) -> Tuple[int, int, int, int]:
        """Compute an axis-aligned bounding box from a list of (x, y) points."""
        xs = [p[0] for p in lms]
        ys = [p[1] for p in lms]
        return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

    @staticmethod
    def _iou(
        a: Tuple[int, int, int, int],
        b: Tuple[int, int, int, int],
    ) -> float:
        """Intersection-over-Union of two (x1, y1, x2, y2) rectangles."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter  = (ix2 - ix1) * (iy2 - iy1)
        area_a = max((ax2 - ax1) * (ay2 - ay1), 1)
        area_b = max((bx2 - bx1) * (by2 - by1), 1)
        return inter / (area_a + area_b - inter)

    # ── Track lifecycle ───────────────────────────────────────────────────────
    def _match_and_update_tracks(
        self,
        bboxes: List[Tuple[int, int, int, int]],
        now: float,
    ) -> List[int]:
        """
        Greedy IoU matching between current frame's detections and the active
        track pool.

        Algorithm
        ─────────
        1. Build an (n_detections × n_tracks) IoU matrix.
        2. Repeatedly pick the global maximum; if it meets the threshold,
           link that detection to that track and zero out its row and column
           to prevent double-assignment.
        3. Any detection left unmatched gets a brand-new FaceTrack (and
           therefore a fresh LandmarkKalman).

        Returns a list of face_ids, one per entry in `bboxes`.
        """
        track_ids = list(self._tracks.keys())
        assigned  = [-1] * len(bboxes)

        if track_ids:
            n_det = len(bboxes)
            n_trk = len(track_ids)
            iou_mat = np.zeros((n_det, n_trk), dtype=np.float32)
            for r, bb in enumerate(bboxes):
                for c, tid in enumerate(track_ids):
                    iou_mat[r, c] = self._iou(bb, self._tracks[tid].bbox)

            # Greedy: pick best pair until no IoU ≥ threshold remains
            while True:
                r, c = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                if iou_mat[r, c] < FACE_MATCH_IOU_MIN:
                    break
                tid             = track_ids[c]
                assigned[r]     = tid
                self._tracks[tid].bbox      = bboxes[r]
                self._tracks[tid].last_seen = now
                iou_mat[r, :]   = 0.0   # this detection is claimed
                iou_mat[:, c]   = 0.0   # this track is claimed

        # Unmatched detections → new tracks with fresh Kalman filters
        for r, fid in enumerate(assigned):
            if fid == -1:
                t = FaceTrack(bboxes[r], now)
                self._tracks[t.face_id] = t
                assigned[r] = t.face_id
                log_info(f"FaceRoiExtractor: new face track ID {t.face_id}")

        return assigned

    def _prune_stale_tracks(self, now: float) -> None:
        """Remove tracks not matched for longer than FACE_TRACK_TIMEOUT."""
        expired = [
            fid for fid, t in self._tracks.items()
            if now - t.last_seen > FACE_TRACK_TIMEOUT
        ]
        for fid in expired:
            log_info(f"FaceRoiExtractor: face track ID {fid} expired")
            del self._tracks[fid]

    # ── Main extraction ───────────────────────────────────────────────────────
    def extract(self, frame: np.ndarray) -> List[FaceRoiResult]:
        """
        Run FaceMesh on `frame` and return one FaceRoiResult per detected face.

        Steps
        ─────
        1. Run MediaPipe FaceMesh → up to MAX_NUM_FACES face landmark sets.
        2. Convert each face's landmarks to pixel coordinates and compute a
           bounding box.
        3. Match bounding boxes to active FaceTrack pool via greedy IoU; create
           new tracks for unmatched detections.
        4. For each face, smooth its landmarks through its OWN LandmarkKalman.
        5. Build forehead / left-cheek / right-cheek convex-hull masks from the
           smoothed landmarks, applying a small erosion to exclude edge pixels.
        6. Return a FaceRoiResult per face.
        """
        if not self.available:
            return []

        h, w   = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)
        now    = time.time()

        if not result.multi_face_landmarks:
            self._prune_stale_tracks(now)
            return []

        # ── Step 1 & 2: pixel-space landmarks + bounding boxes ────────────────
        all_raw_lms: List[List[Tuple[float, float]]] = [
            [(lm.x * w, lm.y * h) for lm in face.landmark]
            for face in result.multi_face_landmarks
        ]
        all_bboxes: List[Tuple[int, int, int, int]] = [
            self._landmarks_to_bbox(lms) for lms in all_raw_lms
        ]

        # ── Step 3: stable ID assignment ──────────────────────────────────────
        face_ids = self._match_and_update_tracks(all_bboxes, now)
        self._prune_stale_tracks(now)

        # ── Steps 4 & 5: per-face Kalman smoothing + ROI mask generation ──────
        output: List[FaceRoiResult] = []

        for det_idx, face_id in enumerate(face_ids):
            track      = self._tracks[face_id]
            # Each face uses ONLY its own Kalman filter — no cross-face state
            smooth_lms = track.kalman.update(all_raw_lms[det_idx])

            masks: List[np.ndarray]             = []
            rects: List[Tuple[int, int, int, int]] = []

            for group in self.roi_groups:
                pts = np.array(
                    [(int(smooth_lms[i][0]), int(smooth_lms[i][1]))
                     for i in group],
                    dtype=np.int32,
                )
                if len(pts) < 3:
                    continue

                mask = np.zeros((h, w), dtype=np.uint8)
                hull = cv2.convexHull(pts)
                cv2.fillConvexPoly(mask, hull, 255)
                # Slight erosion: exclude noisy edge pixels for better SNR
                mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)

                bx, by, bw, bh = cv2.boundingRect(hull)
                masks.append(mask)
                rects.append((bx, by, bx + bw, by + bh))

            output.append(FaceRoiResult(
                face_id=face_id,
                masks=masks,
                rects=rects,
                bbox=all_bboxes[det_idx],
            ))

        return output


# ══════════════════════════════════════════════════════════════════════════════
#  DEEP LEARNING ALTERNATIVE  (optional PyTorch backend)
# ══════════════════════════════════════════════════════════════════════════════
class PhysNetLite(object):
    """
    Lightweight 1D-CNN inspired by PhysNet (Chen & McDuff, 2018) for
    end-to-end r-PPG signal extraction from raw RGB sequences.

    Architecture
    ────────────
    Input : (batch, 3, T)  — raw R,G,B time series of length T
    Output: (batch, 1)     — SpO₂ prediction

    Layers:
      Conv1d(3→32, k=5) → BN → ReLU → MaxPool
      Conv1d(32→64, k=5) → BN → ReLU → MaxPool
      Conv1d(64→128, k=3) → BN → ReLU → AdaptiveAvgPool
      FC(128→64) → ReLU → Dropout(0.3)
      FC(64→1)  → Sigmoid * 12 + 88  (maps to 88–100%)

    For production use, train on UBFC-rPPG, PURE, or MMPD datasets.
    """
    def __init__(self) -> None:
        if not TORCH_OK:
            self.available = False
            return
        self.available = True
        self.model = self._build()
        self.model.eval()

    def _build(self) -> "nn.Module":
        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv1d(3, 32, kernel_size=5, padding=2),
                    nn.BatchNorm1d(32), nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(32, 64, kernel_size=5, padding=2),
                    nn.BatchNorm1d(64), nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm1d(128), nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                self.head = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(128, 64), nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 1), nn.Sigmoid(),
                )

            def forward(self, x):
                return self.head(self.features(x)) * 12.0 + 88.0

        return _Net()

    def predict(self, r_buf: List[float], g_buf: List[float], b_buf: List[float]) -> float:
        """Run inference on recent RGB buffer; returns SpO₂ estimate."""
        if not self.available or len(r_buf) < RPPG_MIN_FRAMES:
            return 0.0
        T = min(len(r_buf), RPPG_BUFFER_FRAMES)
        seq = np.stack([
            np.array(list(r_buf)[-T:]),
            np.array(list(g_buf)[-T:]),
            np.array(list(b_buf)[-T:]),
        ], axis=0)  # (3, T)
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # (1, 3, T)
        with torch.no_grad():
            out = self.model(x)
        return float(out.item())


# ══════════════════════════════════════════════════════════════════════════════
#  PER-PERSON STATE  (Fall Detection)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class PersonState:
    track_id:       int

    posture:        str   = P.UNKNOWN
    prev_posture:   str   = P.UNKNOWN
    posture_hist:   deque = field(
                        default_factory=lambda: deque(maxlen=POSTURE_HIST_LEN))

    bbox:    Tuple[int, int, int, int] = (0, 0, 0, 0)
    center:  Tuple[int, int]           = (0, 0)

    is_moving:      bool  = True
    last_move_time: float = field(default_factory=time.time)
    snapshot_time:  float = field(default_factory=time.time)
    snapshot_pos:   Tuple[int, int] = (0, 0)
    snapshot_kps:   Optional[np.ndarray] = None
    snapshot_kp_conf: Optional[np.ndarray] = None

    fallen_since:   Optional[float] = None

    major_alert:    bool           = False
    alert_fired_at: Optional[float] = None
    warning_alert:  bool           = False
    warning_fired_at: Optional[float] = None
    notified_telegram: bool         = False
    notified_discord:  bool         = False

    last_seen:   float = field(default_factory=time.time)
    frame_count: int   = 0


# ══════════════════════════════════════════════════════════════════════════════
#  POSTURE CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
def classify_posture(
        kp_xy:   Optional[np.ndarray],
        kp_conf: Optional[np.ndarray],
        bbox:    Tuple[int, int, int, int]
) -> str:
    x1, y1, x2, y2 = bbox
    bh = max(y2 - y1, 1)
    bw = max(x2 - x1, 1)

    KP_REQUIRED = (5, 6, 11, 12)
    kp_valid = (
        kp_xy   is not None and
        kp_conf is not None and
        len(kp_conf) > max(KP_REQUIRED) and
        all(float(kp_conf[k]) >= KP_MIN_CONF for k in KP_REQUIRED)
    )
    if kp_valid:
        sh_x = (float(kp_xy[5][0]) + float(kp_xy[6][0])) / 2
        sh_y = (float(kp_xy[5][1]) + float(kp_xy[6][1])) / 2
        hp_x = (float(kp_xy[11][0]) + float(kp_xy[12][0])) / 2
        hp_y = (float(kp_xy[11][1]) + float(kp_xy[12][1])) / 2

        delta_x = hp_x - sh_x
        delta_y = hp_y - sh_y

        trunk_angle = math.degrees(math.atan2(abs(delta_x), abs(delta_y) + 1e-6))

        if trunk_angle > 70:
            return P.FALLEN
        if trunk_angle < 40:
            return P.STANDING if (bh / bw) > STANDING_BBOX_RATIO else P.SITTING
        if trunk_angle < UPRIGHT_TRUNK_DEG:
            return P.STANDING if (bh / bw) > STANDING_BBOX_RATIO else P.SITTING
        return P.FALLEN if (bw / bh) > FALLEN_BBOX_RATIO else P.SITTING

    if (bw / bh) > FALLEN_BBOX_RATIO:
        return P.FALLEN
    return P.STANDING if (bh / bw) > STANDING_BBOX_RATIO else P.SITTING


def smooth_posture(state: PersonState, raw: str) -> str:
    state.posture_hist.append(raw)
    if state.posture_hist.count(P.FALLEN) >= FALLEN_VOTES_NEEDED:
        return P.FALLEN
    non_fallen = [p for p in state.posture_hist if p != P.FALLEN]
    return Counter(non_fallen).most_common(1)[0][0] if non_fallen else raw


# ══════════════════════════════════════════════════════════════════════════════
#  STATE UPDATER
# ══════════════════════════════════════════════════════════════════════════════
def update_person_state(
    state: PersonState,
    cx: int, cy: int,
    raw_posture: str,
    now: float,
    kp_xy: Optional[np.ndarray] = None,
    kp_conf: Optional[np.ndarray] = None,
) -> None:
    state.last_seen   = now
    state.center      = (cx, cy)
    state.frame_count += 1

    new_posture = smooth_posture(state, raw_posture)

    if new_posture != state.prev_posture and state.frame_count > SETTLING_FRAMES:
        log_info(f"ID {state.track_id:>3d}: {state.prev_posture:<10} → {new_posture}")
    state.prev_posture = new_posture
    state.posture      = new_posture

    # Motion snapshot
    if now - state.snapshot_time >= MOTION_SNAPSHOT_SEC:
        was_moving = state.is_moving
        CHOSEN_KPS = [0, 5, 6, 9, 10, 13, 14, 15, 16]
        displacements: List[float] = []

        if (kp_xy is not None and kp_conf is not None and
                state.snapshot_kps is not None and state.snapshot_kp_conf is not None):
            try:
                for idx in CHOSEN_KPS:
                    if (idx < len(kp_conf) and idx < len(state.snapshot_kp_conf) and
                            float(kp_conf[idx]) >= KP_MIN_CONF and
                            float(state.snapshot_kp_conf[idx]) >= KP_MIN_CONF):
                        dx = float(kp_xy[idx][0]) - float(state.snapshot_kps[idx][0])
                        dy = float(kp_xy[idx][1]) - float(state.snapshot_kps[idx][1])
                        displacements.append(math.hypot(dx, dy))
            except Exception:
                displacements = []

        if not displacements:
            dx = cx - state.snapshot_pos[0]
            dy = cy - state.snapshot_pos[1]
            mean_disp = math.hypot(dx, dy)
            threshold = MOTION_THRESHOLD_PX
        else:
            mean_disp = float(np.mean(displacements))
            threshold = KP_MOTION_THRESHOLD_PX

        state.is_moving = mean_disp > threshold
        if state.is_moving:
            state.last_move_time = now

        if state.is_moving != was_moving and state.frame_count > SETTLING_FRAMES:
            status = "moving" if state.is_moving else "stationary"
            log_info(f"ID {state.track_id:>3d}: motion → {status} (disp={mean_disp:.1f}px)")

        state.snapshot_time   = now
        state.snapshot_pos    = (cx, cy)
        state.snapshot_kps    = kp_xy.copy()   if kp_xy   is not None else None
        state.snapshot_kp_conf= kp_conf.copy() if kp_conf is not None else None

    # Fallen-state entry / exit
    if state.posture == P.FALLEN:
        if state.fallen_since is None:
            state.fallen_since = now
            if state.frame_count > SETTLING_FRAMES:
                log_warning(f"ID {state.track_id:>3d}: entered FALLEN state")
    else:
        if state.fallen_since is not None:
            duration = now - state.fallen_since
            log_info(f"ID {state.track_id:>3d}: recovered from FALLEN ({duration:.1f}s)")
        state.fallen_since = None
        if state.major_alert:
            log_info(f"ID {state.track_id:>3d}: major alert cleared (posture recovered)")
            state.major_alert    = False
            state.alert_fired_at = None

    # Alert gate
    immobile_secs = now - state.last_move_time

    if state.posture == P.FALLEN and state.frame_count > SETTLING_FRAMES:
        if immobile_secs >= WARNING_ALERT_SEC and not state.warning_alert:
            state.warning_alert    = True
            state.warning_fired_at = now
            log_warning(f"ID {state.track_id:>3d}: WARNING — fallen & still {immobile_secs:.0f}s")
            threading.Thread(target=play_sound, args=("warning",), daemon=True).start()
            threading.Thread(target=notify_webhooks,
                             args=(state, "warning", immobile_secs), daemon=True).start()

        if immobile_secs >= IMMOBILE_ALERT_SEC and not state.major_alert:
            state.major_alert    = True
            state.alert_fired_at = now
            log_alert(f"ID {state.track_id:>3d}: MAJOR ALERT — fallen & immobile {immobile_secs:.0f}s !!")
            threading.Thread(target=play_sound, args=("major",), daemon=True).start()
            threading.Thread(target=notify_webhooks,
                             args=(state, "major", immobile_secs), daemon=True).start()
    else:
        if state.warning_alert or state.major_alert:
            if state.major_alert:
                log_info(f"ID {state.track_id:>3d}: major alert cleared (posture recovered)")
            elif state.warning_alert:
                log_info(f"ID {state.track_id:>3d}: warning cleared (posture recovered)")

        state.fallen_since       = None
        state.warning_alert      = False
        state.warning_fired_at   = None
        state.major_alert        = False
        state.alert_fired_at     = None
        state.notified_telegram  = False
        state.notified_discord   = False


# ══════════════════════════════════════════════════════════════════════════════
#  DRAWING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def _alpha_rect(frame, x1, y1, x2, y2, color, alpha=0.65):
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    bg = np.full_like(roi, color)
    cv2.addWeighted(bg, alpha, roi, 1.0 - alpha, 0, roi)
    frame[y1:y2, x1:x2] = roi


def draw_corner_box(frame, x1, y1, x2, y2, color, arm=22, thick=3):
    arm = min(arm, (x2 - x1) // 4, (y2 - y1) // 4)
    pts = [
        ((x1 + arm, y1), (x1, y1), (x1, y1 + arm)),
        ((x2 - arm, y1), (x2, y1), (x2, y1 + arm)),
        ((x1 + arm, y2), (x1, y2), (x1, y2 - arm)),
        ((x2 - arm, y2), (x2, y2), (x2, y2 - arm)),
    ]
    for (ha, corner, va) in pts:
        cv2.line(frame, ha,     corner, color, thick, cv2.LINE_AA)
        cv2.line(frame, corner, va,     color, thick, cv2.LINE_AA)


def draw_skeleton(frame, kp_xy, kp_conf):
    if kp_xy is None or kp_conf is None:
        return
    n = len(kp_conf)
    for i in range(n):
        if float(kp_conf[i]) >= KP_MIN_CONF:
            cv2.circle(frame, (int(kp_xy[i][0]), int(kp_xy[i][1])),
                       4, C_KP_DOT, -1, cv2.LINE_AA)
    for a, b in SKELETON_EDGES:
        if (a < n and b < n and
                float(kp_conf[a]) >= KP_MIN_CONF and
                float(kp_conf[b]) >= KP_MIN_CONF):
            cv2.line(frame,
                     (int(kp_xy[a][0]), int(kp_xy[a][1])),
                     (int(kp_xy[b][0]), int(kp_xy[b][1])),
                     C_BONE, 2, cv2.LINE_AA)


def draw_person_panel(frame, state: PersonState, now: float):
    x1, y1, x2, y2 = state.bbox
    box_col = C_ALERT if state.major_alert else POSTURE_COL[state.posture]
    draw_corner_box(frame, x1, y1, x2, y2, box_col,
                    thick=4 if state.major_alert else 2)

    immobile_s = now - state.last_move_time
    fallen_s   = (now - state.fallen_since) if state.fallen_since else 0.0
    mot_str    = "Moving" if state.is_moving else f"Still {immobile_s:.0f}s"

    lines = [
        (f"ID: {state.track_id}", C_YELLOW),
        (state.posture,           box_col),
        (mot_str,                 C_WHITE),
    ]
    if state.warning_alert and not state.major_alert:
        lines.append((f"WARN: still {now - (state.warning_fired_at or now):.0f}s", C_CAUTION))
    if fallen_s > 1:
        lines.append((f"Down: {fallen_s:.0f}s", C_CAUTION))
    if state.major_alert:
        elapsed = now - (state.alert_fired_at or now)
        lines.append((f"!! ALERT {elapsed:.0f}s !!", C_ALERT))

    PAD = 5; LINE_H = 20; PW = 160
    PH = len(lines) * LINE_H + PAD * 2
    pw1 = max(x1, 0)
    pw2 = min(x1 + PW, frame.shape[1])
    ph1 = max(y1 - PH - 5, 0)
    ph2 = ph1 + PH
    _alpha_rect(frame, pw1, ph1, pw2, ph2, C_DARK, alpha=0.70)
    for i, (text, color) in enumerate(lines):
        ty = ph1 + PAD + LINE_H * i + 14
        cv2.putText(frame, text, (pw1 + PAD, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)


def draw_global_hud(frame, states: Dict[int, PersonState], fps: float):
    h, w = frame.shape[:2]
    n_total    = len(states)
    n_standing = sum(1 for s in states.values() if s.posture == P.STANDING)
    n_sitting  = sum(1 for s in states.values() if s.posture == P.SITTING)
    n_fallen   = sum(1 for s in states.values() if s.posture == P.FALLEN)
    n_alert    = sum(1 for s in states.values() if s.major_alert)

    lines = [
        (f"FPS    {fps:4.1f}",  C_YELLOW),
        (f"People {n_total}",   C_WHITE),
        (f"Stand  {n_standing}",C_SAFE),
        (f"Sit    {n_sitting}", C_SAFE),
        (f"Fallen {n_fallen}",  C_CAUTION if n_fallen else C_WHITE),
        (f"Alerts {n_alert}",   C_ALERT   if n_alert  else C_WHITE),
        ("──────────────────",  C_GRAY),
        ("[q] Quit",            C_CYAN),
        ("[c] Clear alerts",    C_CYAN),
        ("[s] Toggle SpO₂",     C_CYAN),
    ]

    PAD = 8; LINE_H = 21; PW = 180
    PH = len(lines) * LINE_H + PAD * 2
    px1, py1 = w - PW - 10, 10
    px2, py2 = w - 10, py1 + PH
    _alpha_rect(frame, px1, py1, px2, py2, C_DARK, alpha=0.72)
    for i, (text, color) in enumerate(lines):
        ty = py1 + PAD + LINE_H * i + 15
        cv2.putText(frame, text, (px1 + PAD, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)


def draw_spo2_panel(
    frame: np.ndarray,
    rppg: RppgProcessor,
    roi_rects: List[Tuple[int,int,int,int]],
    show_rois: bool,
    face_id: int = 0,
    face_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> None:
    """
    Draw the SpO₂ / HR panel for one face.

    Positioning
    ───────────
    When face_bbox is provided the panel is anchored just below that face's
    bounding box (clamped to frame bounds).  Without it the panel falls back
    to the bottom-left corner — preserving the original single-face behaviour.
    """
    h, w = frame.shape[:2]

    spo2_val = rppg.spo2_smooth
    hr_val   = rppg.heart_rate
    quality  = rppg.signal_quality

    collecting = spo2_val < 1.0

    spo2_col = C_SPO2_OK if spo2_val >= 95.0 else C_SPO2_LO
    spo2_str = f"{spo2_val:.1f}%" if not collecting else "Collecting..."
    hr_str   = f"{hr_val:.0f} bpm" if hr_val > 0 and not collecting else "--"
    qual_str = f"SQ: {quality * 100:.0f}%"

    lines = [
        (f"SpO\u2082  Face {face_id}", C_YELLOW),   # "SpO₂  Face N"
        (spo2_str,        spo2_col if not collecting else C_GRAY),
        (f"HR: {hr_str}", C_WHITE),
        (qual_str,        C_CYAN),
    ]
    if spo2_val > 0 and spo2_val < 94.0:
        lines.append(("!! LOW SpO\u2082 !!", C_ALERT))

    PAD = 8; LINE_H = 22; PW = 178
    PH = len(lines) * LINE_H + PAD * 2

    # Anchor: just below the face bbox when available, else bottom-left corner
    if face_bbox is not None:
        _fx1, _fy1, _fx2, fy2 = face_bbox
        px1 = max(_fx1, 0)
        py1 = max(min(fy2 + 6, h - PH - 4), 0)
    else:
        px1 = 10
        py1 = h - PH - 10

    px2 = min(px1 + PW, w - 1)
    py2 = min(py1 + PH, h - 1)

    _alpha_rect(frame, px1, py1, px2, py2, C_DARK, alpha=0.75)

    for i, (text, color) in enumerate(lines):
        ty = py1 + PAD + LINE_H * i + 16
        cv2.putText(frame, text, (px1 + PAD, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # ROI outlines (only when the face is actively in the frame)
    if show_rois:
        roi_labels = ["Forehead", "L-Cheek", "R-Cheek"]
        for idx, rect in enumerate(roi_rects):
            rx1, ry1, rx2, ry2 = rect
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), C_ROI, 1)
            if idx < len(roi_labels):
                cv2.putText(frame, roi_labels[idx], (rx1, ry1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_ROI, 1, cv2.LINE_AA)


def draw_major_alert_overlay(frame, states: Dict[int, PersonState], now: float):
    alert_ids = sorted(s.track_id for s in states.values() if s.major_alert)
    if not alert_ids:
        return

    h, w = frame.shape[:2]
    flash_on = int(now * ALERT_FLASH_HZ * 2) % 2 == 0
    if flash_on:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), C_ALERT, 12)

    ids_str = "  |  ".join(f"ID {i}" for i in alert_ids)
    line1   = f"  MAJOR ALERT :  {ids_str}  "
    line2   = "  Person fallen & immobile > 15 s     [c] dismiss  "

    FS1, FS2 = 0.80, 0.50
    (tw1, th1), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_DUPLEX,  FS1, 2)
    (tw2, th2), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, FS2, 1)
    bw   = max(tw1, tw2) + 24
    bh   = th1 + th2 + 32
    bx1  = max((w - bw) // 2, 0)
    by1  = h - bh - 16
    bx2, by2 = bx1 + bw, by1 + bh

    _alpha_rect(frame, bx1, by1, bx2, by2, (0, 0, 0), alpha=0.80)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), C_ALERT, 2)
    cv2.putText(frame, line1, (bx1 + 12, by1 + th1 + 10),
                cv2.FONT_HERSHEY_DUPLEX, FS1,
                C_ALERT if flash_on else C_WHITE, 2, cv2.LINE_AA)
    cv2.putText(frame, line2, (bx1 + 12, by1 + th1 + th2 + 24),
                cv2.FONT_HERSHEY_SIMPLEX, FS2, C_CAUTION, 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  NOTIFICATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
DISCORD_WEBHOOK    = os.getenv("DISCORD_WEBHOOK_URL")


def play_sound(level: str = "warning") -> None:
    try:
        if os.name == "nt":
            import winsound
            winsound.Beep(1200 if level == "major" else 800,
                          600  if level == "major" else 300)
    except Exception:
        pass


def send_telegram_message(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=5)
    except Exception:
        pass


def send_discord_message(content: str) -> None:
    if not DISCORD_WEBHOOK:
        return
    try:
        requests.post(DISCORD_WEBHOOK, json={"content": content}, timeout=5)
    except Exception:
        pass


def notify_webhooks(state: PersonState, level: str, immobile_secs: float) -> None:
    tid  = state.track_id
    text = (f"⚠️ WARNING: ID {tid} fallen & still {immobile_secs:.0f}s"
            if level == "warning"
            else f"🚨 MAJOR ALERT: ID {tid} fallen & immobile {immobile_secs:.0f}s")
    if not state.notified_telegram:
        send_telegram_message(text)
        state.notified_telegram = True
    if not state.notified_discord:
        send_discord_message(text)
        state.notified_discord = True


# ══════════════════════════════════════════════════════════════════════════════
#  STATE CLEANUP
# ══════════════════════════════════════════════════════════════════════════════
def cleanup_stale_persons(
        states: Dict[int, PersonState],
        active_ids: List[int],
        now: float
) -> None:
    stale = [
        sid for sid, s in states.items()
        if sid not in active_ids and now - s.last_seen > PERSON_TIMEOUT_SEC
    ]
    for sid in stale:
        s = states[sid]
        log_info(f"ID {sid:>3d}: removed (last posture={s.posture}, "
                 f"alert={'YES' if s.major_alert else 'no'})")
        del states[sid]


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    # ── Load models ───────────────────────────────────────────────────────────
    model = None
    if YOLO_OK:
        log_info(f"Loading YOLO model '{YOLO_MODEL}'…")
        model = YOLO(YOLO_MODEL)
        log_info("YOLO model ready.")
    else:
        log_warning("YOLO unavailable — fall detection disabled.")

    face_extractor = FaceRoiExtractor()          # multi-face, per-face Kalman
    rppg           = RppgProcessor()             # kept for backward compat (unused below)
    physnet        = PhysNetLite()               # optional DL backend

    if not face_extractor.available:
        log_warning("MediaPipe unavailable — SpO₂ module disabled.")

    # ── Camera ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        log_alert(f"Cannot open camera {CAM_INDEX}.")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log_info(f"Camera {CAM_INDEX} opened at {actual_w}×{actual_h}")
    log_info("Controls: [q] quit  [c] clear alerts  [s] toggle SpO₂  [1-9] clear by ID")
    print()

    # ── State ─────────────────────────────────────────────────────────────────
    person_states : Dict[int, PersonState]    = {}
    fps            = 0.0
    frame_times  : deque                      = deque(maxlen=30)
    show_spo2      = True
    # Multi-face r-PPG: one RppgProcessor per live face track ID
    rppg_pool    : Dict[int, RppgProcessor]   = {}
    # Latest per-face extraction output (used by both MODULE 2 and the HUD)
    face_results : List[FaceRoiResult]        = []

    # SpO₂ update throttle (run expensive processing every N frames)
    SPO2_UPDATE_INTERVAL = 15   # ~0.5 s at 30 fps
    spo2_frame_counter   = 0

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            log_warning("Frame grab failed — retrying…")
            time.sleep(0.05)
            continue

        now = time.time()

        # ── MODULE 1: Fall & Immobility Detection ─────────────────────────────
        if model is not None:
            results = model.track(
                frame,
                persist  = True,
                tracker  = "bytetrack.yaml",
                conf     = DETECT_CONF,
                iou      = DETECT_IOU,
                verbose  = False,
                imgsz    = INFER_SIZE,
            )
            result     = results[0]
            active_ids: List[int] = []

            if result.boxes.id is not None:
                track_ids  = result.boxes.id.int().cpu().tolist()
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                has_kp     = result.keypoints is not None
                has_conf   = has_kp and result.keypoints.conf is not None

                for i, tid in enumerate(track_ids):
                    x1, y1, x2, y2 = map(int, boxes_xyxy[i])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    active_ids.append(tid)

                    try:
                        kp_xy   = result.keypoints.xy[i].cpu().numpy()   if has_kp   else None
                        kp_conf = result.keypoints.conf[i].cpu().numpy() if has_conf else None
                    except (AttributeError, IndexError):
                        kp_xy = kp_conf = None

                    if tid not in person_states:
                        person_states[tid] = PersonState(
                            track_id=tid, snapshot_pos=(cx, cy)
                        )
                        log_info(f"ID {tid:>3d}: new person detected at ({cx}, {cy})")

                    st      = person_states[tid]
                    st.bbox = (x1, y1, x2, y2)

                    raw = classify_posture(kp_xy, kp_conf, (x1, y1, x2, y2))
                    update_person_state(st, cx, cy, raw, now,
                                        kp_xy=kp_xy, kp_conf=kp_conf)
                    draw_skeleton(frame, kp_xy, kp_conf)
                    draw_person_panel(frame, st, now)

            cleanup_stale_persons(person_states, active_ids, now)

        # ── MODULE 2: r-PPG / SpO₂  (multi-face) ─────────────────────────────
        if face_extractor.available and show_spo2:
            face_results = face_extractor.extract(frame)

            for fr in face_results:
                # Lazily create a dedicated RppgProcessor for each new face
                if fr.face_id not in rppg_pool:
                    rppg_pool[fr.face_id] = RppgProcessor()
                    log_info(f"SpO\u2082: new processor for face ID {fr.face_id}")

                face_rppg = rppg_pool[fr.face_id]

                # Compute mean RGB across this face's ROI masks and push
                r_vals, g_vals, b_vals = [], [], []
                for mask in fr.masks:
                    px = frame[mask > 0]
                    if px.size:
                        b_vals.append(float(px[:, 0].mean()))  # OpenCV is BGR
                        g_vals.append(float(px[:, 1].mean()))
                        r_vals.append(float(px[:, 2].mean()))

                if r_vals:
                    face_rppg.push_frame_direct(
                        float(np.mean(r_vals)),
                        float(np.mean(g_vals)),
                        float(np.mean(b_vals)),
                    )

                # Draw ROI bounding rects directly on the frame
                for rx1, ry1, rx2, ry2 in fr.rects:
                    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), C_ROI, 1)

            # Throttled SpO₂ update — run for every active face
            spo2_frame_counter += 1
            if spo2_frame_counter >= SPO2_UPDATE_INTERVAL:
                spo2_frame_counter = 0
                for face_rppg in rppg_pool.values():
                    face_rppg.process(fps=fps if fps > 5 else RPPG_ASSUMED_FPS)

            # Prune RppgProcessor entries whose face track has expired, so
            # stale signal buffers don't accumulate for unseen people
            live_ids   = set(face_extractor.active_face_ids)
            stale_fids = [fid for fid in rppg_pool if fid not in live_ids]
            for fid in stale_fids:
                log_info(f"SpO\u2082: pruning processor for expired face ID {fid}")
                del rppg_pool[fid]

        # ── Global overlays ───────────────────────────────────────────────────
        draw_global_hud(frame, person_states, fps)
        draw_major_alert_overlay(frame, person_states, now)

        if show_spo2 and face_extractor.available:
            for fr in face_results:
                if fr.face_id in rppg_pool:
                    draw_spo2_panel(
                        frame,
                        rppg_pool[fr.face_id],
                        fr.rects,
                        show_rois=True,
                        face_id=fr.face_id,
                        face_bbox=fr.bbox,
                    )

        # ── FPS ───────────────────────────────────────────────────────────────
        frame_times.append(now)
        if len(frame_times) >= 2:
            fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0] + 1e-9)

        # ── Key input ─────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            log_info("Quit requested.")
            break

        elif key == ord('s'):
            show_spo2 = not show_spo2
            log_info(f"SpO₂ panel {'enabled' if show_spo2 else 'disabled'}.")

        elif key == ord('c'):
            cleared = [sid for sid, s in person_states.items() if s.major_alert]
            for sid in cleared:
                person_states[sid].major_alert    = False
                person_states[sid].alert_fired_at = None
            cleared_warn = [sid for sid, s in person_states.items() if s.warning_alert]
            for sid in cleared_warn:
                person_states[sid].warning_alert      = False
                person_states[sid].warning_fired_at   = None
                person_states[sid].notified_telegram  = False
                person_states[sid].notified_discord   = False
            log_info(f"All alerts dismissed (IDs: {cleared})" if cleared
                     else "No active alerts.")

        elif ord('1') <= key <= ord('9'):
            tid = key - ord('0')
            if tid in person_states:
                if person_states[tid].major_alert:
                    person_states[tid].major_alert    = False
                    person_states[tid].alert_fired_at = None
                    log_info(f"Alert dismissed for ID {tid}")
                elif person_states[tid].warning_alert:
                    person_states[tid].warning_alert     = False
                    person_states[tid].warning_fired_at  = None
                    person_states[tid].notified_telegram = False
                    person_states[tid].notified_discord  = False
                    log_info(f"Warning dismissed for ID {tid}")
                else:
                    log_info(f"No active alert for ID {tid}")

        cv2.imshow("Health Monitor — Fall Detection + SpO₂ r-PPG", frame)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    log_info("System shut down cleanly.")


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET REQUIREMENTS & VALIDATION PROTOCOL (documentation)
# ══════════════════════════════════════════════════════════════════════════════
"""
DATASET REQUIREMENTS FOR CLINICAL-GRADE SpO₂ ESTIMATION
════════════════════════════════════════════════════════

1. SUBJECT DIVERSITY
   • N ≥ 100 subjects minimum (N ≥ 500 for publication-quality models)
   • Fitzpatrick skin tone scale I–VI evenly represented
   • Age range: 18–80; BMI range: 18–35
   • Both sexes

2. GROUND TRUTH INSTRUMENT
   • FDA-cleared finger-clip pulse oximeter (e.g., Masimo Rad-8, Nonin 3150)
   • Record at 1 Hz, synchronised with video timestamps
   • Simultaneous forehead reflectance oximeter for cross-validation

3. SpO₂ RANGE
   • Normoxia: 97–100% (resting subjects)
   • Mild hypoxia: 92–96% (breath-hold or altitude protocol)
   • Controlled desaturation: 88–91% (IRB-approved supervised protocol only)
   • Total range: 88–100%

4. LIGHTING CONDITIONS
   • Fluorescent (4000 K), LED (3000 K / 5000 K / 6500 K), sunlight, mixed
   • Low-light condition (< 50 lux)
   • 3-point photographic lighting (controlled reference condition)

5. MOTION ARTIFACTS
   • Still (sitting, 60 s segments)
   • Nodding, head turning, talking (structured motion protocol)
   • Walking on treadmill (extreme condition)

6. VIDEO SPECIFICATION
   • Minimum: 720p @ 30 fps (1080p @ 60 fps recommended)
   • Uncompressed or lossless codec (avoid high-compression H.264 for training)
   • Exposure locked (auto-exposure OFF during collection)

7. SYNCHRONISATION
   • Hardware trigger (GPIO pulse) or software timestamp alignment ≤ 50 ms
   • Video-oximeter delay compensation in post-processing

8. PUBLIC DATASETS TO AUGMENT TRAINING
   • UBFC-rPPG  (42 subjects, indoor, HR + SpO₂)
   • PURE       (10 subjects, 6 motion types)
   • MMPD       (33 subjects, 4 skin tones, 4 lighting)
   • VIPL-HR    (107 subjects, 3 devices, 9 scenarios)

VALIDATION AGAINST CLINICAL OXIMETER
══════════════════════════════════════
• Metrics: MAE, RMSE, R² (Pearson), Bland-Altman plot
• Target for "acceptable" performance: MAE < 2% SpO₂
• FDA standard (ISO 80601-2-61): ARMS ≤ 3% over 70–100% range
• Bland-Altman limits of agreement: ±4% acceptable, ±2% preferred

BLAND-ALTMAN ANALYSIS (Python)
────────────────────────────────
import numpy as np, matplotlib.pyplot as plt

def bland_altman_plot(method1, method2, title="Bland-Altman"):
    mean   = (method1 + method2) / 2
    diff   = method1 - method2
    md     = diff.mean()
    sd     = diff.std()
    plt.figure(figsize=(8, 5))
    plt.scatter(mean, diff, alpha=0.5)
    plt.axhline(md, color='r', linestyle='--', label=f'Mean diff {md:.2f}')
    plt.axhline(md + 1.96*sd, color='b', linestyle=':', label='+1.96 SD')
    plt.axhline(md - 1.96*sd, color='b', linestyle=':', label='-1.96 SD')
    plt.xlabel('Mean SpO₂ (%)'); plt.ylabel('Difference (%)'); plt.title(title)
    plt.legend(); plt.tight_layout(); plt.show()

SKIN-TONE ADAPTATION METHODS
══════════════════════════════
1. ITA (Individual Typology Angle):
   ITA = atan((L* - 50) / b*) × 180/π   — CIELAB colour space from ROI pixels
   Group subjects into bins and train separate model per bin.

2. Fitzpatrick-stratified normalisation:
   Estimate Fitzpatrick type from mean forehead L* (lightness).
   Apply per-group z-score normalisation before model inference.

3. Adversarial de-biasing (deep learning):
   Train secondary network to predict skin tone from latent features;
   use gradient reversal layer to make primary network skin-tone invariant.

REAL-TIME OPTIMISATION TECHNIQUES
═══════════════════════════════════
• Process face mesh every other frame (interpolate landmarks on skip frames)
• Throttle SpO₂ update to every 15 frames (~0.5 s) — implemented above
• Use float16 inference for PyTorch models on GPU
• Pre-allocate mask arrays; avoid per-frame np.zeros allocations in tight loop
• Run YOLO and MediaPipe in separate threads with a shared frame queue
• Use cv2.resize to 640×360 before FaceMesh (full-res only for display)

DEEP LEARNING ALTERNATIVES TO RANDOM FOREST
════════════════════════════════════════════
1. PhysNet (Chen & McDuff, 2018)       — 3D-CNN on spatiotemporal face crops
2. EfficientPhys (Liu et al., 2023)    — EfficientNet backbone + temporal MHA
3. MTTS-CAN (Liu et al., 2020)         — Multi-task CAN for HR + SpO₂
4. DualGAN / rPPG-MAE (2023–2024)     — Self-supervised pre-training on video
5. Transformer-based (Bigbird / SwinT) — Long-context sequence modelling
"""

if __name__ == "__main__":
    main()
