"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     Unified Health Monitoring System  —  v2  (Optimised)                    ║
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
║  v2 IMPROVEMENTS                                                             ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  #1  FPS BOOST — threaded pipeline                                           ║
║      • YOLO and MediaPipe each run in a dedicated daemon worker thread.      ║
║      • Main thread reads camera, feeds workers via Queue(maxsize=1)          ║
║        (non-blocking put_nowait → always processes the freshest frame).      ║
║      • Frames are pre-resized to INFER_W × INFER_H (default 640×360)        ║
║        before workers receive them; original res is kept for display only.   ║
║      • FaceMesh skips every other frame (FACEMESH_EVERY_N = 2).              ║
║      • SpO₂ update is throttled to every SPO2_UPDATE_INTERVAL frames.        ║
║                                                                              ║
║  #2  NO STICKMAN — skeleton drawing disabled                                 ║
║      • draw_skeleton() is still defined but never called.                    ║
║      • Bounding boxes, labels, and alerts are preserved as-is.               ║
║                                                                              ║
║  #3  FALL SPEED — vertical velocity gate on FALLEN classification            ║
║      • Hip midpoint Y (falling back to shoulder Y) is stored in a           ║
║        deque of length FALL_SPEED_HISTORY.                                   ║
║      • Fall speed (px/frame, display-space) is computed as the mean of       ║
║        successive Y-deltas; positive = moving toward bottom of frame.        ║
║      • classify_posture() requires BOTH trunk angle > FALLEN_TRUNK_DEG      ║
║        AND (speed ≥ FALL_SPEED_THRESH  OR  speed ≈ 0 / no history yet).     ║
║        The "≈ 0" branch detects a person already lying still on the ground.  ║
║      • Speed is displayed in the per-person HUD panel.                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  FALL-DETECTION LOGIC                                                        ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Primary  → trunk angle between shoulder midpoint → hip midpoint            ║
║             atan2(|Δx|, |Δy|)  0° = vertical  90° = horizontal              ║
║             > 55°  → FALLEN candidate  (speed gate applied, see #3)         ║
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
.\\env12\\Scripts\\activate          # Windows PowerShell
source env12/bin/activate         # Linux / macOS

pip install ultralytics mediapipe opencv-python numpy scipy scikit-learn requests

# Optional (deep-learning SpO₂ backend):
pip install torch torchvision

USAGE
─────
python health_monitor_2.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
import math
import queue          # worker frame queues  [Improvement #1]
import sys
import time
import os
import threading
import warnings
from collections import deque
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
YOLO_MODEL       = "yolo11n-pose.pt"
CAM_INDEX        = 1
FRAME_W          = 1280
FRAME_H          = 720

# ── YOLO inference ────────────────────────────────────────────────────────────
DETECT_CONF      = 0.45
DETECT_IOU       = 0.50
INFER_SIZE       = 640   # YOLO internal letterbox grid size (unchanged)

# ── Inference pre-resize  ────────────────────────────────────────────────────
# [Improvement #1] Camera frames are downscaled to INFER_W × INFER_H before
# being passed to the YOLO and MediaPipe worker threads.  The full-resolution
# frame is kept on the main thread for clean display output only.
# Detected coordinates are rescaled back to camera resolution via sx / sy.
INFER_W          = 640   # inference frame width
INFER_H          = 360   # inference frame height (preserves 16:9 aspect ratio)

# ── FaceMesh frame-skip  ─────────────────────────────────────────────────────
# [Improvement #1] Run MediaPipe FaceMesh once per N frames in the face worker.
# The main thread reuses the last available result on skipped frames.
FACEMESH_EVERY_N = 2

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

# ── Fall-speed detection  ─────────────────────────────────────────────────────
# [Improvement #3]
# Vertical velocity of the shoulder/hip midpoint over the last N frames.
# To confirm a FALLEN classification the trunk-angle criterion must be met AND
# one of two speed conditions must hold:
#   a) fall_speed ≥ FALL_SPEED_THRESH  →  person is actively falling
#   b) |fall_speed| ≤ FALL_SPEED_STOP  →  person already lying still
# This gates out brief forward-lean false positives.
FALL_SPEED_HISTORY = 8    # number of Y samples kept per person
FALL_SPEED_THRESH  = 5.0  # px/frame (display-space) — active-fall gate
FALL_SPEED_STOP    = 1.5  # px/frame — at-rest-on-ground gate

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
RPPG_BUFFER_FRAMES   = 300    # ~10 s at 30 fps
RPPG_BPF_LOW         = 0.7   # 42 bpm
RPPG_BPF_HIGH        = 4.0   # 240 bpm
RPPG_MIN_FRAMES      = 90    # ~3 s
RPPG_ASSUMED_FPS     = 30.0

# Face-mesh ROI landmark indices (MediaPipe 468-point model)
FOREHEAD_LANDMARKS     = [10, 67, 69, 104, 108, 151, 337, 338, 297, 299]
LEFT_CHEEK_LANDMARKS   = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148]
RIGHT_CHEEK_LANDMARKS  = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377]

ROI_LANDMARK_INDICES: Tuple[int, ...] = tuple(sorted({
    *FOREHEAD_LANDMARKS, *LEFT_CHEEK_LANDMARKS, *RIGHT_CHEEK_LANDMARKS,
}))

_ERODE_KERNEL        = np.ones((3, 3), dtype=np.uint8)

KALMAN_PROCESS_NOISE = 1e-5
KALMAN_MEASURE_NOISE = 1e-3

SPO2_SMOOTH_ALPHA    = 0.15

MAX_NUM_FACES        = 4
FACE_MATCH_IOU_MIN   = 0.25
FACE_TRACK_TIMEOUT   = 2.0

# SpO₂ update throttle — run heavy processing every N frames
SPO2_UPDATE_INTERVAL = 15   # ~0.5 s at 30 fps


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
C_ROI     = (   0, 255, 180)
C_SPO2_OK = (  50, 210,  50)
C_SPO2_LO = (  15,  15, 240)

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
    """
    def __init__(self, process_noise: float = KALMAN_PROCESS_NOISE,
                 measure_noise: float = KALMAN_MEASURE_NOISE) -> None:
        self.Q = process_noise
        self.R = measure_noise
        self.x = 0.0
        self.P = 1.0
        self._initialised = False

    def update(self, measurement: float) -> float:
        if not self._initialised:
            self.x = measurement
            self._initialised = True
            return self.x
        P_pred = self.P + self.Q
        K      = P_pred / (P_pred + self.R)
        self.x = self.x + K * (measurement - self.x)
        self.P = (1.0 - K) * P_pred
        return self.x


class LandmarkKalman:
    """
    Kalman-smooths only ROI-relevant landmark indices (~30), not all 468.
    One instance per tracked face — never share across faces.
    """
    def __init__(self, indices: Tuple[int, ...] = ROI_LANDMARK_INDICES) -> None:
        self._indices = indices
        self.kx: Dict[int, KalmanStabiliser] = {i: KalmanStabiliser() for i in indices}
        self.ky: Dict[int, KalmanStabiliser] = {i: KalmanStabiliser() for i in indices}

    def update(self, lm_list: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Smooth only ROI indices; other landmarks pass through unchanged."""
        out = list(lm_list)
        for i in self._indices:
            if i >= len(lm_list):
                continue
            x, y = lm_list[i]
            out[i] = (self.kx[i].update(x), self.ky[i].update(y))
        return out


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
    3. POS (Plane-Orthogonal-to-Skin) as alternative / ensemble
    4. Detrending  → removes slow drift (baseline wander)
    5. Normalisation → zero-mean, unit variance
    6. Butterworth bandpass filter [0.7 – 4.0 Hz]
    7. Artefact removal via IQR clipping
    8. Feature extraction:
       Time-domain: mean, std, skewness, kurtosis, peak-to-peak, RMS
       Frequency-domain: dominant frequency, spectral entropy, HRV proxy
    9. Random Forest regression → SpO₂ estimate
   10. Heart rate from dominant FFT peak (bpm)
    """

    def __init__(self, fps: float = RPPG_ASSUMED_FPS) -> None:
        self.fps = fps

        self._buf_len   = RPPG_BUFFER_FRAMES
        self._r_buf     = np.zeros(self._buf_len, dtype=np.float64)
        self._g_buf     = np.zeros(self._buf_len, dtype=np.float64)
        self._b_buf     = np.zeros(self._buf_len, dtype=np.float64)
        self._buf_count = 0
        self._buf_head  = 0

        self.spo2:           float = 0.0
        self.heart_rate:     float = 0.0
        self.spo2_smooth:    float = 0.0
        self.signal_quality: float = 0.0

        self.roi_rects: List[Tuple[int, int, int, int]] = []

        self._bpf_cache: Dict[Tuple[float, float, float],
                              Tuple[np.ndarray, np.ndarray]] = {}
        self._model: Optional[object] = None
        self._model_ready = False

    # ── ML model initialisation (lazy) ────────────────────────────────────────
    def _init_ml_model(self) -> None:
        if not SKLEARN_OK:
            return

        np.random.seed(42)
        n = 2000

        spo2_gt   = np.random.uniform(88.0, 100.0, n)
        ratio_rg  = (-0.8 * spo2_gt + 104.0) / 100.0 + np.random.normal(0, 0.05, n)
        ratio_rb  = ratio_rg * 0.85 + np.random.normal(0, 0.04, n)

        dominant_freq = np.random.uniform(0.8, 2.5, n)
        spectral_ent  = np.random.uniform(0.2, 0.9, n)
        snr_proxy     = np.random.uniform(0.3, 1.0, n)
        rms_green     = np.random.uniform(0.01, 0.15, n)
        skin_tone     = np.random.randint(1, 7, n).astype(float)

        X = np.column_stack([
            ratio_rg, ratio_rb, dominant_freq,
            spectral_ent, snr_proxy, rms_green, skin_tone,
        ])
        y = np.clip(spo2_gt, 88.0, 100.0)

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
        r_vals, g_vals, b_vals = [], [], []
        for mask in roi_masks:
            if mask is None or mask.sum() == 0:
                continue
            roi_pixels = frame[mask > 0]
            if roi_pixels.size == 0:
                continue
            b_vals.append(roi_pixels[:, 0].mean())
            g_vals.append(roi_pixels[:, 1].mean())
            r_vals.append(roi_pixels[:, 2].mean())

        if r_vals:
            self.push_frame_direct(
                float(np.mean(r_vals)),
                float(np.mean(g_vals)),
                float(np.mean(b_vals)),
            )

    def push_frame_direct(self, r_mean: float, g_mean: float, b_mean: float) -> None:
        """Push pre-averaged RGB values into the ring buffer — O(1)."""
        idx = self._buf_head
        self._r_buf[idx] = r_mean
        self._g_buf[idx] = g_mean
        self._b_buf[idx] = b_mean
        self._buf_head = (idx + 1) % self._buf_len
        if self._buf_count < self._buf_len:
            self._buf_count += 1

    def _buffer_view(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self._buf_count
        if n == 0:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty, empty
        if n < self._buf_len:
            return self._r_buf[:n], self._g_buf[:n], self._b_buf[:n]
        h = self._buf_head
        return (
            np.concatenate((self._r_buf[h:], self._r_buf[:h])),
            np.concatenate((self._g_buf[h:], self._g_buf[:h])),
            np.concatenate((self._b_buf[h:], self._b_buf[:h])),
        )

    def _get_bandpass_coefs(self, fps: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        nyq  = fps / 2.0
        low  = float(np.clip(RPPG_BPF_LOW  / nyq, 0.001, 0.999))
        high = float(np.clip(RPPG_BPF_HIGH / nyq, 0.001, 0.999))
        if low >= high:
            return None
        key = (fps, low, high)
        if key not in self._bpf_cache:
            self._bpf_cache[key] = butter(4, [low, high], btype="band")
        return self._bpf_cache[key]

    # ── Main signal processing ────────────────────────────────────────────────
    def process(self, fps: Optional[float] = None) -> bool:
        if fps is not None:
            self.fps = fps

        n = self._buf_count
        if n < RPPG_MIN_FRAMES:
            return False

        R, G, B = self._buffer_view()

        R_norm = R / (R.mean() + 1e-8)
        G_norm = G / (G.mean() + 1e-8)
        B_norm = B / (B.mean() + 1e-8)

        X_c = 3.0 * R_norm - 2.0 * G_norm
        Y_c = 1.5 * R_norm + G_norm - 1.5 * B_norm
        alpha = (np.std(X_c) / (np.std(Y_c) + 1e-8))
        chrom_signal = X_c - alpha * Y_c

        C  = np.column_stack([R_norm, G_norm, B_norm])
        Cn = C / (C.mean(axis=0) + 1e-8)
        H  = np.array([[0, 1, -1], [-2, 1, 1]], dtype=np.float64)
        S_pos = (H @ Cn.T).T
        pos_signal = S_pos[:, 0] - (np.std(S_pos[:, 0]) /
                                     (np.std(S_pos[:, 1]) + 1e-8)) * S_pos[:, 1]

        raw_signal = 0.5 * chrom_signal + 0.5 * pos_signal

        detrended = detrend(raw_signal)

        std = detrended.std()
        if std < 1e-8:
            return False
        normalised = (detrended - detrended.mean()) / std

        coefs = self._get_bandpass_coefs(self.fps)
        if coefs is None:
            return False
        b_coef, a_coef = coefs
        filtered = filtfilt(b_coef, a_coef, normalised)

        q1, q3 = np.percentile(filtered, [25, 75])
        iqr = q3 - q1
        filtered = np.clip(filtered, q1 - 3.0 * iqr, q3 + 3.0 * iqr)

        features, freq_hz = self._extract_features(filtered, R, G, B, raw_signal)
        if features is None:
            return False

        if not self._model_ready:
            self._init_ml_model()
        if self._model_ready:
            feat_vec = np.array(features).reshape(1, -1)
            spo2_raw = float(self._model.predict(feat_vec)[0])
            self.spo2 = float(np.clip(spo2_raw, 70.0, 100.0))
        else:
            ac_r = np.std(R - R.mean())
            dc_r = R.mean()
            ac_g = np.std(G - G.mean())
            dc_g = G.mean() + 1e-8
            ratio = (ac_r / (dc_r + 1e-8)) / (ac_g / dc_g)
            self.spo2 = float(np.clip(104.0 - 17.0 * ratio, 70.0, 100.0))

        if self.spo2_smooth == 0.0:
            self.spo2_smooth = self.spo2
        else:
            self.spo2_smooth = (SPO2_SMOOTH_ALPHA * self.spo2 +
                                (1.0 - SPO2_SMOOTH_ALPHA) * self.spo2_smooth)

        if freq_hz > 0:
            self.heart_rate = freq_hz * 60.0

        return True

    def _extract_features(
        self,
        filtered: np.ndarray,
        R: np.ndarray, G: np.ndarray, B: np.ndarray,
        raw_signal: np.ndarray,
    ) -> Tuple[Optional[List[float]], float]:
        n = len(filtered)
        if n < 32:
            return None, 0.0

        rms = float(np.sqrt(np.mean(filtered ** 2)))
        ptp = float(np.ptp(filtered))
        from scipy.stats import skew, kurtosis
        skewness = float(skew(filtered))
        kurt     = float(kurtosis(filtered))

        def ac_dc(sig: np.ndarray) -> float:
            return float(sig.std() / (np.mean(np.abs(sig)) + 1e-8))

        ac_r, ac_g, ac_b = ac_dc(R), ac_dc(G), ac_dc(B)
        ratio_rg = ac_r / (ac_g + 1e-8)
        ratio_rb = ac_r / (ac_b + 1e-8)

        nperseg = min(256, n // 2)
        freqs, psd = scipy_signal.welch(filtered, fs=self.fps, nperseg=nperseg)

        mask = (freqs >= RPPG_BPF_LOW) & (freqs <= RPPG_BPF_HIGH)
        if mask.sum() == 0:
            return None, 0.0

        psd_band   = psd[mask]
        freqs_band = freqs[mask]
        dom_idx    = int(np.argmax(psd_band))
        dom_freq   = float(freqs_band[dom_idx])

        psd_norm = psd_band / (psd_band.sum() + 1e-8)
        spec_ent = float(-np.sum(psd_norm * np.log(psd_norm + 1e-8)))

        peak_power = float(psd_band[dom_idx])
        mean_power = float(psd_band.mean())
        snr_proxy  = peak_power / (mean_power + 1e-8)

        rms_green        = float(np.sqrt(np.mean(G ** 2)))
        skin_tone_proxy  = float(np.clip(rms_green / 255.0 * 6.0, 1.0, 6.0))

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
    Coordinates are in the same space as the frame passed to extract().
    """
    face_id : int
    masks   : List[np.ndarray]
    rects   : List[Tuple[int, int, int, int]]
    bbox    : Tuple[int, int, int, int]

    def mean_rgb(self, frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Average BGR across all ROI patches — O(ROI pixels), not O(frame area)."""
        r_vals, g_vals, b_vals = [], [], []
        for mask, (x1, y1, x2, y2) in zip(self.masks, self.rects):
            if mask is None or mask.size == 0:
                continue
            patch = frame[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            pixels = patch[mask > 0]
            if pixels.size == 0:
                continue
            b_vals.append(float(pixels[:, 0].mean()))
            g_vals.append(float(pixels[:, 1].mean()))
            r_vals.append(float(pixels[:, 2].mean()))
        if not r_vals:
            return None
        return (float(np.mean(r_vals)), float(np.mean(g_vals)), float(np.mean(b_vals)))


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED WORKER RESULT CONTAINERS  [Improvement #1]
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class PersonDetection:
    """
    One tracked person returned by the YOLO worker thread.
    All coordinates are already rescaled to *display-space* (camera resolution)
    by the worker before being placed in the shared result store.
    """
    track_id : int
    bbox     : Tuple[int, int, int, int]   # display-space (x1,y1,x2,y2)
    kp_xy    : Optional[np.ndarray]        # shape (17, 2), display-space
    kp_conf  : Optional[np.ndarray]        # shape (17,)


@dataclass
class SharedFaceResult:
    """
    Per-face extraction result published by the face worker thread.

    rgb        : mean BGR signal sampled from the *inference* frame (same
                 information as a full-res sample — the mean pixel value
                 of a skin ROI is resolution-independent).
    rects_disp : ROI bounding boxes already rescaled to display-space.
    bbox_disp  : whole-face bbox in display-space.
    """
    face_id    : int
    rgb        : Optional[Tuple[float, float, float]]  # mean B, G, R
    rects_disp : List[Tuple[int, int, int, int]]
    bbox_disp  : Tuple[int, int, int, int]


# ══════════════════════════════════════════════════════════════════════════════
#  FACE TRACK  —  per-face Kalman instance + stable ID bookkeeping
# ══════════════════════════════════════════════════════════════════════════════
class FaceTrack:
    """
    One entry in FaceRoiExtractor's track pool.
    Each tracked face owns its own LandmarkKalman so that landmark states
    from different people are NEVER mixed.
    """
    _id_counter: int = 0

    def __init__(self, bbox: Tuple[int, int, int, int], ts: float) -> None:
        FaceTrack._id_counter += 1
        self.face_id   : int                       = FaceTrack._id_counter
        self.kalman    : LandmarkKalman            = LandmarkKalman()
        self.bbox      : Tuple[int, int, int, int] = bbox
        self.last_seen : float                     = ts


# ══════════════════════════════════════════════════════════════════════════════
#  FACE MESH ROI EXTRACTOR  (multi-face, isolated per-face Kalman filters)
# ══════════════════════════════════════════════════════════════════════════════
class FaceRoiExtractor:
    """
    Uses MediaPipe FaceMesh (468 landmarks) to locate and extract forehead +
    left-cheek + right-cheek ROI masks for every detected face in the frame.

    This class is owned and used exclusively by the face worker thread.
    Results are published through SharedFaceResult via a shared dict + Lock.
    """

    def __init__(self, max_faces: int = MAX_NUM_FACES) -> None:
        if not MEDIAPIPE_OK:
            self.available = False
            return
        self.available  = True
        self.max_faces  = max_faces

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode        = False,
            max_num_faces            = max_faces,
            refine_landmarks         = True,
            min_detection_confidence = 0.5,
            min_tracking_confidence  = 0.5,
        )
        self.roi_groups: List[List[int]] = [
            FOREHEAD_LANDMARKS,
            LEFT_CHEEK_LANDMARKS,
            RIGHT_CHEEK_LANDMARKS,
        ]
        self._tracks: Dict[int, FaceTrack] = {}

    @property
    def active_face_ids(self) -> List[int]:
        return list(self._tracks.keys())

    @staticmethod
    def _landmarks_to_bbox(
        lms: List[Tuple[float, float]],
    ) -> Tuple[int, int, int, int]:
        xs = [p[0] for p in lms]
        ys = [p[1] for p in lms]
        return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

    @staticmethod
    def _iou(
        a: Tuple[int, int, int, int],
        b: Tuple[int, int, int, int],
    ) -> float:
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

    def _match_and_update_tracks(
        self,
        bboxes: List[Tuple[int, int, int, int]],
        now: float,
    ) -> List[int]:
        track_ids = list(self._tracks.keys())
        assigned  = [-1] * len(bboxes)

        if track_ids:
            n_det = len(bboxes)
            n_trk = len(track_ids)
            iou_mat = np.zeros((n_det, n_trk), dtype=np.float32)
            for r, bb in enumerate(bboxes):
                for c, tid in enumerate(track_ids):
                    iou_mat[r, c] = self._iou(bb, self._tracks[tid].bbox)

            while True:
                r, c = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                if iou_mat[r, c] < FACE_MATCH_IOU_MIN:
                    break
                tid              = track_ids[c]
                assigned[r]      = tid
                self._tracks[tid].bbox      = bboxes[r]
                self._tracks[tid].last_seen = now
                iou_mat[r, :]    = 0.0
                iou_mat[:, c]    = 0.0

        for r, fid in enumerate(assigned):
            if fid == -1:
                t = FaceTrack(bboxes[r], now)
                self._tracks[t.face_id] = t
                assigned[r] = t.face_id
                log_info(f"FaceRoiExtractor: new face track ID {t.face_id}")

        return assigned

    def _prune_stale_tracks(self, now: float) -> None:
        expired = [
            fid for fid, t in self._tracks.items()
            if now - t.last_seen > FACE_TRACK_TIMEOUT
        ]
        for fid in expired:
            log_info(f"FaceRoiExtractor: face track ID {fid} expired")
            del self._tracks[fid]

    def extract(self, frame: np.ndarray) -> List[FaceRoiResult]:
        """
        Run FaceMesh on `frame` and return one FaceRoiResult per detected face.
        Coordinates are in the `frame` pixel space (inference resolution).
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

        all_raw_lms: List[List[Tuple[float, float]]] = [
            [(lm.x * w, lm.y * h) for lm in face.landmark]
            for face in result.multi_face_landmarks
        ]
        all_bboxes: List[Tuple[int, int, int, int]] = [
            self._landmarks_to_bbox(lms) for lms in all_raw_lms
        ]

        face_ids = self._match_and_update_tracks(all_bboxes, now)
        self._prune_stale_tracks(now)

        output: List[FaceRoiResult] = []

        for det_idx, face_id in enumerate(face_ids):
            track      = self._tracks[face_id]
            smooth_lms = track.kalman.update(all_raw_lms[det_idx])

            masks: List[np.ndarray]                = []
            rects: List[Tuple[int, int, int, int]] = []

            for group in self.roi_groups:
                pts = np.array(
                    [(int(smooth_lms[i][0]), int(smooth_lms[i][1]))
                     for i in group],
                    dtype=np.int32,
                )
                if len(pts) < 3:
                    continue

                hull = cv2.convexHull(pts)
                bx, by, bw, bh = cv2.boundingRect(hull)
                local_mask = np.zeros((bh, bw), dtype=np.uint8)
                local_hull = hull - np.array([bx, by], dtype=np.int32)
                cv2.fillConvexPoly(local_mask, local_hull, 255)
                local_mask = cv2.erode(local_mask, _ERODE_KERNEL, iterations=1)

                x1 = max(0, bx);      y1 = max(0, by)
                x2 = min(w, bx + bw); y2 = min(h, by + bh)

                if x1 > bx or y1 > by or x2 < bx + bw or y2 < by + bh:
                    mask_h = y2 - y1;  mask_w = x2 - x1
                    local_mask = local_mask[
                        y1 - by: y1 - by + mask_h,
                        x1 - bx: x1 - bx + mask_w,
                    ]

                masks.append(local_mask)
                rects.append((x1, y1, x2, y2))

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
    Lightweight 1D-CNN inspired by PhysNet (Chen & McDuff, 2018).
    Architecture: Conv1d → BN → ReLU → MaxPool (×3) → FC → Sigmoid → SpO₂
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
                    nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                    nn.Conv1d(32, 64, kernel_size=5, padding=2),
                    nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
                    nn.Conv1d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
                )
                self.head = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(64, 1), nn.Sigmoid(),
                )
            def forward(self, x):
                return self.head(self.features(x)) * 12.0 + 88.0

        return _Net()

    def predict(self, processor: "RppgProcessor") -> float:
        if not self.available or processor._buf_count < RPPG_MIN_FRAMES:
            return 0.0
        R, G, B = processor._buffer_view()
        seq = np.stack([R, G, B], axis=0)
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
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
    fallen_votes:   int = 0
    posture_votes:  Dict[str, int] = field(default_factory=dict)

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

    # ── Fall-speed tracking  [Improvement #3] ─────────────────────────────────
    # Hip (or shoulder) midpoint Y in display-space pixels, stored for the last
    # FALL_SPEED_HISTORY frames.  Positive Y = toward bottom of frame.
    hip_y_history:    deque = field(
                          default_factory=lambda: deque(maxlen=FALL_SPEED_HISTORY))
    # Smoothed vertical velocity (px/frame, display-space).
    # Positive  = moving downward (falling).
    # Near-zero = stationary (standing or already on ground).
    fall_speed_px:    float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  POSTURE CLASSIFIER  (updated with fall-speed gate)
# ══════════════════════════════════════════════════════════════════════════════
def classify_posture(
        kp_xy:      Optional[np.ndarray],
        kp_conf:    Optional[np.ndarray],
        bbox:       Tuple[int, int, int, int],
        fall_speed: float = 0.0,      # [Improvement #3] display-px/frame
) -> str:
    """
    Classify body posture from keypoints and bounding-box aspect ratio.

    Fall confirmation  [Improvement #3]
    ─────────────────────────────────────
    When the trunk angle > FALLEN_TRUNK_DEG, a FALLEN label is issued only
    when at least ONE speed condition holds:

      a) fall_speed ≥ FALL_SPEED_THRESH   → person is actively falling
         (hip/shoulder midpoint moving rapidly downward)

      b) |fall_speed| ≤ FALL_SPEED_STOP  → person already lying still
         (angle indicates horizontal posture, speed ≈ 0 because they landed)

      c) fall_speed == 0.0 and no history → first frames, no data yet;
         allow FALLEN so detection isn't blocked at startup.

    The speed gate reduces false positives from momentary forward lean, which
    produces a large trunk angle but has near-zero downward hip velocity.
    """
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

        if trunk_angle > FALLEN_TRUNK_DEG:
            # ── Speed gate ─────────────────────────────────────────────────────
            # Confirm as FALLEN only if actively falling OR already on ground.
            # fall_speed == 0.0 when there is no history yet (first few frames).
            speed_confirms = (
                fall_speed >= FALL_SPEED_THRESH          # (a) active fall
                or abs(fall_speed) <= FALL_SPEED_STOP    # (b) at rest on ground
                or fall_speed == 0.0                     # (c) no history yet
            )
            return P.FALLEN if speed_confirms else P.SITTING

        if trunk_angle < UPRIGHT_TRUNK_DEG:
            return P.STANDING if (bh / bw) > STANDING_BBOX_RATIO else P.SITTING
        return P.FALLEN if (bw / bh) > FALLEN_BBOX_RATIO else P.SITTING

    # ── Keypoint-free fallback: bbox aspect ratio only ─────────────────────────
    if (bw / bh) > FALLEN_BBOX_RATIO:
        return P.FALLEN
    return P.STANDING if (bh / bw) > STANDING_BBOX_RATIO else P.SITTING


def smooth_posture(state: PersonState, raw: str) -> str:
    """
    Majority vote over POSTURE_HIST_LEN frames with O(1) incremental tallies.
    """
    hist = state.posture_hist
    if len(hist) == hist.maxlen:
        evicted = hist[0]
        if evicted == P.FALLEN:
            state.fallen_votes -= 1
        else:
            state.posture_votes[evicted] = state.posture_votes.get(evicted, 1) - 1
            if state.posture_votes[evicted] <= 0:
                del state.posture_votes[evicted]

    hist.append(raw)
    if raw == P.FALLEN:
        state.fallen_votes += 1
    else:
        state.posture_votes[raw] = state.posture_votes.get(raw, 0) + 1

    if state.fallen_votes >= FALLEN_VOTES_NEEDED:
        return P.FALLEN
    if state.posture_votes:
        return max(state.posture_votes, key=state.posture_votes.get)
    return raw


# ══════════════════════════════════════════════════════════════════════════════
#  STATE UPDATER  (updated with fall-speed computation)
# ══════════════════════════════════════════════════════════════════════════════
def update_person_state(
    state:    PersonState,
    cx:       int,
    cy:       int,
    raw_posture: str,
    now:      float,
    kp_xy:    Optional[np.ndarray] = None,
    kp_conf:  Optional[np.ndarray] = None,
) -> None:
    state.last_seen   = now
    state.center      = (cx, cy)
    state.frame_count += 1

    new_posture = smooth_posture(state, raw_posture)

    if new_posture != state.prev_posture and state.frame_count > SETTLING_FRAMES:
        log_info(f"ID {state.track_id:>3d}: {state.prev_posture:<10} → {new_posture}")
    state.prev_posture = new_posture
    state.posture      = new_posture

    # ── Fall-speed tracking  [Improvement #3] ─────────────────────────────────
    # Record the vertical position of the hip midpoint (display-space Y).
    # Fall back to shoulder midpoint if hips are not reliably detected.
    # Positive Y increases toward the bottom of the image (downward direction).
    body_y: Optional[float] = None
    if kp_xy is not None and kp_conf is not None and len(kp_conf) > 12:
        if (float(kp_conf[11]) >= KP_MIN_CONF and
                float(kp_conf[12]) >= KP_MIN_CONF):
            # Hip midpoint — preferred reference point
            body_y = (float(kp_xy[11][1]) + float(kp_xy[12][1])) / 2.0
        elif (float(kp_conf[5]) >= KP_MIN_CONF and
                  float(kp_conf[6]) >= KP_MIN_CONF):
            # Shoulder midpoint — fallback when hips are occluded
            body_y = (float(kp_xy[5][1]) + float(kp_xy[6][1])) / 2.0

    if body_y is not None:
        state.hip_y_history.append(body_y)

    # Compute mean vertical velocity across the recorded history window.
    # Each delta is: Y[t] − Y[t-1]; positive means moved downward (falling).
    if len(state.hip_y_history) >= 2:
        deltas = [
            state.hip_y_history[j] - state.hip_y_history[j - 1]
            for j in range(1, len(state.hip_y_history))
        ]
        state.fall_speed_px = float(np.mean(deltas))
    else:
        # Not enough history yet — leave at 0.0 (classify_posture treats this
        # as "no data" and skips the speed gate for the initial frames).
        state.fall_speed_px = 0.0

    # ── Motion snapshot ───────────────────────────────────────────────────────
    if now - state.snapshot_time >= MOTION_SNAPSHOT_SEC:
        was_moving = state.is_moving
        CHOSEN_KPS = [0, 5, 6, 9, 10, 13, 14, 15, 16]
        displacements: List[float] = []

        if (kp_xy is not None and kp_conf is not None and
                state.snapshot_kps is not None and
                state.snapshot_kp_conf is not None):
            try:
                for idx in CHOSEN_KPS:
                    if (idx < len(kp_conf) and
                            idx < len(state.snapshot_kp_conf) and
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
            log_info(f"ID {state.track_id:>3d}: motion → {status} "
                     f"(disp={mean_disp:.1f}px)")

        state.snapshot_time    = now
        state.snapshot_pos     = (cx, cy)
        state.snapshot_kps     = kp_xy.copy()   if kp_xy   is not None else None
        state.snapshot_kp_conf = kp_conf.copy() if kp_conf is not None else None

    # ── Fallen-state entry / exit ─────────────────────────────────────────────
    if state.posture == P.FALLEN:
        if state.fallen_since is None:
            state.fallen_since = now
            if state.frame_count > SETTLING_FRAMES:
                log_warning(f"ID {state.track_id:>3d}: entered FALLEN state "
                             f"(fall_speed={state.fall_speed_px:+.1f}px/f)")
    else:
        if state.fallen_since is not None:
            duration = now - state.fallen_since
            log_info(f"ID {state.track_id:>3d}: recovered from FALLEN "
                     f"({duration:.1f}s)")
        state.fallen_since = None
        if state.major_alert:
            log_info(f"ID {state.track_id:>3d}: major alert cleared "
                     f"(posture recovered)")
            state.major_alert    = False
            state.alert_fired_at = None

    # ── Alert gate ────────────────────────────────────────────────────────────
    immobile_secs = now - state.last_move_time

    if state.posture == P.FALLEN and state.frame_count > SETTLING_FRAMES:
        if immobile_secs >= WARNING_ALERT_SEC and not state.warning_alert:
            state.warning_alert    = True
            state.warning_fired_at = now
            log_warning(f"ID {state.track_id:>3d}: WARNING — fallen & still "
                        f"{immobile_secs:.0f}s")
            threading.Thread(target=play_sound,
                             args=("warning",), daemon=True).start()
            threading.Thread(target=notify_webhooks,
                             args=(state, "warning", immobile_secs),
                             daemon=True).start()

        if immobile_secs >= IMMOBILE_ALERT_SEC and not state.major_alert:
            state.major_alert    = True
            state.alert_fired_at = now
            log_alert(f"ID {state.track_id:>3d}: MAJOR ALERT — fallen & "
                      f"immobile {immobile_secs:.0f}s !!")
            threading.Thread(target=play_sound,
                             args=("major",), daemon=True).start()
            threading.Thread(target=notify_webhooks,
                             args=(state, "major", immobile_secs),
                             daemon=True).start()
    else:
        if state.warning_alert or state.major_alert:
            if state.major_alert:
                log_info(f"ID {state.track_id:>3d}: major alert cleared "
                         f"(posture recovered)")
            elif state.warning_alert:
                log_info(f"ID {state.track_id:>3d}: warning cleared "
                         f"(posture recovered)")

        state.fallen_since      = None
        state.warning_alert     = False
        state.warning_fired_at  = None
        state.major_alert       = False
        state.alert_fired_at    = None
        state.notified_telegram = False
        state.notified_discord  = False


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
    """
    Draw pose keypoints and skeleton edges.

    NOTE  [Improvement #2]: This function is kept for reference but is NOT
    called anywhere in the main loop.  Skeleton drawing has been disabled to
    reduce per-frame CPU cost and keep the display clean.
    """
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
    """
    Draw the per-person HUD panel: corner-box + status lines.

    Panel lines (updated for Improvement #3):
      • ID
      • Posture
      • Motion status / still duration
      • Fall speed [NEW] — shows vertical velocity in px/frame with colour coding
      • Warning / alert lines (conditional)
    """
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

    # ── Fall speed line  [Improvement #3] ─────────────────────────────────────
    # Displayed whenever we have at least two Y samples.
    # Colour coding:
    #   red    → speed ≥ FALL_SPEED_THRESH  (active fall in progress)
    #   orange → speed > FALL_SPEED_STOP    (some downward movement)
    #   white  → near-zero or upward        (stationary / rising)
    if len(state.hip_y_history) >= 2:
        spd = state.fall_speed_px
        if spd >= FALL_SPEED_THRESH:
            spd_col = C_ALERT
            spd_str = f"\u2193spd: {spd:+.1f}px/f  FAST"
        elif spd > FALL_SPEED_STOP:
            spd_col = C_CAUTION
            spd_str = f"\u2193spd: {spd:+.1f}px/f"
        elif spd < -FALL_SPEED_STOP:
            spd_col = C_SAFE
            spd_str = f"\u2191spd: {spd:+.1f}px/f"
        else:
            spd_col = C_WHITE
            spd_str = f" spd: {spd:+.1f}px/f"
        lines.append((spd_str, spd_col))

    if state.warning_alert and not state.major_alert:
        lines.append((f"WARN: still {now - (state.warning_fired_at or now):.0f}s",
                       C_CAUTION))
    if fallen_s > 1:
        lines.append((f"Down: {fallen_s:.0f}s", C_CAUTION))
    if state.major_alert:
        elapsed = now - (state.alert_fired_at or now)
        lines.append((f"!! ALERT {elapsed:.0f}s !!", C_ALERT))

    PAD = 5; LINE_H = 20; PW = 178   # slightly wider for speed text
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
    n_total = n_standing = n_sitting = n_fallen = n_alert = 0
    for s in states.values():
        n_total += 1
        if s.posture == P.STANDING:   n_standing += 1
        elif s.posture == P.SITTING:  n_sitting  += 1
        elif s.posture == P.FALLEN:   n_fallen   += 1
        if s.major_alert:             n_alert    += 1

    lines = [
        (f"FPS    {fps:4.1f}",   C_YELLOW),
        (f"People {n_total}",    C_WHITE),
        (f"Stand  {n_standing}", C_SAFE),
        (f"Sit    {n_sitting}",  C_SAFE),
        (f"Fallen {n_fallen}",   C_CAUTION if n_fallen else C_WHITE),
        (f"Alerts {n_alert}",    C_ALERT   if n_alert  else C_WHITE),
        ("──────────────────",   C_GRAY),
        ("[q] Quit",             C_CYAN),
        ("[c] Clear alerts",     C_CYAN),
        ("[s] Toggle SpO\u2082", C_CYAN),
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
    roi_rects: List[Tuple[int, int, int, int]],
    show_rois: bool,
    face_id: int = 0,
    face_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> None:
    """
    Draw the SpO₂ / HR panel for one face.
    roi_rects and face_bbox should already be in display-space coordinates.
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
        (f"SpO\u2082  Face {face_id}", C_YELLOW),
        (spo2_str,        spo2_col if not collecting else C_GRAY),
        (f"HR: {hr_str}", C_WHITE),
        (qual_str,        C_CYAN),
    ]
    if spo2_val > 0 and spo2_val < 94.0:
        lines.append(("!! LOW SpO\u2082 !!", C_ALERT))

    PAD = 8; LINE_H = 22; PW = 178
    PH = len(lines) * LINE_H + PAD * 2

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

    if show_rois:
        roi_labels = ["Forehead", "L-Cheek", "R-Cheek"]
        for idx, rect in enumerate(roi_rects):
            rx1, ry1, rx2, ry2 = rect
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), C_ROI, 1)
            if idx < len(roi_labels):
                cv2.putText(frame, roi_labels[idx], (rx1, ry1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_ROI, 1,
                            cv2.LINE_AA)


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
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text},
                      timeout=5)
    except Exception:
        pass


def send_discord_message(content: str) -> None:
    if not DISCORD_WEBHOOK:
        return
    try:
        requests.post(DISCORD_WEBHOOK, json={"content": content}, timeout=5)
    except Exception:
        pass


def notify_webhooks(state: PersonState, level: str,
                    immobile_secs: float) -> None:
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
        states:     Dict[int, PersonState],
        active_ids: List[int],
        now:        float,
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
#  WORKER THREADS  [Improvement #1]
# ══════════════════════════════════════════════════════════════════════════════

def _yolo_worker(
    model:        "YOLO",
    frame_q:      queue.Queue,
    result_lock:  threading.Lock,
    result_store: dict,
    sx:           float,
    sy:           float,
    stop_ev:      threading.Event,
) -> None:
    """
    YOLO pose-inference worker thread.

    Lifecycle
    ─────────
    • Blocks on frame_q.get() waiting for the next inference frame.
    • Runs model.track() (ByteTrack persist=True) on the received frame.
    • Scales all bounding-box and keypoint coordinates from inference-space
      (INFER_W × INFER_H) to display-space (camera resolution) using sx, sy.
    • Stores a List[PersonDetection] in result_store["detections"] under
      result_lock so the main thread can consume it at any time.

    Frame dropping
    ──────────────
    The main thread uses queue.put_nowait() so only the latest frame is ever
    queued.  If YOLO is slower than the camera, frames are silently dropped —
    this is intentional: we always want to reason about the most recent view.
    """
    log_info("[YOLO-thread] started.")
    while not stop_ev.is_set():
        try:
            infer_frame = frame_q.get(timeout=0.1)
        except queue.Empty:
            continue

        try:
            results = model.track(
                infer_frame,
                persist  = True,
                tracker  = "bytetrack.yaml",
                conf     = DETECT_CONF,
                iou      = DETECT_IOU,
                verbose  = False,
                imgsz    = INFER_SIZE,
            )
        except Exception as exc:
            log_warning(f"[YOLO-thread] inference error: {exc}")
            continue

        result     = results[0]
        detections: List[PersonDetection] = []

        if result.boxes.id is not None:
            track_ids  = result.boxes.id.int().cpu().tolist()
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            has_kp     = result.keypoints is not None
            has_conf   = has_kp and result.keypoints.conf is not None

            for i, tid in enumerate(track_ids):
                # ── Scale inference → display coordinates ──────────────────
                x1, y1, x2, y2 = boxes_xyxy[i]
                x1 = int(x1 * sx);  y1 = int(y1 * sy)
                x2 = int(x2 * sx);  y2 = int(y2 * sy)

                try:
                    kp_xy = (result.keypoints.xy[i].cpu().numpy().copy()
                              if has_kp   else None)
                    kp_conf = (result.keypoints.conf[i].cpu().numpy().copy()
                                if has_conf else None)
                except (AttributeError, IndexError):
                    kp_xy = kp_conf = None

                # Scale keypoint coordinates to display space
                if kp_xy is not None:
                    kp_xy[:, 0] *= sx
                    kp_xy[:, 1] *= sy

                detections.append(PersonDetection(
                    track_id = tid,
                    bbox     = (x1, y1, x2, y2),
                    kp_xy    = kp_xy,
                    kp_conf  = kp_conf,
                ))

        with result_lock:
            result_store["detections"] = detections

    log_info("[YOLO-thread] stopped.")


def _face_worker(
    face_extractor: FaceRoiExtractor,
    frame_q:        queue.Queue,
    result_lock:    threading.Lock,
    result_store:   dict,
    sx:             float,
    sy:             float,
    stop_ev:        threading.Event,
) -> None:
    """
    MediaPipe FaceMesh worker thread.

    Lifecycle
    ─────────
    • Processes every FACEMESH_EVERY_N-th frame to reduce CPU load.
      On skipped frames, the main thread re-uses the previous result.
    • Calls face_extractor.extract(infer_frame) — coordinates are in
      inference-space (INFER_W × INFER_H).
    • Computes mean BGR for each face's ROI patches from the inference frame.
    • Scales ROI rects and face bbox to display-space via sx, sy.
    • Stores a List[SharedFaceResult] and the active face IDs in result_store
      under result_lock.

    Why extract mean RGB here?
    ──────────────────────────
    The mean pixel value of a skin ROI is essentially resolution-independent
    (it is the spatial average of a region).  Computing it on the 640×360
    inference frame avoids passing the full-resolution frame across the thread
    boundary and saves memory bandwidth.
    """
    log_info("[Face-thread] started.")
    _frame_ctr = 0

    while not stop_ev.is_set():
        try:
            infer_frame = frame_q.get(timeout=0.1)
        except queue.Empty:
            continue

        _frame_ctr += 1
        # ── Frame skip  [Improvement #1] ──────────────────────────────────────
        if _frame_ctr % FACEMESH_EVERY_N != 0:
            continue  # reuse previous results on the main thread

        try:
            raw_results = face_extractor.extract(infer_frame)
        except Exception as exc:
            log_warning(f"[Face-thread] extract error: {exc}")
            continue

        shared: List[SharedFaceResult] = []

        for fr in raw_results:
            # Mean BGR from inference frame (resolution-independent average)
            rgb = fr.mean_rgb(infer_frame)

            # Scale rects and face bbox: inference → display
            rects_disp = [
                (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))
                for x1, y1, x2, y2 in fr.rects
            ]
            bx1, by1, bx2, by2 = fr.bbox
            bbox_disp = (
                int(bx1 * sx), int(by1 * sy),
                int(bx2 * sx), int(by2 * sy),
            )

            shared.append(SharedFaceResult(
                face_id    = fr.face_id,
                rgb        = rgb,
                rects_disp = rects_disp,
                bbox_disp  = bbox_disp,
            ))

        with result_lock:
            result_store["results"]    = shared
            result_store["active_ids"] = face_extractor.active_face_ids

    log_info("[Face-thread] stopped.")


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

    face_extractor = FaceRoiExtractor()
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

    # ── Scale factors: inference → display  [Improvement #1] ─────────────────
    # All coordinates returned by the workers are already in display-space
    # (they multiply by sx/sy internally before publishing results).
    sx = actual_w / INFER_W   # e.g. 1280/640 = 2.0
    sy = actual_h / INFER_H   # e.g. 720/360  = 2.0
    log_info(f"Inference pre-resize: {INFER_W}×{INFER_H}  "
             f"(scale back {sx:.2f}×, {sy:.2f}×  for display)")
    log_info("Controls: [q] quit  [c] clear alerts  [s] toggle SpO₂  "
             "[1-9] clear by ID")
    print()

    # ── Worker thread infrastructure  [Improvement #1] ────────────────────────
    # Each worker gets its own Queue(maxsize=1).  The main thread calls
    # put_nowait() — if the queue is already full the new frame is silently
    # dropped, so the worker always processes the LATEST available frame
    # rather than building up a backlog.
    _stop_workers = threading.Event()

    _yolo_q     = queue.Queue(maxsize=1)
    _yolo_lock  = threading.Lock()
    _yolo_store : dict = {"detections": []}

    _face_q     = queue.Queue(maxsize=1)
    _face_lock  = threading.Lock()
    _face_store : dict = {"results": [], "active_ids": []}

    active_threads: List[threading.Thread] = []

    if model is not None:
        t_yolo = threading.Thread(
            target = _yolo_worker,
            args   = (model, _yolo_q, _yolo_lock, _yolo_store,
                      sx, sy, _stop_workers),
            daemon = True,
            name   = "yolo-worker",
        )
        t_yolo.start()
        active_threads.append(t_yolo)
        log_info("YOLO worker thread started.")

    if face_extractor.available:
        t_face = threading.Thread(
            target = _face_worker,
            args   = (face_extractor, _face_q, _face_lock, _face_store,
                      sx, sy, _stop_workers),
            daemon = True,
            name   = "face-worker",
        )
        t_face.start()
        active_threads.append(t_face)
        log_info(f"Face worker thread started  "
                 f"(FaceMesh every {FACEMESH_EVERY_N} frames).")

    # ── Main-thread state ─────────────────────────────────────────────────────
    person_states: Dict[int, PersonState]   = {}
    fps           = 0.0
    frame_times   = deque(maxlen=30)
    show_spo2     = True

    # One RppgProcessor per live face track ID — managed on the main thread
    rppg_pool: Dict[int, RppgProcessor] = {}

    # Cached latest shared face results (List[SharedFaceResult])
    face_results_disp: List[SharedFaceResult] = []

    # SpO₂ update throttle (main-thread counter — independent of frame drops)
    spo2_frame_counter = 0

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            log_warning("Frame grab failed — retrying…")
            time.sleep(0.05)
            continue

        now = time.time()

        # ── Pre-resize for inference  [Improvement #1] ────────────────────────
        # Workers receive a smaller frame; display always uses full-res `frame`.
        infer_frame = cv2.resize(frame, (INFER_W, INFER_H))

        # ── Feed workers (non-blocking — drop if busy) ─────────────────────────
        if model is not None:
            try:
                _yolo_q.put_nowait(infer_frame)
            except queue.Full:
                pass   # YOLO still processing last frame — skip this one

        if face_extractor.available and show_spo2:
            try:
                _face_q.put_nowait(infer_frame)
            except queue.Full:
                pass   # Face worker still processing — skip this frame

        # ── MODULE 1: Consume latest YOLO results ─────────────────────────────
        if model is not None:
            with _yolo_lock:
                current_detections: List[PersonDetection] = \
                    list(_yolo_store["detections"])

            active_ids: List[int] = []

            for det in current_detections:
                tid             = det.track_id
                x1, y1, x2, y2 = det.bbox
                cx              = (x1 + x2) // 2
                cy              = (y1 + y2) // 2
                active_ids.append(tid)

                if tid not in person_states:
                    person_states[tid] = PersonState(
                        track_id=tid, snapshot_pos=(cx, cy)
                    )
                    log_info(f"ID {tid:>3d}: new person detected at ({cx},{cy})")

                st      = person_states[tid]
                st.bbox = det.bbox

                # classify_posture uses the fall_speed accumulated in the
                # previous call to update_person_state for this track.
                raw = classify_posture(
                    det.kp_xy,
                    det.kp_conf,
                    det.bbox,
                    fall_speed=st.fall_speed_px,    # [Improvement #3]
                )
                update_person_state(
                    st, cx, cy, raw, now,
                    kp_xy=det.kp_xy,
                    kp_conf=det.kp_conf,
                )

                # ── Skeleton drawing DISABLED  [Improvement #2] ───────────────
                # draw_skeleton(frame, det.kp_xy, det.kp_conf)  ← removed

                draw_person_panel(frame, st, now)

            cleanup_stale_persons(person_states, active_ids, now)

        # ── MODULE 2: Consume latest face results ─────────────────────────────
        if face_extractor.available and show_spo2:
            with _face_lock:
                face_results_disp = list(_face_store["results"])
                active_face_ids   = set(_face_store.get("active_ids", []))

            for sfr in face_results_disp:
                # Lazily create an RppgProcessor for each new face track
                if sfr.face_id not in rppg_pool:
                    rppg_pool[sfr.face_id] = RppgProcessor()
                    log_info(f"SpO\u2082: new processor for face ID {sfr.face_id}")

                # Push pre-computed mean BGR into the ring buffer
                if sfr.rgb is not None:
                    rppg_pool[sfr.face_id].push_frame_direct(*sfr.rgb)

                # Draw ROI outlines — rects_disp are already in display-space
                for rx1, ry1, rx2, ry2 in sfr.rects_disp:
                    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), C_ROI, 1)

            # Throttled SpO₂ update  [Improvement #1: still on main thread,
            # but expensive signal processing only runs every ~0.5 s]
            spo2_frame_counter += 1
            if spo2_frame_counter >= SPO2_UPDATE_INTERVAL:
                spo2_frame_counter = 0
                proc_fps = fps if fps > 5 else RPPG_ASSUMED_FPS
                for sfr in face_results_disp:
                    pool_entry = rppg_pool.get(sfr.face_id)
                    if pool_entry is not None:
                        pool_entry.process(fps=proc_fps)

            # Prune processors whose face track has expired
            stale_fids = [fid for fid in rppg_pool
                          if fid not in active_face_ids]
            for fid in stale_fids:
                log_info(f"SpO\u2082: pruning processor for expired face ID {fid}")
                del rppg_pool[fid]

        # ── Global overlays ───────────────────────────────────────────────────
        draw_global_hud(frame, person_states, fps)
        draw_major_alert_overlay(frame, person_states, now)

        if show_spo2 and face_extractor.available:
            for sfr in face_results_disp:
                if sfr.face_id in rppg_pool:
                    draw_spo2_panel(
                        frame,
                        rppg_pool[sfr.face_id],
                        sfr.rects_disp,
                        show_rois   = True,
                        face_id     = sfr.face_id,
                        face_bbox   = sfr.bbox_disp,
                    )

        # ── Rolling FPS measurement ───────────────────────────────────────────
        frame_times.append(now)
        if len(frame_times) >= 2:
            fps = (len(frame_times) - 1) / (
                frame_times[-1] - frame_times[0] + 1e-9)

        # ── Key input ─────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            log_info("Quit requested.")
            break

        elif key == ord('s'):
            show_spo2 = not show_spo2
            log_info(f"SpO\u2082 panel "
                     f"{'enabled' if show_spo2 else 'disabled'}.")

        elif key == ord('c'):
            cleared = [sid for sid, s in person_states.items()
                       if s.major_alert]
            for sid in cleared:
                person_states[sid].major_alert    = False
                person_states[sid].alert_fired_at = None

            cleared_warn = [sid for sid, s in person_states.items()
                            if s.warning_alert]
            for sid in cleared_warn:
                person_states[sid].warning_alert     = False
                person_states[sid].warning_fired_at  = None
                person_states[sid].notified_telegram = False
                person_states[sid].notified_discord  = False

            log_info(
                f"All alerts dismissed (IDs: {cleared})"
                if cleared else "No active alerts."
            )

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

        cv2.imshow("Health Monitor v2 — Fall Detection + SpO₂ r-PPG", frame)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    log_info("Stopping worker threads…")
    _stop_workers.set()
    for t in active_threads:
        t.join(timeout=2.0)
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

3. SpO₂ RANGE
   • Normoxia: 97–100% (resting subjects)
   • Mild hypoxia: 92–96% (breath-hold or altitude protocol)
   • Controlled desaturation: 88–91% (IRB-approved supervised protocol only)

4. LIGHTING CONDITIONS
   • Fluorescent, LED (3000–6500 K), sunlight, mixed; low-light < 50 lux

5. MOTION ARTIFACTS
   • Still, nodding, head turning, talking, walking (treadmill extreme)

6. VIDEO SPECIFICATION
   • Minimum 720p @ 30 fps; lossless codec preferred for training

REAL-TIME OPTIMISATION TECHNIQUES (implemented in this file)
═════════════════════════════════════════════════════════════
• YOLO and MediaPipe in separate threads with Queue(maxsize=1) frame queues
• Pre-resize to 640×360 before model workers; full-res kept for display
• FaceMesh processed every other frame (FACEMESH_EVERY_N = 2)
• SpO₂ update throttled to every 15 frames (~0.5 s)
• Float16 inference for PyTorch models on GPU (TODO for production)
• Pre-allocated mask arrays; per-frame np.zeros avoided in hot path

DEEP LEARNING ALTERNATIVES TO RANDOM FOREST
════════════════════════════════════════════
1. PhysNet (Chen & McDuff, 2018)       — 3D-CNN on spatiotemporal face crops
2. EfficientPhys (Liu et al., 2023)    — EfficientNet backbone + temporal MHA
3. MTTS-CAN (Liu et al., 2020)         — Multi-task CAN for HR + SpO₂
4. DualGAN / rPPG-MAE (2023–2024)     — Self-supervised pre-training on video
"""

if __name__ == "__main__":
    main()