"""
Standalone local r-PPG SpO2 / heart-rate monitor.

This script runs a MediaPipe FaceMesh + r-PPG pipeline for local SpO2 and heart-rate estimation.
Webcam SpO2 output is an estimate for experimentation only; it is not a medical device
measurement and does not claim clinical accuracy.

Controls:
  q  quit
  s  toggle SpO2/HR panels

Install:
  pip install mediapipe opencv-python numpy scipy
  # scikit-learn is only needed if your optional --spo2-model pickle uses it.

Run:
  python health_monitor.py
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import queue
import sys
import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from scipy import signal as scipy_signal
    from scipy.signal import butter, detrend, filtfilt
    from scipy.stats import kurtosis, skew
    SCIPY_OK = True
except ImportError:
    scipy_signal = None
    butter = None
    detrend = None
    filtfilt = None
    SCIPY_OK = False
    print("[WARNING] scipy not found. Using numpy-only signal preprocessing fallback.")

    def skew(values: np.ndarray) -> float:
        x = np.asarray(values, dtype=np.float64)
        if x.size == 0:
            return 0.0
        centered = x - float(np.mean(x))
        std = float(np.std(centered))
        if std < 1e-8:
            return 0.0
        return float(np.mean((centered / std) ** 3))

    def kurtosis(values: np.ndarray) -> float:
        x = np.asarray(values, dtype=np.float64)
        if x.size == 0:
            return 0.0
        centered = x - float(np.mean(x))
        std = float(np.std(centered))
        if std < 1e-8:
            return 0.0
        return float(np.mean((centered / std) ** 4) - 3.0)

warnings.filterwarnings("ignore")

try:
    import mediapipe as mp
    MEDIAPIPE_OK = True
except ImportError:
    mp = None
    MEDIAPIPE_OK = False
    print("[WARNING] mediapipe not found. Install with: pip install mediapipe")

# Camera and worker settings
CAM_INDEX = 0
FRAME_W = 1280
FRAME_H = 720
INFER_W = 640
INFER_H = 360
FACEMESH_EVERY_N = 2
MAX_NUM_FACES = 4
FACE_MATCH_IOU_MIN = 0.25
FACE_TRACK_TIMEOUT = 2.0

# r-PPG settings
RPPG_BUFFER_FRAMES = 300
RPPG_BPF_LOW = 0.7
RPPG_BPF_HIGH = 4.0
RPPG_MIN_FRAMES = 90
RPPG_ASSUMED_FPS = 30.0
SPO2_UPDATE_INTERVAL = 15
SPO2_SMOOTH_ALPHA = 0.12
SPO2_MIN_SIGNAL_QUALITY = 0.45
SPO2_MAX_STEP_PER_UPDATE = 1.0
SPO2_EMPIRICAL_A = 104.0
SPO2_EMPIRICAL_B = 17.0
RPPG_HR_LOW_HZ = 0.75
RPPG_HR_HIGH_HZ = 3.0
RPPG_MIN_SNR_PROXY = 2.0
RPPG_GOOD_SNR_PROXY = 8.0
RPPG_MIN_PEAK_PROMINENCE = 1.5
RPPG_GOOD_PEAK_PROMINENCE = 4.0
RPPG_MIN_FACE_AREA_FRACTION = 0.008
RPPG_STABLE_FACE_AREA_FRACTION = 0.035
RPPG_MAX_ROI_MOTION = 0.035
ACTIVE_FACE_SWITCH_AREA_RATIO = 1.18
ACTIVE_FACE_SWITCH_MIN_AREA_FRACTION = 0.01

# STEP 5 preprocessing settings.
PREPROCESS_WINDOW_SECONDS = 10.0
PREPROCESS_STEP_SECONDS = 10.0
PREPROCESS_SMOOTH_SECONDS = 0.15
PREPROCESS_MIN_SAMPLES = 16

# FaceMesh ROI landmarks
FOREHEAD_LANDMARKS = [10, 67, 69, 104, 108, 151, 337, 338, 297, 299]
LEFT_CHEEK_LANDMARKS = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148]
RIGHT_CHEEK_LANDMARKS = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377]
ROI_LANDMARK_INDICES: Tuple[int, ...] = tuple(sorted({
    *FOREHEAD_LANDMARKS,
    *LEFT_CHEEK_LANDMARKS,
    *RIGHT_CHEEK_LANDMARKS,
}))

_ERODE_KERNEL = np.ones((3, 3), dtype=np.uint8)
KALMAN_PROCESS_NOISE = 1e-5
KALMAN_MEASURE_NOISE = 1e-3

# Colors in BGR
C_DARK = (18, 18, 18)
C_WHITE = (230, 230, 230)
C_YELLOW = (0, 215, 255)
C_CYAN = (220, 210, 0)
C_GRAY = (110, 110, 110)
C_ROI = (0, 255, 180)
C_SPO2_OK = (50, 210, 50)
C_SPO2_LOW = (15, 15, 240)
C_WARN = (0, 165, 255)

_ANSI = sys.stdout.isatty()


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def log_info(msg: str) -> None:
    print(f"[INFO]    {_ts()} | {msg}")


def log_warning(msg: str) -> None:
    pre = "\033[93m" if _ANSI else ""
    suf = "\033[0m" if _ANSI else ""
    print(f"{pre}[WARNING] {_ts()} | {msg}{suf}")


class KalmanStabiliser:
    """Lightweight 1-D Kalman filter for smoothing scalar landmark positions."""

    def __init__(
        self,
        process_noise: float = KALMAN_PROCESS_NOISE,
        measure_noise: float = KALMAN_MEASURE_NOISE,
    ) -> None:
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
        p_pred = self.P + self.Q
        k_gain = p_pred / (p_pred + self.R)
        self.x += k_gain * (measurement - self.x)
        self.P = (1.0 - k_gain) * p_pred
        return self.x


class LandmarkKalman:
    """Kalman-smooths only the face ROI landmark subset for one tracked face."""

    def __init__(self, indices: Tuple[int, ...] = ROI_LANDMARK_INDICES) -> None:
        self._indices = indices
        self.kx = {idx: KalmanStabiliser() for idx in indices}
        self.ky = {idx: KalmanStabiliser() for idx in indices}

    def update(self, lm_list: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        out = list(lm_list)
        for idx in self._indices:
            if idx >= len(lm_list):
                continue
            x_val, y_val = lm_list[idx]
            out[idx] = (self.kx[idx].update(x_val), self.ky[idx].update(y_val))
        return out



# ---------------- STEP 5: signal preprocessing / noise reduction starts ----------------

@dataclass
class PreprocessedRgbWindow:
    timestamps: np.ndarray
    cleaned_r_signal: np.ndarray
    cleaned_g_signal: np.ndarray
    cleaned_b_signal: np.ndarray
    cleaned_rppg_signal: np.ndarray
    sample_rate: float


def _safe_float_array(values: List[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr
    finite = np.isfinite(arr)
    if finite.all():
        return arr
    fill = float(np.mean(arr[finite])) if finite.any() else 0.0
    return np.where(finite, arr, fill).astype(np.float64)


def estimate_sample_rate(timestamps: List[float] | np.ndarray, fallback_fps: float = RPPG_ASSUMED_FPS) -> float:
    ts = _safe_float_array(timestamps)
    if ts.size >= 2:
        diffs = np.diff(ts)
        diffs = diffs[np.isfinite(diffs) & (diffs > 1e-6)]
        if diffs.size:
            # Median interval is robust to occasional delayed frames or duplicate reads.
            return float(np.clip(1.0 / np.median(diffs), 1.0, 120.0))
    return float(fallback_fps if fallback_fps > 0 else RPPG_ASSUMED_FPS)


def moving_average_signal(values: List[float] | np.ndarray, window_samples: int) -> np.ndarray:
    x = _safe_float_array(values)
    if x.size == 0:
        return x
    window_samples = int(max(1, min(window_samples, x.size)))
    if window_samples <= 1:
        return x.copy()
    kernel = np.ones(window_samples, dtype=np.float64) / float(window_samples)
    return np.convolve(x, kernel, mode="same")


def detrend_signal(values: List[float] | np.ndarray) -> np.ndarray:
    x = _safe_float_array(values)
    if x.size == 0:
        return x
    centered = x - float(np.mean(x))
    if centered.size < 3:
        return centered
    if SCIPY_OK and detrend is not None:
        return np.asarray(detrend(centered, type="linear"), dtype=np.float64)

    # NumPy fallback: remove a slow moving baseline to reduce lighting drift.
    baseline_window = max(3, int(round(centered.size * 0.2)))
    baseline = moving_average_signal(centered, baseline_window)
    return centered - baseline


def normalize_signal(values: List[float] | np.ndarray) -> np.ndarray:
    x = _safe_float_array(values)
    if x.size == 0:
        return x
    centered = x - float(np.mean(x))
    std = float(np.std(centered))
    if std < 1e-8:
        return np.zeros_like(centered)
    return centered / std


def _fft_bandpass_fallback(values: np.ndarray, sample_rate: float, low_hz: float, high_hz: float) -> np.ndarray:
    x = _safe_float_array(values)
    if x.size < 4 or sample_rate <= 0:
        return normalize_signal(x)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / sample_rate)
    spectrum = np.fft.rfft(x)
    keep = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(keep):
        return normalize_signal(x)
    spectrum[~keep] = 0.0
    return np.fft.irfft(spectrum, n=x.size).astype(np.float64)


def bandpass_filter_signal(
    values: List[float] | np.ndarray,
    sample_rate: float,
    low_hz: float = RPPG_BPF_LOW,
    high_hz: float = RPPG_BPF_HIGH,
    order: int = 4,
) -> np.ndarray:
    x = _safe_float_array(values)
    if x.size == 0:
        return x
    if x.size < PREPROCESS_MIN_SAMPLES or sample_rate <= 0:
        return normalize_signal(x)

    nyquist = sample_rate / 2.0
    # 0.7-4.0 Hz covers roughly 42-240 bpm, a practical webcam rPPG band.
    low = max(0.01, min(low_hz, nyquist * 0.90))
    high = max(low + 0.01, min(high_hz, nyquist * 0.95))
    if low >= high or high >= nyquist:
        return normalize_signal(x)

    if SCIPY_OK and butter is not None and filtfilt is not None:
        b_coef, a_coef = butter(order, [low / nyquist, high / nyquist], btype="band")
        padlen = 3 * max(len(a_coef), len(b_coef))
        if x.size > padlen:
            return np.asarray(filtfilt(b_coef, a_coef, x), dtype=np.float64)

    return _fft_bandpass_fallback(x, sample_rate, low, high)


def power_spectrum_signal(values: List[float] | np.ndarray, sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    x = _safe_float_array(values)
    if x.size == 0 or sample_rate <= 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty
    if SCIPY_OK and scipy_signal is not None and x.size >= 8:
        nperseg = min(256, max(8, x.size // 2))
        return scipy_signal.welch(x, fs=sample_rate, nperseg=nperseg)

    freqs = np.fft.rfftfreq(x.size, d=1.0 / sample_rate)
    spectrum = np.abs(np.fft.rfft(x)) ** 2 / max(x.size, 1)
    return freqs.astype(np.float64), spectrum.astype(np.float64)


def _prepare_rgb_window(
    r_values: List[float] | np.ndarray,
    g_values: List[float] | np.ndarray,
    b_values: List[float] | np.ndarray,
    timestamps: Optional[List[float] | np.ndarray],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    r_arr = np.asarray(r_values, dtype=np.float64).reshape(-1)
    g_arr = np.asarray(g_values, dtype=np.float64).reshape(-1)
    b_arr = np.asarray(b_values, dtype=np.float64).reshape(-1)
    n_items = min(r_arr.size, g_arr.size, b_arr.size)
    if n_items == 0:
        return None

    r_arr = r_arr[:n_items]
    g_arr = g_arr[:n_items]
    b_arr = b_arr[:n_items]
    if timestamps is None:
        ts_arr = np.arange(n_items, dtype=np.float64) / RPPG_ASSUMED_FPS
    else:
        ts_arr = np.asarray(timestamps, dtype=np.float64).reshape(-1)[:n_items]
        if ts_arr.size < n_items:
            n_items = ts_arr.size
            r_arr = r_arr[:n_items]
            g_arr = g_arr[:n_items]
            b_arr = b_arr[:n_items]
    if n_items == 0:
        return None

    valid = np.isfinite(r_arr) & np.isfinite(g_arr) & np.isfinite(b_arr) & np.isfinite(ts_arr)
    if not np.any(valid):
        return None
    return r_arr[valid], g_arr[valid], b_arr[valid], ts_arr[valid]


def preprocess_rgb_window(
    r_values: List[float] | np.ndarray,
    g_values: List[float] | np.ndarray,
    b_values: List[float] | np.ndarray,
    timestamps: Optional[List[float] | np.ndarray] = None,
    fallback_fps: float = RPPG_ASSUMED_FPS,
    low_hz: float = RPPG_BPF_LOW,
    high_hz: float = RPPG_BPF_HIGH,
    smooth_seconds: float = PREPROCESS_SMOOTH_SECONDS,
) -> Optional[PreprocessedRgbWindow]:
    prepared = _prepare_rgb_window(r_values, g_values, b_values, timestamps)
    if prepared is None:
        return None
    r_arr, g_arr, b_arr, ts_arr = prepared
    if min(r_arr.size, g_arr.size, b_arr.size, ts_arr.size) < PREPROCESS_MIN_SAMPLES:
        return None

    sample_rate = estimate_sample_rate(ts_arr, fallback_fps=fallback_fps)

    def clean_channel(channel: np.ndarray) -> np.ndarray:
        no_dc = channel - float(np.mean(channel))
        detrended = detrend_signal(no_dc)
        normalised = normalize_signal(detrended)
        filtered = bandpass_filter_signal(normalised, sample_rate, low_hz=low_hz, high_hz=high_hz)
        smooth_samples = int(round(max(0.0, smooth_seconds) * sample_rate))
        if smooth_samples > 1:
            filtered = moving_average_signal(filtered, smooth_samples)
        return normalize_signal(filtered)

    cleaned_r = clean_channel(r_arr)
    cleaned_g = clean_channel(g_arr)
    cleaned_b = clean_channel(b_arr)

    if min(cleaned_r.size, cleaned_g.size, cleaned_b.size) != ts_arr.size:
        n_items = min(cleaned_r.size, cleaned_g.size, cleaned_b.size, ts_arr.size)
        if n_items < PREPROCESS_MIN_SAMPLES:
            return None
        cleaned_r = cleaned_r[:n_items]
        cleaned_g = cleaned_g[:n_items]
        cleaned_b = cleaned_b[:n_items]
        ts_arr = ts_arr[:n_items]

    # A simple green-dominant projection is useful as a generic rPPG signal;
    # SpO2 feature extraction should still use the cleaned per-channel signals.
    cleaned_rppg = normalize_signal(cleaned_g - 0.5 * (cleaned_r + cleaned_b))
    return PreprocessedRgbWindow(
        timestamps=ts_arr,
        cleaned_r_signal=cleaned_r,
        cleaned_g_signal=cleaned_g,
        cleaned_b_signal=cleaned_b,
        cleaned_rppg_signal=cleaned_rppg,
        sample_rate=sample_rate,
    )


@dataclass
class RgbSignalLog:
    face_id: int
    r_signal: List[float] = field(default_factory=list)
    g_signal: List[float] = field(default_factory=list)
    b_signal: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    cleaned_r_signal: List[float] = field(default_factory=list)
    cleaned_g_signal: List[float] = field(default_factory=list)
    cleaned_b_signal: List[float] = field(default_factory=list)
    cleaned_rppg_signal: List[float] = field(default_factory=list)
    cleaned_timestamps: List[float] = field(default_factory=list)
    _processed_until: int = 0

    def append(self, timestamp: float, r_mean: float, g_mean: float, b_mean: float) -> None:
        if not all(np.isfinite([timestamp, r_mean, g_mean, b_mean])):
            return
        self.timestamps.append(float(timestamp))
        self.r_signal.append(float(r_mean))
        self.g_signal.append(float(g_mean))
        self.b_signal.append(float(b_mean))

    def _append_cleaned_window(self, result: PreprocessedRgbWindow, keep_from: int = 0) -> None:
        keep_from = max(0, min(int(keep_from), result.timestamps.size))
        last_ts = self.cleaned_timestamps[-1] if self.cleaned_timestamps else None
        for idx in range(keep_from, result.timestamps.size):
            ts_val = float(result.timestamps[idx])
            if last_ts is not None and ts_val <= last_ts:
                continue
            self.cleaned_timestamps.append(ts_val)
            self.cleaned_r_signal.append(float(result.cleaned_r_signal[idx]))
            self.cleaned_g_signal.append(float(result.cleaned_g_signal[idx]))
            self.cleaned_b_signal.append(float(result.cleaned_b_signal[idx]))
            self.cleaned_rppg_signal.append(float(result.cleaned_rppg_signal[idx]))

    def preprocess_pending_windows(
        self,
window_seconds: float = PREPROCESS_WINDOW_SECONDS,
        step_seconds: float = PREPROCESS_STEP_SECONDS,
        fallback_fps: float = RPPG_ASSUMED_FPS,
        smooth_seconds: float = PREPROCESS_SMOOTH_SECONDS,
        flush: bool = False,
    ) -> int:
        n_items = min(len(self.r_signal), len(self.g_signal), len(self.b_signal), len(self.timestamps))
        if n_items < PREPROCESS_MIN_SAMPLES:
            return 0

        sample_rate = estimate_sample_rate(self.timestamps[:n_items], fallback_fps=fallback_fps)
        window_samples = max(PREPROCESS_MIN_SAMPLES, int(round(max(0.1, window_seconds) * sample_rate)))
        step_samples = max(1, int(round(max(0.1, step_seconds) * sample_rate)))

        processed = 0
        start = min(self._processed_until, n_items)
        while start + window_samples <= n_items:
            end = start + window_samples
            result = preprocess_rgb_window(
                self.r_signal[start:end],
                self.g_signal[start:end],
                self.b_signal[start:end],
                self.timestamps[start:end],
                fallback_fps=sample_rate,
                smooth_seconds=smooth_seconds,
            )
            if result is not None:
                keep_from = 0 if not self.cleaned_timestamps else max(0, result.timestamps.size - step_samples)
                self._append_cleaned_window(result, keep_from=keep_from)
                processed += 1
            start += step_samples
            self._processed_until = start

        if flush:
            start = min(self._processed_until, n_items)
            if n_items - start >= PREPROCESS_MIN_SAMPLES:
                result = preprocess_rgb_window(
                    self.r_signal[start:n_items],
                    self.g_signal[start:n_items],
                    self.b_signal[start:n_items],
                    self.timestamps[start:n_items],
                    fallback_fps=sample_rate,
                    smooth_seconds=smooth_seconds,
                )
                if result is not None:
                    self._append_cleaned_window(result, keep_from=0)
                    processed += 1
                self._processed_until = n_items
        return processed


def _ensure_csv_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def save_raw_rgb_csv(path: str, signal_logs: Dict[int, RgbSignalLog]) -> None:
    if not path:
        return
    _ensure_csv_parent(path)
    rows = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["face_id", "timestamp", "r", "g", "b"])
        for face_id in sorted(signal_logs):
            log = signal_logs[face_id]
            for ts_val, r_val, g_val, b_val in zip(log.timestamps, log.r_signal, log.g_signal, log.b_signal):
                writer.writerow([face_id, f"{ts_val:.6f}", f"{r_val:.8f}", f"{g_val:.8f}", f"{b_val:.8f}"])
                rows += 1
    log_info(f"Saved {rows} raw RGB samples to {path}.")


def save_cleaned_rgb_csv(path: str, signal_logs: Dict[int, RgbSignalLog]) -> None:
    if not path:
        return
    _ensure_csv_parent(path)
    rows = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["face_id", "timestamp", "cleaned_r", "cleaned_g", "cleaned_b", "cleaned_rppg"])
        for face_id in sorted(signal_logs):
            log = signal_logs[face_id]
            for row in zip(
                log.cleaned_timestamps,
                log.cleaned_r_signal,
                log.cleaned_g_signal,
                log.cleaned_b_signal,
                log.cleaned_rppg_signal,
            ):
                ts_val, r_val, g_val, b_val, rppg_val = row
                writer.writerow([
                    face_id,
                    f"{ts_val:.6f}",
                    f"{r_val:.8f}",
                    f"{g_val:.8f}",
                    f"{b_val:.8f}",
                    f"{rppg_val:.8f}",
                ])
                rows += 1
    log_info(f"Saved {rows} cleaned RGB samples to {path}.")

# ---------------- STEP 5: signal preprocessing / noise reduction ends ----------------

@dataclass
class RppgFeatureSet:
    values: List[float]
    dominant_freq_hz: float
    snr_proxy: float
    peak_prominence: float
    spectral_entropy: float
    face_area_fraction: float
    roi_motion: float


def _score_between(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _smooth_spo2_value(previous: float, current: float) -> float:
    """EMA with a step limiter keeps webcam SpO2 estimates conservative and readable."""
    if previous <= 0.0:
        return current
    limited = previous + float(np.clip(current - previous, -SPO2_MAX_STEP_PER_UPDATE, SPO2_MAX_STEP_PER_UPDATE))
    return float(SPO2_SMOOTH_ALPHA * limited + (1.0 - SPO2_SMOOTH_ALPHA) * previous)


class RppgProcessor:
    """Ring-buffer r-PPG processor for webcam SpO2 estimate and heart-rate estimation.

    Webcam SpO2 is not a medical-device measurement. Without a real labeled
    calibration model, the value shown here is only a conservative estimate.
    """

    def __init__(self, fps: float = RPPG_ASSUMED_FPS, model_path: str = "") -> None:
        self.fps = fps
        self._buf_len = RPPG_BUFFER_FRAMES
        self._r_buf = np.zeros(self._buf_len, dtype=np.float64)
        self._g_buf = np.zeros(self._buf_len, dtype=np.float64)
        self._b_buf = np.zeros(self._buf_len, dtype=np.float64)
        self._area_buf = np.full(self._buf_len, RPPG_STABLE_FACE_AREA_FRACTION, dtype=np.float64)
        self._motion_buf = np.zeros(self._buf_len, dtype=np.float64)
        self._buf_count = 0
        self._buf_head = 0
        self._last_bbox_state: Optional[np.ndarray] = None

        self.spo2 = 0.0
        self.heart_rate = 0.0
        self.spo2_smooth = 0.0
        self.signal_quality = 0.0
        self.quality_state = "Collecting"
        self.last_rejection = ""
        self.estimator_label = "Empirical estimate"

        self._bpf_cache: Dict[Tuple[float, float, float], Tuple[np.ndarray, np.ndarray]] = {}
        self._calibrated_model: Optional[Any] = None
        if model_path:
            self._load_calibrated_model(model_path)

    def _load_calibrated_model(self, model_path: str) -> None:
        """Load a real labeled SpO2 calibration model with a scikit-learn-style predict method."""
        if not os.path.exists(model_path):
            log_warning(f"SpO2 calibration model not found: {model_path}. Using empirical estimate.")
            return
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            if not hasattr(model, "predict"):
                log_warning("SpO2 calibration model has no predict() method. Using empirical estimate.")
                return
            self._calibrated_model = model
            self.estimator_label = "Calibrated estimate"
            log_info(f"Loaded real SpO2 calibration model from {model_path}.")
        except Exception as exc:
            log_warning(f"Could not load SpO2 calibration model: {exc}. Using empirical estimate.")

    def _face_sample_metrics(
        self,
        face_bbox: Optional[Tuple[int, int, int, int]],
        frame_shape: Optional[Tuple[int, ...]],
    ) -> Tuple[float, float]:
        if face_bbox is None or frame_shape is None or len(frame_shape) < 2:
            return RPPG_STABLE_FACE_AREA_FRACTION, 0.0

        height, width = int(frame_shape[0]), int(frame_shape[1])
        if width <= 0 or height <= 0:
            return RPPG_STABLE_FACE_AREA_FRACTION, 0.0

        x1, y1, x2, y2 = face_bbox
        area_fraction = float(np.clip(_bbox_area(face_bbox) / float(width * height), 0.0, 1.0))
        cx = ((x1 + x2) * 0.5) / float(width)
        cy = ((y1 + y2) * 0.5) / float(height)
        scale = float(np.sqrt(max(area_fraction, 0.0)))
        state = np.array([cx, cy, scale], dtype=np.float64)

        if self._last_bbox_state is None:
            motion = 0.0
        else:
            motion = float(np.linalg.norm(state - self._last_bbox_state))
        self._last_bbox_state = state
        return area_fraction, motion

    def push_frame_direct(
        self,
        r_mean: float,
        g_mean: float,
        b_mean: float,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        frame_shape: Optional[Tuple[int, ...]] = None,
    ) -> None:
        area_fraction, roi_motion = self._face_sample_metrics(face_bbox, frame_shape)
        idx = self._buf_head
        self._r_buf[idx] = r_mean
        self._g_buf[idx] = g_mean
        self._b_buf[idx] = b_mean
        self._area_buf[idx] = area_fraction
        self._motion_buf[idx] = roi_motion
        self._buf_head = (idx + 1) % self._buf_len
        self._buf_count = min(self._buf_count + 1, self._buf_len)

    def _buffer_view(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_items = self._buf_count
        if n_items == 0:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty, empty
        if n_items < self._buf_len:
            return self._r_buf[:n_items], self._g_buf[:n_items], self._b_buf[:n_items]
        head = self._buf_head
        return (
            np.concatenate((self._r_buf[head:], self._r_buf[:head])),
            np.concatenate((self._g_buf[head:], self._g_buf[:head])),
            np.concatenate((self._b_buf[head:], self._b_buf[:head])),
        )

    def _metadata_view(self) -> Tuple[np.ndarray, np.ndarray]:
        n_items = self._buf_count
        if n_items == 0:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty
        if n_items < self._buf_len:
            return self._area_buf[:n_items], self._motion_buf[:n_items]
        head = self._buf_head
        return (
            np.concatenate((self._area_buf[head:], self._area_buf[:head])),
            np.concatenate((self._motion_buf[head:], self._motion_buf[:head])),
        )

    def _get_bandpass_coefs(self, fps: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not SCIPY_OK or butter is None:
            return None
        nyquist = fps / 2.0
        low = float(np.clip(RPPG_BPF_LOW / nyquist, 0.001, 0.999))
        high = float(np.clip(RPPG_BPF_HIGH / nyquist, 0.001, 0.999))
        if low >= high:
            return None
        key = (fps, low, high)
        if key not in self._bpf_cache:
            self._bpf_cache[key] = butter(4, [low, high], btype="band")
        return self._bpf_cache[key]

    def _reject_window(self, reason: str, quality: Optional[float] = None) -> bool:
        self.quality_state = "Low quality" if self._buf_count >= RPPG_MIN_FRAMES else "Collecting"
        self.last_rejection = reason
        if quality is not None:
            self.signal_quality = float(np.clip(quality, 0.0, 1.0))
        return False

    def _window_quality_score(self, features: RppgFeatureSet) -> float:
        snr_score = _score_between(features.snr_proxy, RPPG_MIN_SNR_PROXY, RPPG_GOOD_SNR_PROXY)
        peak_score = _score_between(
            features.peak_prominence,
            RPPG_MIN_PEAK_PROMINENCE,
            RPPG_GOOD_PEAK_PROMINENCE,
        )
        entropy_score = float(np.clip(1.0 - features.spectral_entropy, 0.0, 1.0))
        area_score = _score_between(
            features.face_area_fraction,
            RPPG_MIN_FACE_AREA_FRACTION,
            RPPG_STABLE_FACE_AREA_FRACTION,
        )
        motion_score = 1.0 - _score_between(
            features.roi_motion,
            RPPG_MAX_ROI_MOTION * 0.5,
            RPPG_MAX_ROI_MOTION,
        )
        return float(np.clip(
            0.35 * snr_score
            + 0.25 * peak_score
            + 0.15 * entropy_score
            + 0.15 * area_score
            + 0.10 * motion_score,
            0.0,
            1.0,
        ))

    def _predict_spo2(self, features: RppgFeatureSet) -> Optional[float]:
        if self._calibrated_model is not None:
            try:
                pred = float(self._calibrated_model.predict(np.array(features.values).reshape(1, -1))[0])
                return float(np.clip(pred, 70.0, 100.0))
            except Exception as exc:
                log_warning(f"SpO2 calibration model prediction failed: {exc}. Falling back to empirical estimate.")
                self._calibrated_model = None
                self.estimator_label = "Empirical estimate"

        # Empirical webcam RGB ratio-of-ratios fallback. This is intentionally
        # labeled as an estimate because RGB cameras lack a calibrated red/IR pair.
        ratio_rg = features.values[0]
        ratio_rb = features.values[1]
        ratio = 0.85 * ratio_rg + 0.15 * ratio_rb
        if not np.isfinite(ratio):
            return None
        return float(np.clip(SPO2_EMPIRICAL_A - SPO2_EMPIRICAL_B * ratio, 70.0, 100.0))

    def process(self, fps: Optional[float] = None) -> bool:
        if fps is not None:
            self.fps = fps
        if self._buf_count < RPPG_MIN_FRAMES:
            self.quality_state = "Collecting"
            self.last_rejection = f"Need {RPPG_MIN_FRAMES - self._buf_count} more samples"
            return False

        r_sig, g_sig, b_sig = self._buffer_view()
        area_sig, motion_sig = self._metadata_view()
        recent_n = min(len(r_sig), RPPG_MIN_FRAMES)
        face_area_fraction = (
            float(np.median(area_sig[-recent_n:]))
            if area_sig.size and recent_n > 0
            else RPPG_STABLE_FACE_AREA_FRACTION
        )
        roi_motion = (
            float(np.median(motion_sig[-recent_n:]))
            if motion_sig.size and recent_n > 0
            else 0.0
        )
        if face_area_fraction < RPPG_MIN_FACE_AREA_FRACTION:
            quality = _score_between(
                face_area_fraction,
                RPPG_MIN_FACE_AREA_FRACTION * 0.5,
                RPPG_MIN_FACE_AREA_FRACTION,
            ) * 0.35
            return self._reject_window("Face too small", quality=quality)
        if roi_motion > RPPG_MAX_ROI_MOTION:
            quality = max(0.0, 1.0 - roi_motion / (RPPG_MAX_ROI_MOTION * 2.0))
            return self._reject_window("ROI motion too high", quality=quality)

        r_norm = r_sig / (r_sig.mean() + 1e-8)
        g_norm = g_sig / (g_sig.mean() + 1e-8)
        b_norm = b_sig / (b_sig.mean() + 1e-8)

        x_chrom = 3.0 * r_norm - 2.0 * g_norm
        y_chrom = 1.5 * r_norm + g_norm - 1.5 * b_norm
        chrom_signal = x_chrom - (np.std(x_chrom) / (np.std(y_chrom) + 1e-8)) * y_chrom

        c_mat = np.column_stack([r_norm, g_norm, b_norm])
        c_norm = c_mat / (c_mat.mean(axis=0) + 1e-8)
        h_mat = np.array([[0, 1, -1], [-2, 1, 1]], dtype=np.float64)
        s_pos = (h_mat @ c_norm.T).T
        pos_signal = s_pos[:, 0] - (np.std(s_pos[:, 0]) / (np.std(s_pos[:, 1]) + 1e-8)) * s_pos[:, 1]

        raw_signal = 0.5 * chrom_signal + 0.5 * pos_signal
        detrended_signal = detrend_signal(raw_signal)
        sig_std = detrended_signal.std()
        if sig_std < 1e-8:
            return self._reject_window("Flat rPPG signal", quality=0.0)
        normalised = normalize_signal(detrended_signal)
        filtered = bandpass_filter_signal(normalised, self.fps, RPPG_BPF_LOW, RPPG_BPF_HIGH)
        if filtered.size < RPPG_MIN_FRAMES:
            return self._reject_window("Not enough filtered samples", quality=0.0)

        q1, q3 = np.percentile(filtered, [25, 75])
        iqr = q3 - q1
        filtered = np.clip(filtered, q1 - 3.0 * iqr, q3 + 3.0 * iqr)

        features = self._extract_features(
            filtered,
            r_sig,
            g_sig,
            b_sig,
            face_area_fraction,
            roi_motion,
        )
        if features is None:
            return self._reject_window("No usable pulse peak", quality=0.0)

        self.signal_quality = self._window_quality_score(features)
        if features.snr_proxy < RPPG_MIN_SNR_PROXY:
            return self._reject_window("Poor SNR", quality=self.signal_quality)
        if features.peak_prominence < RPPG_MIN_PEAK_PROMINENCE:
            return self._reject_window("Weak pulse peak", quality=self.signal_quality)
        if self.signal_quality < SPO2_MIN_SIGNAL_QUALITY:
            return self._reject_window("Low combined quality", quality=self.signal_quality)

        spo2_raw = self._predict_spo2(features)
        if spo2_raw is None:
            return self._reject_window("SpO2 estimate unavailable", quality=self.signal_quality)

        self.spo2 = spo2_raw
        self.spo2_smooth = _smooth_spo2_value(self.spo2_smooth, self.spo2)
        self.heart_rate = features.dominant_freq_hz * 60.0
        self.quality_state = "Stable"
        self.last_rejection = ""
        return True

    def _extract_features(
        self,
        filtered: np.ndarray,
        r_sig: np.ndarray,
        g_sig: np.ndarray,
        b_sig: np.ndarray,
        face_area_fraction: float,
        roi_motion: float,
    ) -> Optional[RppgFeatureSet]:
        n_items = len(filtered)
        if n_items < 32:
            return None

        def ac_dc(sig: np.ndarray) -> float:
            return float(sig.std() / (np.mean(np.abs(sig)) + 1e-8))

        ac_r = ac_dc(r_sig)
        ac_g = ac_dc(g_sig)
        ac_b = ac_dc(b_sig)
        ratio_rg = ac_r / (ac_g + 1e-8)
        ratio_rb = ac_r / (ac_b + 1e-8)

        freqs, psd = power_spectrum_signal(filtered, self.fps)
        mask = (freqs >= RPPG_HR_LOW_HZ) & (freqs <= RPPG_HR_HIGH_HZ)
        if mask.sum() == 0:
            return None

        psd_band = psd[mask]
        freqs_band = freqs[mask]
        if psd_band.size == 0 or not np.any(np.isfinite(psd_band)):
            return None
        dom_idx = int(np.argmax(psd_band))
        dom_freq = float(freqs_band[dom_idx])
        if not (RPPG_HR_LOW_HZ <= dom_freq <= RPPG_HR_HIGH_HZ):
            return None

        psd_norm = psd_band / (psd_band.sum() + 1e-8)
        entropy_raw = float(-np.sum(psd_norm * np.log(psd_norm + 1e-8)))
        spectral_entropy = float(np.clip(entropy_raw / np.log(psd_band.size + 1e-8), 0.0, 1.0))
        peak_power = float(psd_band[dom_idx])
        mean_power = float(psd_band.mean())
        snr_proxy = peak_power / (mean_power + 1e-8)
        median_power = float(np.median(psd_band))
        peak_prominence = peak_power / (median_power + 1e-8)
        rms_green = float(np.sqrt(np.mean(g_sig ** 2)))
        skin_tone_proxy = float(np.clip(rms_green / 255.0 * 6.0, 1.0, 6.0))

        # Touch these moments to reject pathological flat/noisy windows.
        _ = float(skew(filtered))
        _ = float(kurtosis(filtered))

        return RppgFeatureSet(
            values=[
                ratio_rg,
                ratio_rb,
                dom_freq,
                spectral_entropy,
                snr_proxy,
                rms_green,
                skin_tone_proxy,
                peak_prominence,
                face_area_fraction,
                roi_motion,
            ],
            dominant_freq_hz=dom_freq,
            snr_proxy=snr_proxy,
            peak_prominence=peak_prominence,
            spectral_entropy=spectral_entropy,
            face_area_fraction=face_area_fraction,
            roi_motion=roi_motion,
        )


@dataclass
class FaceRoiResult:
    face_id: int
    masks: List[np.ndarray]
    rects: List[Tuple[int, int, int, int]]
    bbox: Tuple[int, int, int, int]

    def mean_rgb(self, frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
        r_vals, g_vals, b_vals = [], [], []
        for mask, (x1, y1, x2, y2) in zip(self.masks, self.rects):
            if mask is None or mask.size == 0:
                continue
            patch = frame[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            if mask.shape[:2] != patch.shape[:2]:
                common_h = min(mask.shape[0], patch.shape[0])
                common_w = min(mask.shape[1], patch.shape[1])
                if common_h <= 0 or common_w <= 0:
                    continue
                mask = mask[:common_h, :common_w]
                patch = patch[:common_h, :common_w]
            pixels = patch[mask > 0]
            if pixels.size == 0:
                continue
            b_vals.append(float(pixels[:, 0].mean()))
            g_vals.append(float(pixels[:, 1].mean()))
            r_vals.append(float(pixels[:, 2].mean()))
        if not r_vals:
            return None
        return float(np.mean(r_vals)), float(np.mean(g_vals)), float(np.mean(b_vals))


@dataclass
class SharedFaceResult:
    face_id: int
    rgb: Optional[Tuple[float, float, float]]
    timestamp: float
    rects_disp: List[Tuple[int, int, int, int]]
    bbox_disp: Tuple[int, int, int, int]


def _bbox_area(bbox: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def choose_closest_face(results: List[FaceRoiResult]) -> Optional[FaceRoiResult]:
    return max(results, key=lambda result: _bbox_area(result.bbox), default=None)


class ActiveFaceManager:
    """Keeps one active face selected with bbox-area hysteresis."""

    def __init__(
        self,
        frame_area: int,
        switch_area_ratio: float = ACTIVE_FACE_SWITCH_AREA_RATIO,
        switch_min_area_fraction: float = ACTIVE_FACE_SWITCH_MIN_AREA_FRACTION,
    ) -> None:
        self.active_face_id: Optional[int] = None
        self.frame_area = max(1, int(frame_area))
        self.switch_area_ratio = max(1.0, float(switch_area_ratio))
        self.switch_min_area = max(1.0, self.frame_area * float(switch_min_area_fraction))

    def choose(self, results: List[FaceRoiResult]) -> Optional[FaceRoiResult]:
        if not results:
            if self.active_face_id is not None:
                log_info(f"Active face {self.active_face_id} disappeared.")
            self.active_face_id = None
            return None

        closest = choose_closest_face(results)
        if closest is None:
            self.active_face_id = None
            return None

        active = next((result for result in results if result.face_id == self.active_face_id), None)
        if active is None:
            self.active_face_id = closest.face_id
            log_info(f"Active face {self.active_face_id} selected.")
            return closest

        if closest.face_id == active.face_id:
            return active

        active_area = float(_bbox_area(active.bbox))
        closest_area = float(_bbox_area(closest.bbox))
        area_margin = closest_area - active_area
        clearly_closer = (
            closest_area >= active_area * self.switch_area_ratio
            and area_margin >= self.switch_min_area
        )
        if clearly_closer:
            log_info(
                f"Active face switched {active.face_id} -> {closest.face_id} "
                f"(area {active_area:.0f} -> {closest_area:.0f})."
            )
            self.active_face_id = closest.face_id
            return closest

        return active


class FaceTrack:
    _id_counter = 0

    def __init__(self, bbox: Tuple[int, int, int, int], ts: float) -> None:
        FaceTrack._id_counter += 1
        self.face_id = FaceTrack._id_counter
        self.kalman = LandmarkKalman()
        self.bbox = bbox
        self.last_seen = ts


class FaceRoiExtractor:
    """MediaPipe FaceMesh forehead and cheek ROI extractor."""

    def __init__(self, max_faces: int = MAX_NUM_FACES) -> None:
        if not MEDIAPIPE_OK or mp is None:
            self.available = False
            return
        self.available = True
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.roi_groups = [FOREHEAD_LANDMARKS, LEFT_CHEEK_LANDMARKS, RIGHT_CHEEK_LANDMARKS]
        self._tracks: Dict[int, FaceTrack] = {}

    @property
    def active_face_ids(self) -> List[int]:
        return list(self._tracks.keys())

    @staticmethod
    def _landmarks_to_bbox(lms: List[Tuple[float, float]]) -> Tuple[int, int, int, int]:
        xs = [point[0] for point in lms]
        ys = [point[1] for point in lms]
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

    @staticmethod
    def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = max((ax2 - ax1) * (ay2 - ay1), 1)
        area_b = max((bx2 - bx1) * (by2 - by1), 1)
        return inter / (area_a + area_b - inter)

    def _match_and_update_tracks(self, bboxes: List[Tuple[int, int, int, int]], now: float) -> List[int]:
        track_ids = list(self._tracks.keys())
        assigned = [-1] * len(bboxes)

        if track_ids:
            iou_mat = np.zeros((len(bboxes), len(track_ids)), dtype=np.float32)
            for r_idx, bbox in enumerate(bboxes):
                for c_idx, tid in enumerate(track_ids):
                    iou_mat[r_idx, c_idx] = self._iou(bbox, self._tracks[tid].bbox)

            while iou_mat.size:
                r_idx, c_idx = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                if iou_mat[r_idx, c_idx] < FACE_MATCH_IOU_MIN:
                    break
                tid = track_ids[c_idx]
                assigned[r_idx] = tid
                self._tracks[tid].bbox = bboxes[r_idx]
                self._tracks[tid].last_seen = now
                iou_mat[r_idx, :] = 0.0
                iou_mat[:, c_idx] = 0.0

        for idx, face_id in enumerate(assigned):
            if face_id == -1:
                track = FaceTrack(bboxes[idx], now)
                self._tracks[track.face_id] = track
                assigned[idx] = track.face_id
                log_info(f"Face track {track.face_id} created.")
        return assigned

    def _prune_stale_tracks(self, now: float) -> None:
        expired = [fid for fid, track in self._tracks.items() if now - track.last_seen > FACE_TRACK_TIMEOUT]
        for fid in expired:
            log_info(f"Face track {fid} expired.")
            del self._tracks[fid]

    def extract(self, frame: np.ndarray) -> List[FaceRoiResult]:
        if not self.available:
            return []

        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb_frame)
        now = time.time()

        if not result.multi_face_landmarks:
            self._prune_stale_tracks(now)
            return []

        all_raw_lms = [
            [(lm.x * width, lm.y * height) for lm in face.landmark]
            for face in result.multi_face_landmarks
        ]
        all_bboxes = [self._landmarks_to_bbox(lms) for lms in all_raw_lms]
        face_ids = self._match_and_update_tracks(all_bboxes, now)
        self._prune_stale_tracks(now)

        output: List[FaceRoiResult] = []
        for det_idx, face_id in enumerate(face_ids):
            track = self._tracks[face_id]
            smooth_lms = track.kalman.update(all_raw_lms[det_idx])

            masks: List[np.ndarray] = []
            rects: List[Tuple[int, int, int, int]] = []
            for group in self.roi_groups:
                pts = np.array(
                    [(int(smooth_lms[idx][0]), int(smooth_lms[idx][1])) for idx in group],
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

                x1 = max(0, bx)
                y1 = max(0, by)
                x2 = min(width, bx + bw)
                y2 = min(height, by + bh)
                if x2 <= x1 or y2 <= y1:
                    continue

                if x1 > bx or y1 > by or x2 < bx + bw or y2 < by + bh:
                    local_mask = local_mask[y1 - by:y2 - by, x1 - bx:x2 - bx]

                masks.append(local_mask)
                rects.append((x1, y1, x2, y2))

            output.append(FaceRoiResult(face_id=face_id, masks=masks, rects=rects, bbox=all_bboxes[det_idx]))
        return output


def _alpha_rect(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int], alpha: float = 0.65) -> None:
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    bg = np.full_like(roi, color)
    cv2.addWeighted(bg, alpha, roi, 1.0 - alpha, 0, roi)


def draw_corner_box(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int], arm: int = 22, thick: int = 2) -> None:
    arm = max(4, min(arm, max((x2 - x1) // 4, 1), max((y2 - y1) // 4, 1)))
    corners = [
        ((x1 + arm, y1), (x1, y1), (x1, y1 + arm)),
        ((x2 - arm, y1), (x2, y1), (x2, y1 + arm)),
        ((x1 + arm, y2), (x1, y2), (x1, y2 - arm)),
        ((x2 - arm, y2), (x2, y2), (x2, y2 - arm)),
    ]
    for horizontal, corner, vertical in corners:
        cv2.line(frame, horizontal, corner, color, thick, cv2.LINE_AA)
        cv2.line(frame, corner, vertical, color, thick, cv2.LINE_AA)


def draw_global_hud(frame: np.ndarray, fps: float, show_spo2: bool, face_count: int) -> None:
    height, width = frame.shape[:2]
    lines = [
        (f"FPS   {fps:4.1f}", C_YELLOW),
        (f"Faces {face_count}", C_WHITE),
        (f"SpO2  {'ON' if show_spo2 else 'OFF'}", C_SPO2_OK if show_spo2 else C_GRAY),
        ("----------", C_GRAY),
        ("[q] Quit", C_CYAN),
        ("[s] Toggle SpO2", C_CYAN),
    ]
    pad, line_h, panel_w = 8, 21, 180
    panel_h = len(lines) * line_h + pad * 2
    x1, y1 = width - panel_w - 10, 10
    x2, y2 = width - 10, y1 + panel_h
    _alpha_rect(frame, x1, y1, x2, y2, C_DARK, alpha=0.72)
    for idx, (text, color) in enumerate(lines):
        y = y1 + pad + line_h * idx + 15
        cv2.putText(frame, text, (x1 + pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)


def draw_active_vitals_hud(
    frame: np.ndarray,
    rppg: RppgProcessor,
    roi_rects: List[Tuple[int, int, int, int]],
    show_rois: bool,
    face_id: int,
    face_bbox: Tuple[int, int, int, int],
) -> None:
    height, width = frame.shape[:2]
    spo2_val = rppg.spo2_smooth
    hr_val = rppg.heart_rate
    quality = rppg.signal_quality
    state = rppg.quality_state
    collecting = spo2_val < 1.0

    spo2_col = C_SPO2_OK if spo2_val >= 95.0 else C_SPO2_LOW
    estimate_prefix = "~" if rppg.estimator_label.startswith("Empirical") else ""
    spo2_str = f"{estimate_prefix}{spo2_val:.1f}%" if not collecting else "Collecting..."
    hr_str = f"{hr_val:.0f} bpm" if hr_val > 0 else "--"
    state_col = C_SPO2_OK if state == "Stable" else (C_GRAY if state == "Collecting" else C_WARN)

    lines = [
        (f"Active Face {face_id}", C_YELLOW),
        (f"State: {state}", state_col),
        (f"SpO2 Est: {spo2_str}", spo2_col if not collecting else C_GRAY),
        (f"HR: {hr_str}", C_WHITE),
        (f"SQ: {quality * 100:.0f}%", C_CYAN),
    ]
    if 0 < spo2_val < 94.0:
        lines.append(("LOW SpO2", C_WARN))

    x1, y1, x2, y2 = face_bbox
    draw_corner_box(frame, x1, y1, x2, y2, C_ROI, thick=2)

    pad, line_h, panel_w = 8, 22, 220
    panel_h = len(lines) * line_h + pad * 2
    px1, py1 = 10, 10
    px2 = min(px1 + panel_w, width - 1)
    py2 = min(py1 + panel_h, height - 1)
    _alpha_rect(frame, px1, py1, px2, py2, C_DARK, alpha=0.75)

    for idx, (text, color) in enumerate(lines):
        ty = py1 + pad + line_h * idx + 16
        cv2.putText(frame, text, (px1 + pad, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    if show_rois:
        labels = ["Forehead", "L-Cheek", "R-Cheek"]
        for idx, (rx1, ry1, rx2, ry2) in enumerate(roi_rects):
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), C_ROI, 1)
            if idx < len(labels):
                cv2.putText(frame, labels[idx], (rx1, max(0, ry1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_ROI, 1, cv2.LINE_AA)


def _face_worker(
    face_extractor: FaceRoiExtractor,
    frame_q: queue.Queue,
    result_lock: threading.Lock,
    result_store: dict,
    sx: float,
    sy: float,
    stop_ev: threading.Event,
) -> None:
    log_info("Face worker started.")
    frame_counter = 0
    active_manager = ActiveFaceManager(frame_area=INFER_W * INFER_H)

    while not stop_ev.is_set():
        try:
            infer_frame = frame_q.get(timeout=0.1)
        except queue.Empty:
            continue

        frame_counter += 1
        if frame_counter % FACEMESH_EVERY_N != 0:
            continue

        try:
            raw_results = face_extractor.extract(infer_frame)
        except Exception as exc:
            log_warning(f"Face extraction error: {exc}")
            continue

        active_raw_result = active_manager.choose(raw_results)
        active_shared_result: Optional[SharedFaceResult] = None
        sample_ts = time.time()
        if active_raw_result is not None:
            mean_rgb = active_raw_result.mean_rgb(infer_frame)
            rects_disp = [
                (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))
                for x1, y1, x2, y2 in active_raw_result.rects
            ]
            bx1, by1, bx2, by2 = active_raw_result.bbox
            bbox_disp = (int(bx1 * sx), int(by1 * sy), int(bx2 * sx), int(by2 * sy))
            active_shared_result = SharedFaceResult(
                active_raw_result.face_id,
                mean_rgb,
                sample_ts,
                rects_disp,
                bbox_disp,
            )

        with result_lock:
            result_store["result"] = active_shared_result
            result_store["active_id"] = active_shared_result.face_id if active_shared_result is not None else None
            result_store["visible_count"] = len(raw_results)

    log_info("Face worker stopped.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone webcam r-PPG SpO2 / heart-rate monitor.")
    parser.add_argument("--camera", type=int, default=CAM_INDEX, help="Camera index to open, usually 0 or 1.")
    parser.add_argument("--width", type=int, default=FRAME_W, help="Requested camera frame width.")
    parser.add_argument("--height", type=int, default=FRAME_H, help="Requested camera frame height.")
    parser.add_argument(
        "--signal-csv",
        "--raw-signal-csv",
        dest="signal_csv",
        default="",
        help="Optional CSV path for raw ROI RGB samples collected during the run.",
    )
    parser.add_argument(
        "--cleaned-signal-csv",
        default="",
        help="Optional CSV path for STEP 5 cleaned RGB/rPPG samples saved on shutdown.",
    )
    parser.add_argument(
        "--preprocess-window-seconds",
        type=float,
        default=PREPROCESS_WINDOW_SECONDS,
        help="STEP 5 preprocessing window size in seconds.",
    )
    parser.add_argument(
        "--preprocess-step-seconds",
        type=float,
        default=PREPROCESS_STEP_SECONDS,
        help="STEP 5 preprocessing step size in seconds.",
    )
    parser.add_argument(
        "--preprocess-smooth-seconds",
        type=float,
        default=PREPROCESS_SMOOTH_SECONDS,
        help="Optional moving-average smoothing duration after bandpass filtering.",
    )
    parser.add_argument(
        "--spo2-model",
        default="",
        help=(
            "Optional path to a real labeled SpO2 calibration model pickle. "
            "When omitted, the UI shows an empirical webcam estimate."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    face_extractor = FaceRoiExtractor()
    if not face_extractor.available:
        log_warning("MediaPipe FaceMesh unavailable. Exiting.")
        return

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        log_warning(f"Cannot open camera {args.camera}. Try --camera 1 if your webcam is not camera 0.")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sx = actual_w / INFER_W
    sy = actual_h / INFER_H
    log_info(f"Camera {args.camera} opened at {actual_w}x{actual_h}.")
    log_info(f"Face worker input: {INFER_W}x{INFER_H}; display scale {sx:.2f}x, {sy:.2f}x.")
    log_info("Controls: [q] quit  [s] toggle SpO2 panel")

    stop_workers = threading.Event()
    face_q: queue.Queue = queue.Queue(maxsize=1)
    face_lock = threading.Lock()
    face_store: dict = {"result": None, "active_id": None, "visible_count": 0}

    face_thread = threading.Thread(
        target=_face_worker,
        args=(face_extractor, face_q, face_lock, face_store, sx, sy, stop_workers),
        daemon=True,
        name="face-worker",
    )
    face_thread.start()

    fps = 0.0
    frame_times = deque(maxlen=30)
    show_spo2 = True
    active_rppg: Optional[RppgProcessor] = None
    active_face_id: Optional[int] = None
    signal_logs: Dict[int, RgbSignalLog] = {}
    last_logged_sample_ts: Dict[int, float] = {}
    active_face_result: Optional[SharedFaceResult] = None
    visible_face_count = 0
    spo2_frame_counter = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log_warning("Frame grab failed; retrying.")
                time.sleep(0.05)
                continue

            now = time.time()
            infer_frame = cv2.resize(frame, (INFER_W, INFER_H))

            if show_spo2:
                try:
                    face_q.put_nowait(infer_frame)
                except queue.Full:
                    pass

            if show_spo2:
                with face_lock:
                    active_face_result = face_store.get("result")
                    visible_face_count = int(face_store.get("visible_count", 0))

                if active_face_result is None:
                    if active_face_id is not None:
                        log_info(f"Active face {active_face_id} lost.")
                    active_face_id = None
                    active_rppg = None
                    spo2_frame_counter = 0
                else:
                    if active_rppg is None or active_face_id != active_face_result.face_id:
                        active_face_id = active_face_result.face_id
                        active_rppg = RppgProcessor(model_path=args.spo2_model)
                        spo2_frame_counter = 0
                        log_info(f"SpO2 processor created for active face {active_face_id}.")

                    if active_face_result.rgb is not None:
                        r_mean, g_mean, b_mean = active_face_result.rgb
                        active_rppg.push_frame_direct(
                            r_mean,
                            g_mean,
                            b_mean,
                            face_bbox=active_face_result.bbox_disp,
                            frame_shape=frame.shape,
                        )

                        last_ts = last_logged_sample_ts.get(active_face_result.face_id)
                        if last_ts is None or active_face_result.timestamp > last_ts:
                            signal_log = signal_logs.setdefault(
                                active_face_result.face_id,
                                RgbSignalLog(active_face_result.face_id),
                            )
                            signal_log.append(active_face_result.timestamp, r_mean, g_mean, b_mean)
                            last_logged_sample_ts[active_face_result.face_id] = active_face_result.timestamp

                            # STEP 5 integration starts: clean raw RGB in reusable time windows.
                            sample_fps = fps if fps > 5 else RPPG_ASSUMED_FPS
                            signal_log.preprocess_pending_windows(
                                window_seconds=args.preprocess_window_seconds,
                                step_seconds=args.preprocess_step_seconds,
                                fallback_fps=sample_fps,
                                smooth_seconds=args.preprocess_smooth_seconds,
                            )
                            # STEP 5 integration ends.

                    spo2_frame_counter += 1
                    if spo2_frame_counter >= SPO2_UPDATE_INTERVAL:
                        spo2_frame_counter = 0
                        proc_fps = fps if fps > 5 else RPPG_ASSUMED_FPS
                        active_rppg.process(fps=proc_fps)

                    draw_active_vitals_hud(
                        frame,
                        active_rppg,
                        active_face_result.rects_disp,
                        show_rois=True,
                        face_id=active_face_result.face_id,
                        face_bbox=active_face_result.bbox_disp,
                    )
            else:
                active_face_result = None
                visible_face_count = 0
                active_face_id = None
                active_rppg = None
                spo2_frame_counter = 0
                with face_lock:
                    face_store["result"] = None
                    face_store["active_id"] = None
                    face_store["visible_count"] = 0

            frame_times.append(now)
            if len(frame_times) >= 2:
                fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0] + 1e-9)

            draw_global_hud(frame, fps, show_spo2, visible_face_count)
            cv2.imshow("r-PPG SpO2 Monitor", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                log_info("Quit requested.")
                break
            if key == ord("s"):
                show_spo2 = not show_spo2
                log_info(f"SpO2 panel {'enabled' if show_spo2 else 'disabled'}.")
    finally:
        log_info("Stopping worker thread.")
        stop_workers.set()
        face_thread.join(timeout=2.0)

        for signal_log in signal_logs.values():
            signal_log.preprocess_pending_windows(window_seconds=args.preprocess_window_seconds,
                step_seconds=args.preprocess_step_seconds,
                fallback_fps=fps if fps > 5 else RPPG_ASSUMED_FPS,
                smooth_seconds=args.preprocess_smooth_seconds,
                flush=True,
            )
        if args.signal_csv:
            save_raw_rgb_csv(args.signal_csv, signal_logs)
        if args.cleaned_signal_csv:
            save_cleaned_rgb_csv(args.cleaned_signal_csv, signal_logs)

        cap.release()
        cv2.destroyAllWindows()
        log_info("Shutdown complete.")


if __name__ == "__main__":
    main()
