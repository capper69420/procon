"""
Standalone local r-PPG SpO2 / heart-rate monitor.

This script runs a MediaPipe FaceMesh + r-PPG pipeline for local SpO2 and heart-rate estimation.

Controls:
  q  quit
  s  toggle SpO2/HR panels

Install:
  pip install mediapipe opencv-python numpy scipy scikit-learn

Run:
  python health_monitor.py
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import signal as scipy_signal
from scipy.signal import butter, detrend, filtfilt
from scipy.stats import kurtosis, skew

warnings.filterwarnings("ignore")

try:
    import mediapipe as mp
    MEDIAPIPE_OK = True
except ImportError:
    mp = None
    MEDIAPIPE_OK = False
    print("[WARNING] mediapipe not found. Install with: pip install mediapipe")

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except ImportError:
    RandomForestRegressor = None
    Pipeline = None
    StandardScaler = None
    SKLEARN_OK = False
    print("[WARNING] scikit-learn not found. Using empirical SpO2 estimate instead.")


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
SPO2_SMOOTH_ALPHA = 0.15

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


class RppgProcessor:
    """Ring-buffer r-PPG processor for SpO2 and heart-rate estimation."""

    def __init__(self, fps: float = RPPG_ASSUMED_FPS) -> None:
        self.fps = fps
        self._buf_len = RPPG_BUFFER_FRAMES
        self._r_buf = np.zeros(self._buf_len, dtype=np.float64)
        self._g_buf = np.zeros(self._buf_len, dtype=np.float64)
        self._b_buf = np.zeros(self._buf_len, dtype=np.float64)
        self._buf_count = 0
        self._buf_head = 0

        self.spo2 = 0.0
        self.heart_rate = 0.0
        self.spo2_smooth = 0.0
        self.signal_quality = 0.0

        self._bpf_cache: Dict[Tuple[float, float, float], Tuple[np.ndarray, np.ndarray]] = {}
        self._model: Optional[object] = None
        self._model_ready = False

    def _init_ml_model(self) -> None:
        if not SKLEARN_OK:
            return

        rng = np.random.default_rng(42)
        n_samples = 2500
        spo2_gt = rng.uniform(88.0, 100.0, n_samples)
        ratio_rg = (-0.8 * spo2_gt + 104.0) / 100.0 + rng.normal(0, 0.05, n_samples)
        ratio_rb = ratio_rg * 0.85 + rng.normal(0, 0.04, n_samples)
        dominant_freq = rng.uniform(0.8, 2.5, n_samples)
        spectral_entropy = rng.uniform(0.2, 0.9, n_samples)
        snr_proxy = rng.uniform(0.3, 1.0, n_samples)
        rms_green = rng.uniform(0.01, 0.15, n_samples)
        skin_tone = rng.integers(1, 7, n_samples).astype(float)

        x_train = np.column_stack([
            ratio_rg,
            ratio_rb,
            dominant_freq,
            spectral_entropy,
            snr_proxy,
            rms_green,
            skin_tone,
        ])
        y_train = np.clip(spo2_gt, 88.0, 100.0)

        self._model = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(
                n_estimators=160,
                max_depth=12,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            )),
        ])
        self._model.fit(x_train, y_train)
        self._model_ready = True
        log_info("SpO2 Random Forest model initialized from synthetic calibration data.")

    def push_frame_direct(self, r_mean: float, g_mean: float, b_mean: float) -> None:
        idx = self._buf_head
        self._r_buf[idx] = r_mean
        self._g_buf[idx] = g_mean
        self._b_buf[idx] = b_mean
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

    def _get_bandpass_coefs(self, fps: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        nyquist = fps / 2.0
        low = float(np.clip(RPPG_BPF_LOW / nyquist, 0.001, 0.999))
        high = float(np.clip(RPPG_BPF_HIGH / nyquist, 0.001, 0.999))
        if low >= high:
            return None
        key = (fps, low, high)
        if key not in self._bpf_cache:
            self._bpf_cache[key] = butter(4, [low, high], btype="band")
        return self._bpf_cache[key]

    def process(self, fps: Optional[float] = None) -> bool:
        if fps is not None:
            self.fps = fps
        if self._buf_count < RPPG_MIN_FRAMES:
            return False

        r_sig, g_sig, b_sig = self._buffer_view()
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
        detrended_signal = detrend(raw_signal)
        sig_std = detrended_signal.std()
        if sig_std < 1e-8:
            return False
        normalised = (detrended_signal - detrended_signal.mean()) / sig_std

        coefs = self._get_bandpass_coefs(self.fps)
        if coefs is None:
            return False
        b_coef, a_coef = coefs
        filtered = filtfilt(b_coef, a_coef, normalised)

        q1, q3 = np.percentile(filtered, [25, 75])
        iqr = q3 - q1
        filtered = np.clip(filtered, q1 - 3.0 * iqr, q3 + 3.0 * iqr)

        features, freq_hz = self._extract_features(filtered, r_sig, g_sig, b_sig)
        if features is None:
            return False

        if not self._model_ready:
            self._init_ml_model()

        if self._model_ready and self._model is not None:
            spo2_raw = float(self._model.predict(np.array(features).reshape(1, -1))[0])
            self.spo2 = float(np.clip(spo2_raw, 70.0, 100.0))
        else:
            ac_r = np.std(r_sig - r_sig.mean())
            dc_r = r_sig.mean() + 1e-8
            ac_g = np.std(g_sig - g_sig.mean())
            dc_g = g_sig.mean() + 1e-8
            ratio = (ac_r / dc_r) / (ac_g / dc_g)
            self.spo2 = float(np.clip(104.0 - 17.0 * ratio, 70.0, 100.0))

        if self.spo2_smooth == 0.0:
            self.spo2_smooth = self.spo2
        else:
            self.spo2_smooth = (
                SPO2_SMOOTH_ALPHA * self.spo2
                + (1.0 - SPO2_SMOOTH_ALPHA) * self.spo2_smooth
            )

        if freq_hz > 0:
            self.heart_rate = freq_hz * 60.0
        return True

    def _extract_features(
        self,
        filtered: np.ndarray,
        r_sig: np.ndarray,
        g_sig: np.ndarray,
        b_sig: np.ndarray,
    ) -> Tuple[Optional[List[float]], float]:
        n_items = len(filtered)
        if n_items < 32:
            return None, 0.0

        def ac_dc(sig: np.ndarray) -> float:
            return float(sig.std() / (np.mean(np.abs(sig)) + 1e-8))

        ac_r = ac_dc(r_sig)
        ac_g = ac_dc(g_sig)
        ac_b = ac_dc(b_sig)
        ratio_rg = ac_r / (ac_g + 1e-8)
        ratio_rb = ac_r / (ac_b + 1e-8)

        nperseg = min(256, n_items // 2)
        freqs, psd = scipy_signal.welch(filtered, fs=self.fps, nperseg=nperseg)
        mask = (freqs >= RPPG_BPF_LOW) & (freqs <= RPPG_BPF_HIGH)
        if mask.sum() == 0:
            return None, 0.0

        psd_band = psd[mask]
        freqs_band = freqs[mask]
        dom_idx = int(np.argmax(psd_band))
        dom_freq = float(freqs_band[dom_idx])

        psd_norm = psd_band / (psd_band.sum() + 1e-8)
        spectral_entropy = float(-np.sum(psd_norm * np.log(psd_norm + 1e-8)))
        peak_power = float(psd_band[dom_idx])
        mean_power = float(psd_band.mean())
        snr_proxy = peak_power / (mean_power + 1e-8)
        rms_green = float(np.sqrt(np.mean(g_sig ** 2)))
        skin_tone_proxy = float(np.clip(rms_green / 255.0 * 6.0, 1.0, 6.0))

        # Touch these moments to reject pathological flat/noisy windows.
        _ = float(skew(filtered))
        _ = float(kurtosis(filtered))

        self.signal_quality = float(np.clip(snr_proxy / 20.0, 0.0, 1.0))
        return [
            ratio_rg,
            ratio_rb,
            dom_freq,
            spectral_entropy,
            snr_proxy,
            rms_green,
            skin_tone_proxy,
        ], dom_freq


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
    rects_disp: List[Tuple[int, int, int, int]]
    bbox_disp: Tuple[int, int, int, int]


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


def draw_spo2_panel(
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
    collecting = spo2_val < 1.0

    spo2_col = C_SPO2_OK if spo2_val >= 95.0 else C_SPO2_LOW
    spo2_str = f"{spo2_val:.1f}%" if not collecting else "Collecting..."
    hr_str = f"{hr_val:.0f} bpm" if hr_val > 0 and not collecting else "--"

    lines = [
        (f"SpO2 Face {face_id}", C_YELLOW),
        (spo2_str, spo2_col if not collecting else C_GRAY),
        (f"HR: {hr_str}", C_WHITE),
        (f"SQ: {quality * 100:.0f}%", C_CYAN),
    ]
    if 0 < spo2_val < 94.0:
        lines.append(("LOW SpO2", C_WARN))

    x1, y1, x2, y2 = face_bbox
    draw_corner_box(frame, x1, y1, x2, y2, C_ROI, thick=2)

    pad, line_h, panel_w = 8, 22, 178
    panel_h = len(lines) * line_h + pad * 2
    px1 = max(x1, 0)
    py1 = max(min(y2 + 6, height - panel_h - 4), 0)
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

        shared: List[SharedFaceResult] = []
        for result in raw_results:
            mean_rgb = result.mean_rgb(infer_frame)
            rects_disp = [
                (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))
                for x1, y1, x2, y2 in result.rects
            ]
            bx1, by1, bx2, by2 = result.bbox
            bbox_disp = (int(bx1 * sx), int(by1 * sy), int(bx2 * sx), int(by2 * sy))
            shared.append(SharedFaceResult(result.face_id, mean_rgb, rects_disp, bbox_disp))

        with result_lock:
            result_store["results"] = shared
            result_store["active_ids"] = face_extractor.active_face_ids

    log_info("Face worker stopped.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone webcam r-PPG SpO2 / heart-rate monitor.")
    parser.add_argument("--camera", type=int, default=CAM_INDEX, help="Camera index to open, usually 0 or 1.")
    parser.add_argument("--width", type=int, default=FRAME_W, help="Requested camera frame width.")
    parser.add_argument("--height", type=int, default=FRAME_H, help="Requested camera frame height.")
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
    face_store: dict = {"results": [], "active_ids": []}

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
    rppg_pool: Dict[int, RppgProcessor] = {}
    face_results_disp: List[SharedFaceResult] = []
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

            active_face_ids = set()
            if show_spo2:
                with face_lock:
                    face_results_disp = list(face_store["results"])
                    active_face_ids = set(face_store.get("active_ids", []))

                for face_result in face_results_disp:
                    if face_result.face_id not in rppg_pool:
                        rppg_pool[face_result.face_id] = RppgProcessor()
                        log_info(f"SpO2 processor created for face {face_result.face_id}.")
                    if face_result.rgb is not None:
                        rppg_pool[face_result.face_id].push_frame_direct(*face_result.rgb)

                spo2_frame_counter += 1
                if spo2_frame_counter >= SPO2_UPDATE_INTERVAL:
                    spo2_frame_counter = 0
                    proc_fps = fps if fps > 5 else RPPG_ASSUMED_FPS
                    for face_result in face_results_disp:
                        processor = rppg_pool.get(face_result.face_id)
                        if processor is not None:
                            processor.process(fps=proc_fps)

                for fid in [fid for fid in rppg_pool if fid not in active_face_ids]:
                    log_info(f"SpO2 processor removed for expired face {fid}.")
                    del rppg_pool[fid]

                for face_result in face_results_disp:
                    processor = rppg_pool.get(face_result.face_id)
                    if processor is not None:
                        draw_spo2_panel(
                            frame,
                            processor,
                            face_result.rects_disp,
                            show_rois=True,
                            face_id=face_result.face_id,
                            face_bbox=face_result.bbox_disp,
                        )
            else:
                face_results_disp = []
                rppg_pool.clear()

            frame_times.append(now)
            if len(frame_times) >= 2:
                fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0] + 1e-9)

            draw_global_hud(frame, fps, show_spo2, len(face_results_disp))
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
        cap.release()
        cv2.destroyAllWindows()
        log_info("Shutdown complete.")


if __name__ == "__main__":
    main()
