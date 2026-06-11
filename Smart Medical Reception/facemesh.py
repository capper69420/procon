"""
facemesh.py — FaceMesh ROI extraction + r-PPG vitals for API use.

Extracted from health_monitor.py for the FastAPI backend.
Processes single frames (numpy BGR arrays or base64 strings).
"""

from __future__ import annotations

import base64
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import signal as scipy_signal
from scipy.signal import butter, filtfilt, detrend

warnings.filterwarnings("ignore")

try:
    import mediapipe as mp
    MEDIAPIPE_OK = True
except ImportError:
    MEDIAPIPE_OK = False

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False


# ── Constants ─────────────────────────────────────────────────────────────────
FOREHEAD_LANDMARKS = [10, 67, 69, 104, 108, 151, 337, 338, 297, 299]
LEFT_CHEEK_LANDMARKS = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148]
RIGHT_CHEEK_LANDMARKS = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377]
ROI_LANDMARK_INDICES: Tuple[int, ...] = tuple(sorted({
    *FOREHEAD_LANDMARKS, *LEFT_CHEEK_LANDMARKS, *RIGHT_CHEEK_LANDMARKS,
}))

_ERODE_KERNEL = np.ones((3, 3), dtype=np.uint8)
RPPG_BUFFER_FRAMES = 300
RPPG_BPF_LOW = 0.7
RPPG_BPF_HIGH = 4.0
RPPG_MIN_FRAMES = 90
RPPG_ASSUMED_FPS = 30.0
SPO2_SMOOTH_ALPHA = 0.15
MAX_NUM_FACES = 4
FACE_MATCH_IOU_MIN = 0.25
FACE_TRACK_TIMEOUT = 2.0
KALMAN_PROCESS_NOISE = 1e-5
KALMAN_MEASURE_NOISE = 1e-3

FALLEN_TRUNK_DEG = 55
UPRIGHT_TRUNK_DEG = 35
FALLEN_BBOX_RATIO = 1.25
STANDING_BBOX_RATIO = 1.50
KP_MIN_CONF = 0.30

YOLO_MODEL = "yolo11n-pose.pt"


class Posture:
    STANDING = "STANDING"
    SITTING = "SITTING"
    FALLEN = "FALLEN"
    UNKNOWN = "UNKNOWN"


# ── Kalman helpers ────────────────────────────────────────────────────────────
class KalmanStabiliser:
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
        k = p_pred / (p_pred + self.R)
        self.x = self.x + k * (measurement - self.x)
        self.P = (1.0 - k) * p_pred
        return self.x


class LandmarkKalman:
    def __init__(self, indices: Tuple[int, ...] = ROI_LANDMARK_INDICES) -> None:
        self._indices = indices
        self.kx: Dict[int, KalmanStabiliser] = {i: KalmanStabiliser() for i in indices}
        self.ky: Dict[int, KalmanStabiliser] = {i: KalmanStabiliser() for i in indices}

    def update(self, lm_list: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        out = list(lm_list)
        for i in self._indices:
            if i >= len(lm_list):
                continue
            x, y = lm_list[i]
            out[i] = (self.kx[i].update(x), self.ky[i].update(y))
        return out


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
        return (float(np.mean(r_vals)), float(np.mean(g_vals)), float(np.mean(b_vals)))


class FaceTrack:
    _id_counter: int = 0

    def __init__(self, bbox: Tuple[int, int, int, int], ts: float) -> None:
        FaceTrack._id_counter += 1
        self.face_id = FaceTrack._id_counter
        self.kalman = LandmarkKalman()
        self.bbox = bbox
        self.last_seen = ts


class FaceRoiExtractor:
    """MediaPipe FaceMesh — forehead + cheek ROIs for r-PPG."""

    def __init__(self, max_faces: int = MAX_NUM_FACES) -> None:
        self.available = MEDIAPIPE_OK
        if not MEDIAPIPE_OK:
            return
        self.max_faces = max_faces
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.roi_groups = [FOREHEAD_LANDMARKS, LEFT_CHEEK_LANDMARKS, RIGHT_CHEEK_LANDMARKS]
        self._tracks: Dict[int, FaceTrack] = {}

    @staticmethod
    def _landmarks_to_bbox(lms: List[Tuple[float, float]]) -> Tuple[int, int, int, int]:
        xs = [p[0] for p in lms]
        ys = [p[1] for p in lms]
        return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

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

    def _match_and_update_tracks(
        self, bboxes: List[Tuple[int, int, int, int]], now: float
    ) -> List[int]:
        track_ids = list(self._tracks.keys())
        assigned = [-1] * len(bboxes)
        if track_ids:
            n_det, n_trk = len(bboxes), len(track_ids)
            iou_mat = np.zeros((n_det, n_trk), dtype=np.float32)
            for r, bb in enumerate(bboxes):
                for c, tid in enumerate(track_ids):
                    iou_mat[r, c] = self._iou(bb, self._tracks[tid].bbox)
            while True:
                r, c = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                if iou_mat[r, c] < FACE_MATCH_IOU_MIN:
                    break
                tid = track_ids[c]
                assigned[r] = tid
                self._tracks[tid].bbox = bboxes[r]
                self._tracks[tid].last_seen = now
                iou_mat[r, :] = 0.0
                iou_mat[:, c] = 0.0
        for r, fid in enumerate(assigned):
            if fid == -1:
                t = FaceTrack(bboxes[r], now)
                self._tracks[t.face_id] = t
                assigned[r] = t.face_id
        return assigned

    def _prune_stale_tracks(self, now: float) -> None:
        expired = [fid for fid, t in self._tracks.items() if now - t.last_seen > FACE_TRACK_TIMEOUT]
        for fid in expired:
            del self._tracks[fid]

    def extract(self, frame: np.ndarray) -> List[FaceRoiResult]:
        if not self.available:
            return []
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)
        now = time.time()
        if not result.multi_face_landmarks:
            self._prune_stale_tracks(now)
            return []

        all_raw_lms = [
            [(lm.x * w, lm.y * h) for lm in face.landmark]
            for face in result.multi_face_landmarks
        ]
        all_bboxes = [self._landmarks_to_bbox(lms) for lms in all_raw_lms]
        face_ids = self._match_and_update_tracks(all_bboxes, now)
        self._prune_stale_tracks(now)

        output: List[FaceRoiResult] = []
        for det_idx, face_id in enumerate(face_ids):
            track = self._tracks[face_id]
            smooth_lms = track.kalman.update(all_raw_lms[det_idx])
            masks, rects = [], []
            for group in self.roi_groups:
                pts = np.array(
                    [(int(smooth_lms[i][0]), int(smooth_lms[i][1])) for i in group],
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
                x1, y1 = max(0, bx), max(0, by)
                x2, y2 = min(w, bx + bw), min(h, by + bh)
                masks.append(local_mask)
                rects.append((x1, y1, x2, y2))
            output.append(FaceRoiResult(face_id=face_id, masks=masks, rects=rects, bbox=all_bboxes[det_idx]))
        return output


class RppgProcessor:
    """Remote photoplethysmography — CHROM/POS pipeline + ML SpO₂ estimate."""

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
        self._bpf_cache: Dict = {}
        self._model = None
        self._model_ready = False

    def _init_ml_model(self) -> None:
        if not SKLEARN_OK:
            return
        np.random.seed(42)
        n = 2000
        spo2_gt = np.random.uniform(88.0, 100.0, n)
        ratio_rg = (-0.8 * spo2_gt + 104.0) / 100.0 + np.random.normal(0, 0.05, n)
        ratio_rb = ratio_rg * 0.85 + np.random.normal(0, 0.04, n)
        X = np.column_stack([
            ratio_rg, ratio_rb,
            np.random.uniform(0.8, 2.5, n),
            np.random.uniform(0.2, 0.9, n),
            np.random.uniform(0.3, 1.0, n),
            np.random.uniform(0.01, 0.15, n),
            np.random.uniform(1, 7, n),
        ])
        y = np.clip(spo2_gt, 88.0, 100.0)
        self._model = Pipeline([
            ("scaler", StandardScaler()),
            ("gbr", GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )),
        ])
        self._model.fit(X, y)
        self._model_ready = True

    def push_frame_direct(self, r_mean: float, g_mean: float, b_mean: float) -> None:
        idx = self._buf_head
        self._r_buf[idx], self._g_buf[idx], self._b_buf[idx] = r_mean, g_mean, b_mean
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

    def process(self, fps: Optional[float] = None) -> bool:
        if fps is not None:
            self.fps = fps
        if self._buf_count < RPPG_MIN_FRAMES:
            return False
        R, G, B = self._buffer_view()
        R_norm = R / (R.mean() + 1e-8)
        G_norm = G / (G.mean() + 1e-8)
        B_norm = B / (B.mean() + 1e-8)
        X_c = 3.0 * R_norm - 2.0 * G_norm
        Y_c = 1.5 * R_norm + G_norm - 1.5 * B_norm
        alpha = np.std(X_c) / (np.std(Y_c) + 1e-8)
        chrom_signal = X_c - alpha * Y_c
        C = np.column_stack([R_norm, G_norm, B_norm])
        Cn = C / (C.mean(axis=0) + 1e-8)
        H = np.array([[0, 1, -1], [-2, 1, 1]], dtype=np.float64)
        S_pos = (H @ Cn.T).T
        pos_signal = S_pos[:, 0] - (np.std(S_pos[:, 0]) / (np.std(S_pos[:, 1]) + 1e-8)) * S_pos[:, 1]
        raw_signal = 0.5 * chrom_signal + 0.5 * pos_signal
        detrended = detrend(raw_signal)
        std = detrended.std()
        if std < 1e-8:
            return False
        normalised = (detrended - detrended.mean()) / std
        nyq = self.fps / 2.0
        low = float(np.clip(RPPG_BPF_LOW / nyq, 0.001, 0.999))
        high = float(np.clip(RPPG_BPF_HIGH / nyq, 0.001, 0.999))
        if low >= high:
            return False
        key = (self.fps, low, high)
        if key not in self._bpf_cache:
            self._bpf_cache[key] = butter(4, [low, high], btype="band")
        b_coef, a_coef = self._bpf_cache[key]
        filtered = filtfilt(b_coef, a_coef, normalised)
        q1, q3 = np.percentile(filtered, [25, 75])
        filtered = np.clip(filtered, q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1))
        features, freq_hz = self._extract_features(filtered, R, G, B)
        if features is None:
            return False
        if not self._model_ready:
            self._init_ml_model()
        if self._model_ready:
            self.spo2 = float(np.clip(self._model.predict(np.array(features).reshape(1, -1))[0], 70.0, 100.0))
        else:
            ratio = (np.std(R) / (R.mean() + 1e-8)) / (np.std(G) / (G.mean() + 1e-8))
            self.spo2 = float(np.clip(104.0 - 17.0 * ratio, 70.0, 100.0))
        if self.spo2_smooth == 0.0:
            self.spo2_smooth = self.spo2
        else:
            self.spo2_smooth = SPO2_SMOOTH_ALPHA * self.spo2 + (1 - SPO2_SMOOTH_ALPHA) * self.spo2_smooth
        if freq_hz > 0:
            self.heart_rate = freq_hz * 60.0
        return True

    def _extract_features(
        self, filtered: np.ndarray, R: np.ndarray, G: np.ndarray, B: np.ndarray
    ) -> Tuple[Optional[List[float]], float]:
        n = len(filtered)
        if n < 32:
            return None, 0.0
        from scipy.stats import skew, kurtosis
        ac_r = float(R.std() / (np.mean(np.abs(R)) + 1e-8))
        ac_g = float(G.std() / (np.mean(np.abs(G)) + 1e-8))
        ac_b = float(B.std() / (np.mean(np.abs(B)) + 1e-8))
        ratio_rg, ratio_rb = ac_r / (ac_g + 1e-8), ac_r / (ac_b + 1e-8)
        nperseg = min(256, n // 2)
        freqs, psd = scipy_signal.welch(filtered, fs=self.fps, nperseg=nperseg)
        mask = (freqs >= RPPG_BPF_LOW) & (freqs <= RPPG_BPF_HIGH)
        if mask.sum() == 0:
            return None, 0.0
        psd_band, freqs_band = psd[mask], freqs[mask]
        dom_idx = int(np.argmax(psd_band))
        dom_freq = float(freqs_band[dom_idx])
        psd_norm = psd_band / (psd_band.sum() + 1e-8)
        spec_ent = float(-np.sum(psd_norm * np.log(psd_norm + 1e-8)))
        snr_proxy = float(psd_band[dom_idx] / (psd_band.mean() + 1e-8))
        rms_green = float(np.sqrt(np.mean(G ** 2)))
        skin_tone = float(np.clip(rms_green / 255.0 * 6.0, 1.0, 6.0))
        self.signal_quality = float(np.clip(snr_proxy / 20.0, 0.0, 1.0))
        return [ratio_rg, ratio_rb, dom_freq, spec_ent, snr_proxy, rms_green, skin_tone], dom_freq


def classify_posture(
    kp_xy: Optional[np.ndarray],
    kp_conf: Optional[np.ndarray],
    bbox: Tuple[int, int, int, int],
) -> str:
    x1, y1, x2, y2 = bbox
    bh, bw = max(y2 - y1, 1), max(x2 - x1, 1)
    KP_REQUIRED = (5, 6, 11, 12)
    kp_valid = (
        kp_xy is not None and kp_conf is not None
        and len(kp_conf) > max(KP_REQUIRED)
        and all(float(kp_conf[k]) >= KP_MIN_CONF for k in KP_REQUIRED)
    )
    if kp_valid:
        sh_x = (float(kp_xy[5][0]) + float(kp_xy[6][0])) / 2
        sh_y = (float(kp_xy[5][1]) + float(kp_xy[6][1])) / 2
        hp_x = (float(kp_xy[11][0]) + float(kp_xy[12][0])) / 2
        hp_y = (float(kp_xy[11][1]) + float(kp_xy[12][1])) / 2
        trunk_angle = math.degrees(math.atan2(abs(hp_x - sh_x), abs(hp_y - sh_y) + 1e-6))
        if trunk_angle > FALLEN_TRUNK_DEG:
            return Posture.FALLEN
        if trunk_angle < UPRIGHT_TRUNK_DEG:
            return Posture.STANDING if (bh / bw) > STANDING_BBOX_RATIO else Posture.SITTING
        return Posture.FALLEN if (bw / bh) > FALLEN_BBOX_RATIO else Posture.SITTING
    if (bw / bh) > FALLEN_BBOX_RATIO:
        return Posture.FALLEN
    return Posture.STANDING if (bh / bw) > STANDING_BBOX_RATIO else Posture.SITTING


@dataclass
class VisionResult:
    spo2: float = 0.0
    heart_rate: float = 0.0
    signal_quality: float = 0.0
    posture_status: str = Posture.UNKNOWN
    posture_confidence: float = 0.0
    fall_detected: bool = False
    immobile_seconds: float = 0.0
    faces_detected: int = 0


# Module-level singletons for streaming sessions (keyed by patient_id in production)
_processors: Dict[str, RppgProcessor] = {}
_fall_detectors: Dict[str, "FallDetector"] = {}


class FallDetector:
    """Single-frame YOLO pose inference for fall detection."""

    def __init__(self) -> None:
        self.available = YOLO_OK
        self._model = YOLO(YOLO_MODEL) if YOLO_OK else None
        self._fallen_since: Optional[float] = None
        self._last_move_time: float = time.time()

    def analyze(self, frame: np.ndarray) -> Tuple[str, float, bool, float]:
        if not self.available or self._model is None:
            return Posture.UNKNOWN, 0.0, False, 0.0
        results = self._model(frame, conf=0.45, verbose=False)
        now = time.time()
        best_posture = Posture.UNKNOWN
        confidence = 0.0
        for r in results:
            if r.keypoints is None or r.boxes is None:
                continue
            for i in range(len(r.boxes)):
                box = r.boxes.xyxy[i].cpu().numpy().astype(int)
                bbox = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                kp = r.keypoints.xy[i].cpu().numpy()
                kp_conf = r.keypoints.conf[i].cpu().numpy() if r.keypoints.conf is not None else None
                posture = classify_posture(kp, kp_conf, bbox)
                conf = float(r.boxes.conf[i]) if r.boxes.conf is not None else 0.5
                if conf > confidence:
                    confidence, best_posture = conf, posture
        fall_detected = best_posture == Posture.FALLEN
        if fall_detected:
            if self._fallen_since is None:
                self._fallen_since = now
            immobile = now - self._last_move_time if best_posture == Posture.FALLEN else 0.0
        else:
            self._fallen_since = None
            self._last_move_time = now
            immobile = 0.0
        if self._fallen_since and fall_detected:
            immobile = now - self._fallen_since
        return best_posture, confidence, fall_detected, immobile


class VisionAnalyzer:
    """High-level API: decode image → vitals + posture."""

    def __init__(self) -> None:
        self.face_extractor = FaceRoiExtractor()

    @staticmethod
    def decode_base64(image_base64: str) -> np.ndarray:
        raw = base64.b64decode(image_base64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image — ensure valid base64 JPEG/PNG")
        return frame

    def analyze_frame(self, frame: np.ndarray, patient_id: str = "default") -> VisionResult:
        result = VisionResult()
        if patient_id not in _processors:
            _processors[patient_id] = RppgProcessor()
        if patient_id not in _fall_detectors:
            _fall_detectors[patient_id] = FallDetector()

        processor = _processors[patient_id]
        fall_detector = _fall_detectors[patient_id]

        faces = self.face_extractor.extract(frame)
        result.faces_detected = len(faces)
        for face in faces:
            rgb = face.mean_rgb(frame)
            if rgb:
                processor.push_frame_direct(*rgb)

        if processor.process():
            result.spo2 = round(processor.spo2_smooth or processor.spo2, 1)
            result.heart_rate = round(processor.heart_rate, 1)
            result.signal_quality = round(processor.signal_quality, 2)

        posture, conf, fall, immobile = fall_detector.analyze(frame)
        result.posture_status = posture
        result.posture_confidence = round(conf, 2)
        result.fall_detected = fall
        result.immobile_seconds = round(immobile, 1)
        return result

    def analyze_base64(self, image_base64: str, patient_id: str = "default") -> VisionResult:
        return self.analyze_frame(self.decode_base64(image_base64), patient_id)
