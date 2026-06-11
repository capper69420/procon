"""
facemesh_pain_scorer.py
========================
Kosen Procon — Smart Medical Reception Kiosk
AI/ML Module: Visual Pain Assessment via MediaPipe FaceMesh

Architecture:
    MediaPipe FaceMesh (468 landmarks)
          ↓
    Geometric Feature Extraction  (6 features per frame)
          ↓
    Calibration Normalization     (personal neutral-face baseline, ~1.5s)
          ↓
    PSPI-inspired AU Weighting    (Action Unit proxy scores 0–1)
          ↓
    Rolling Window + Kalman Smoothing
          ↓
    Pain Score 0–10 + Level Label

Pain-Correlated Action Units (Facial Action Coding System):
    AU4   Brow Lowerer      — corrugator supercilii activation (furrowing)
    AU7   Lid Tightener     — orbicularis oculi palpebral (squinting)
    AU9   Nose Wrinkler     — levator labii superioris alaeque nasi
    AU20  Lip Stretcher     — risorius / platysma (horizontal grimace)
    AU25  Lips Part         — depressor labii (mouth opening)
    AU43  Eyes Closed       — full/partial eye closure

PSPI Formula (Prkachin & Solomon, 2008):
    pain = AU4 + max(AU6, AU7) + max(AU9, AU10) + AU43

This implementation adapts that formula to landmark geometry:
    pain_score = Σ w_i · au_i,   normalized → 0–10

Author:   AI/ML Engineer, Smart Reception Team
Version:  1.0.0

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\env12\Scripts\Activate.ps1
"""

from __future__ import annotations

import numpy as np # type: ignore
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Sequence


# ══════════════════════════════════════════════════════════════════════════════
# LANDMARK INDEX DEFINITIONS
# MediaPipe FaceMesh 468-point canonical model.
# Adjust indices here only if mediapipe version changes.
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/
#            face_geometry/data/canonical_face_model_uv_visualization.png
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class _FaceLandmarks:
    """
    Landmark indices for pain-relevant facial regions.
    All values are indices into `face_results.multi_face_landmarks[i].landmark`.
    """

    # ── Eyes ── (Eye Aspect Ratio: outer, upper×2, inner, lower×2)
    LEFT_EYE_EAR:  tuple = (33,  160, 158, 133, 153, 144)
    RIGHT_EYE_EAR: tuple = (362, 385, 387, 263, 373, 380)

    # ── Eyebrows ── (for AU4 brow-to-eye vertical gap)
    LEFT_INNER_BROW:  int = 107   # Nasal-side brow anchor (person's right eye side)
    RIGHT_INNER_BROW: int = 336   # Nasal-side brow anchor (person's left eye side)
    LEFT_EYE_INNER:   int = 133   # Inner corner of person's right eye
    RIGHT_EYE_INNER:  int = 362   # Inner corner of person's left eye

    # ── Nose ── (for AU9 nose-wing spread)
    LEFT_NOSTRIL:   int = 129     # Left alar landmark
    RIGHT_NOSTRIL:  int = 358     # Right alar landmark

    # ── Mouth ── (for AU20 horizontal stretch + AU25 vertical opening)
    MOUTH_LEFT:     int = 61      # Left mouth corner
    MOUTH_RIGHT:    int = 291     # Right mouth corner
    UPPER_LIP_IN:   int = 13      # Inner upper lip midpoint
    LOWER_LIP_IN:   int = 14      # Inner lower lip midpoint

    # ── Face dimension anchors ── (normalization)
    FOREHEAD:       int = 10
    CHIN:           int = 152


FL = _FaceLandmarks()   # Module-level singleton — import this in other files


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PainScoringConfig:
    """
    All numeric constants for the scorer in one place.
    Tune these during validation with real patient expressions.
    """

    # ── PSPI-inspired AU weights ──────────────────────────────────────────────
    # Higher weight = that AU contributes more to the final pain score.
    # Calibrated to approximate clinical PSPI scale (0–16 internally, mapped 0–10).
    w_au4_brow:        float = 2.5   # Brow lowering — strongest single pain indicator
    w_au43_eye:        float = 2.0   # Eye squinting / closure
    w_au9_nose:        float = 1.5   # Nose wrinkling
    w_au20_grimace:    float = 1.5   # Horizontal mouth stretch (grimace)
    w_au25_lip:        float = 1.0   # Lip parting
    w_bilateral_bonus: float = 0.5   # Bonus when brow lowering is bilateral (symmetrical)

    # ── AU activation sensitivity ─────────────────────────────────────────────
    # These define the fraction-of-face-height delta that corresponds to full
    # AU activation (score = 1.0). Smaller → more sensitive.
    brow_delta_max:  float = 0.06   # 6% of face height → max AU4 score
    ear_delta_max:   float = 0.12   # 12% relative EAR drop → max AU43 score
    nose_delta_max:  float = 0.04
    mouth_v_max:     float = 0.05   # Vertical lip separation
    grimace_ratio_max: float = 0.08  # Ratio shift for horizontal grimace

    # ── Calibration ───────────────────────────────────────────────────────────
    calibration_frames: int   = 45     # ~1.5s at 30fps (patient "looks at camera")
    calib_trim_pct:     float = 0.10   # Remove top+bottom 10% before computing mean

    # ── Temporal smoothing ────────────────────────────────────────────────────
    window_size:        int   = 15     # Rolling window (frames) before Kalman input
    kalman_Q:           float = 0.08   # Process noise (how fast signal can change)
    kalman_R:           float = 0.40   # Measurement noise (how noisy raw score is)


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES (OUTPUT)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AUScores:
    """Individual Action Unit proxy scores. Each value ∈ [0, 1]."""
    au4_brow_lowering:  float = 0.0
    au43_eye_closure:   float = 0.0
    au9_nose_wrinkle:   float = 0.0
    au20_grimace:       float = 0.0
    au25_lip_part:      float = 0.0
    bilateral_bonus:    float = 0.0

    def to_dict(self) -> dict:
        return {
            "AU4":  round(self.au4_brow_lowering, 3),
            "AU43": round(self.au43_eye_closure, 3),
            "AU9":  round(self.au9_nose_wrinkle, 3),
            "AU20": round(self.au20_grimace, 3),
            "AU25": round(self.au25_lip_part, 3),
            "bilateral_bonus": round(self.bilateral_bonus, 3),
        }


@dataclass
class PainScoreResult:
    """Complete output from a single FaceMeshPainScorer.score() call."""
    pain_score:           float    # Final smoothed score ∈ [0, 10]
    pain_score_raw:       float    # Instantaneous unsmoothed score ∈ [0, 10]
    pain_level:           str      # 'none'|'mild'|'moderate'|'severe'|'extreme'
    au_scores:            AUScores # Per-AU breakdown for dashboard display
    is_calibrated:        bool     # False during first ~45 frames (show UI progress)
    calibration_progress: float    # 0.0→1.0 (percentage complete)
    face_size_norm:       float    # Normalized face height (quality check; < 0.2 = too far)

    def to_api_dict(self) -> dict:
        """Serialize for FastAPI response / Supabase insert."""
        return {
            "pain_score":           round(self.pain_score, 2),
            "pain_score_raw":       round(self.pain_score_raw, 2),
            "pain_level":           self.pain_level,
            "face_action_units":    self.au_scores.to_dict(),
            "is_calibrated":        self.is_calibrated,
            "calibration_progress": round(self.calibration_progress, 2),
        }


# ══════════════════════════════════════════════════════════════════════════════
# KALMAN SMOOTHER
# ══════════════════════════════════════════════════════════════════════════════

class KalmanSmoother1D:
    """
    Minimal 1-D Kalman filter for scalar time-series smoothing.

    Purpose: Prevents momentary facial expression spikes (blink, cough) from
    triggering false Level-C escalations. The filter "believes" the signal
    changes smoothly over time (Q) and that individual measurements are noisy (R).

    Higher Q → faster response to real pain escalation.
    Higher R → smoother output but slower to react.
    """

    def __init__(self, Q: float = 0.08, R: float = 0.40):
        self.Q = Q        # Process noise variance
        self.R = R        # Measurement noise variance
        self._x = 0.0     # State estimate
        self._P = 1.0     # Estimate error covariance
        self._init = False

    def update(self, z: float) -> float:
        """Feed one measurement, get back the filtered estimate."""
        if not self._init:
            self._x = z
            self._init = True
            return z

        # ── Predict ────────────────────────────────────────────
        P_pred = self._P + self.Q

        # ── Update (Kalman gain) ────────────────────────────────
        K = P_pred / (P_pred + self.R)          # Kalman gain ∈ [0, 1]
        self._x = self._x + K * (z - self._x)  # Corrected estimate
        self._P = (1.0 - K) * P_pred            # Updated covariance

        return float(self._x)

    def reset(self):
        self._x = 0.0
        self._P = 1.0
        self._init = False


# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRY HELPERS (module-level, no state)
# ══════════════════════════════════════════════════════════════════════════════

def _dist(a, b) -> float:
    """Euclidean distance between two MediaPipe NormalizedLandmark objects."""
    return float(np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2))


def _eye_aspect_ratio(lm: Sequence, indices: tuple) -> float:
    """
    Eye Aspect Ratio (EAR) — Soukupová & Čech, 2016.

        EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2 · ‖p1−p4‖)

    Interpretation:
        Open eye  (neutral): EAR ≈ 0.25–0.32
        Squinting (pain):    EAR ≈ 0.15–0.22
        Closed:              EAR ≈ 0.05

    Args:
        lm:      Full landmark list (lm[i] = NormalizedLandmark)
        indices: (p1, p2, p3, p4, p5, p6) — see LEFT_EYE_EAR / RIGHT_EYE_EAR
    """
    p1, p2, p3, p4, p5, p6 = [lm[i] for i in indices]
    A = _dist(p2, p6)
    B = _dist(p3, p5)
    C = _dist(p1, p4)
    return (A + B) / (2.0 * C + 1e-8)


def _trimmed_mean(values: list[float], trim_pct: float = 0.10) -> float:
    """Mean after removing top and bottom trim_pct fraction. More robust than raw mean."""
    n = len(values)
    if n == 0:
        return 0.0
    k = max(1, int(n * trim_pct))
    trimmed = sorted(values)[k:-k] if k < n // 2 else values
    return float(np.mean(trimmed))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SCORER CLASS
# ══════════════════════════════════════════════════════════════════════════════

class FaceMeshPainScorer:
    """
    Converts MediaPipe FaceMesh landmark output to a calibrated 0–10 pain score.

    Lifecycle per patient:
        1. Patient steps in front of kiosk.
        2. Call reset() to clear the previous patient's calibration.
        3. Call score() once per frame in your video loop.
        4. During the first ~1.5 s (calibration_frames), is_calibrated = False
           and the score is meaningless — show a "Please look at the camera" UI.
        5. After calibration, scores are reliable and Kalman-smoothed.
        6. After the session, query get_session_stats() for the Supabase insert.

    Example (single-face kiosk):
        scorer = FaceMeshPainScorer()

        while cap.isOpened():
            ret, frame = cap.read()
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                continue
            result = scorer.score(results.multi_face_landmarks[0].landmark)
            print(f"Pain: {result.pain_score:.1f}/10  [{result.pain_level}]  "
                  f"AU4={result.au_scores.au4_brow_lowering:.2f}  "
                  f"Calib: {result.calibration_progress*100:.0f}%")
    """

    def __init__(self, config: Optional[PainScoringConfig] = None):
        self.cfg = config or PainScoringConfig()

        # ── Calibration buffers ────────────────────────────────
        self._cal: dict[str, list[float]] = {
            "ear_l": [], "ear_r": [], "brow": [],
            "nose":  [], "mouth_v": [], "mouth_h": [],
        }
        # Calibrated baselines (means from neutral-face frames)
        self._base: dict[str, Optional[float]] = {k: None for k in self._cal}
        self.is_calibrated: bool = False

        # ── Smoothing ──────────────────────────────────────────
        self._window:  deque = deque(maxlen=self.cfg.window_size)
        self._kalman:  KalmanSmoother1D = KalmanSmoother1D(
            Q=self.cfg.kalman_Q, R=self.cfg.kalman_R
        )

        # ── Session statistics (for Supabase insert at session end) ─
        self._frame_count:   int   = 0
        self._score_history: deque = deque(maxlen=300)  # ~10s at 30fps

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────────

    def score(self, landmarks) -> PainScoreResult:
        """
        Main entry point. Call once per face per frame.

        Args:
            landmarks: face_results.multi_face_landmarks[0].landmark
                       (NormalizedLandmarkList of 468 points)

        Returns:
            PainScoreResult — safe to read during/after calibration.
        """
        lm = landmarks
        self._frame_count += 1

        # ── 1. Face size (quality + normalization denominator) ──
        face_h = _dist(lm[FL.FOREHEAD], lm[FL.CHIN])

        # ── 2. Extract raw geometric features ──────────────────
        ear_l   = _eye_aspect_ratio(lm, FL.LEFT_EYE_EAR)
        ear_r   = _eye_aspect_ratio(lm, FL.RIGHT_EYE_EAR)
        brow_d  = self._feat_brow(lm, face_h)
        nose_d  = self._feat_nose(lm, face_h)
        mouth_v = self._feat_mouth_vertical(lm, face_h)
        mouth_h = self._feat_mouth_horizontal(lm, face_h)

        # ── 3. Feed calibration buffers ─────────────────────────
        self._calibrate(ear_l, ear_r, brow_d, nose_d, mouth_v, mouth_h)
        calib_progress = min(1.0, len(self._cal["ear_l"]) / self.cfg.calibration_frames)

        # ── 4. Compute AU proxy scores ──────────────────────────
        au = self._compute_aus(ear_l, ear_r, brow_d, nose_d, mouth_v, mouth_h)

        # ── 5. Weighted PSPI-inspired sum → normalize to 0–10 ──
        cfg = self.cfg
        total_weight = (
            cfg.w_au4_brow + cfg.w_au43_eye + cfg.w_au9_nose +
            cfg.w_au20_grimace + cfg.w_au25_lip + cfg.w_bilateral_bonus
        )
        weighted_sum = (
            au.au4_brow_lowering * cfg.w_au4_brow   +
            au.au43_eye_closure  * cfg.w_au43_eye    +
            au.au9_nose_wrinkle  * cfg.w_au9_nose    +
            au.au20_grimace      * cfg.w_au20_grimace +
            au.au25_lip_part     * cfg.w_au25_lip    +
            au.bilateral_bonus   * cfg.w_bilateral_bonus
        )
        raw_10 = float(np.clip(weighted_sum / total_weight * 10.0, 0.0, 10.0))

        if not self.is_calibrated:
            raw_10 = 0.0     # Don't emit scores until baseline is established

        # ── 6. Rolling-window average → Kalman smooth ───────────
        self._window.append(raw_10)
        windowed  = float(np.mean(self._window))
        smoothed  = float(np.clip(self._kalman.update(windowed), 0.0, 10.0))

        self._score_history.append(smoothed)

        return PainScoreResult(
            pain_score           = round(smoothed, 2),
            pain_score_raw       = round(raw_10,  2),
            pain_level           = self._level_label(smoothed),
            au_scores            = au,
            is_calibrated        = self.is_calibrated,
            calibration_progress = round(calib_progress, 3),
            face_size_norm       = round(face_h, 4),
        )

    def reset(self):
        """
        Reset for a new patient session.
        Call this every time a new patient steps up to the kiosk.
        """
        for lst in self._cal.values():
            lst.clear()
        for k in self._base:
            self._base[k] = None
        self.is_calibrated = False
        self._window.clear()
        self._kalman.reset()
        self._frame_count = 0
        self._score_history.clear()

    def get_session_peak(self) -> float:
        """Peak pain score during the session. Insert into Supabase at session end."""
        return float(max(self._score_history)) if self._score_history else 0.0

    def get_session_mean(self) -> float:
        """Mean pain score across the session (post-calibration)."""
        return float(np.mean(list(self._score_history))) if self._score_history else 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # FEATURE EXTRACTION  (private)
    # ──────────────────────────────────────────────────────────────────────────

    def _feat_brow(self, lm, face_h: float) -> float:
        """
        AU4 geometry: average vertical distance from inner eyebrow to inner eye corner,
        normalized by face height.

        Coordinate note: MediaPipe y=0 is TOP of image, y=1 is BOTTOM.
        Eyebrow y < Eye y in normal state (brow above eye).
        When brow lowers toward eye (AU4), (eye.y - brow.y) DECREASES.
        """
        left_gap  = (lm[FL.LEFT_EYE_INNER].y  - lm[FL.LEFT_INNER_BROW].y)  / (face_h + 1e-8)
        right_gap = (lm[FL.RIGHT_EYE_INNER].y - lm[FL.RIGHT_INNER_BROW].y) / (face_h + 1e-8)
        return float((left_gap + right_gap) / 2.0)

    def _feat_brow_bilateral(self, lm, face_h: float) -> tuple[float, float]:
        """Returns (left_gap, right_gap) separately for bilateral symmetry check."""
        left  = (lm[FL.LEFT_EYE_INNER].y  - lm[FL.LEFT_INNER_BROW].y)  / (face_h + 1e-8)
        right = (lm[FL.RIGHT_EYE_INNER].y - lm[FL.RIGHT_INNER_BROW].y) / (face_h + 1e-8)
        return float(left), float(right)

    def _feat_nose(self, lm, face_h: float) -> float:
        """
        AU9 geometry: horizontal spread of nose wings, normalized by face height.
        Nasal wrinkling causes slight lateral displacement of alar landmarks.
        """
        return float(_dist(lm[FL.LEFT_NOSTRIL], lm[FL.RIGHT_NOSTRIL]) / (face_h + 1e-8))

    def _feat_mouth_vertical(self, lm, face_h: float) -> float:
        """AU25 geometry: inner-lip vertical gap normalized by face height."""
        return float(_dist(lm[FL.UPPER_LIP_IN], lm[FL.LOWER_LIP_IN]) / (face_h + 1e-8))

    def _feat_mouth_horizontal(self, lm, face_h: float) -> float:
        """AU20 geometry: mouth-corner horizontal distance normalized by face height."""
        return float(_dist(lm[FL.MOUTH_LEFT], lm[FL.MOUTH_RIGHT]) / (face_h + 1e-8))

    # ──────────────────────────────────────────────────────────────────────────
    # CALIBRATION  (private)
    # ──────────────────────────────────────────────────────────────────────────

    def _calibrate(self, ear_l, ear_r, brow, nose, mouth_v, mouth_h):
        """
        Accumulate neutral-face frames to build personal baseline.
        Using the patient's OWN neutral face removes inter-person geometry bias.
        (A naturally narrow-eyed person should not score higher than a wide-eyed one.)
        """
        if self.is_calibrated:
            return

        self._cal["ear_l"].append(ear_l)
        self._cal["ear_r"].append(ear_r)
        self._cal["brow"].append(brow)
        self._cal["nose"].append(nose)
        self._cal["mouth_v"].append(mouth_v)
        self._cal["mouth_h"].append(mouth_h)

        if len(self._cal["brow"]) >= self.cfg.calibration_frames:
            t = self.cfg.calib_trim_pct
            for k, vals in self._cal.items():
                self._base[k] = _trimmed_mean(vals, t)
            self.is_calibrated = True

    # ──────────────────────────────────────────────────────────────────────────
    # AU SCORE COMPUTATION  (private)
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_aus(
        self,
        ear_l: float, ear_r: float,
        brow:  float, nose: float,
        mouth_v: float, mouth_h: float,
    ) -> AUScores:
        """
        Convert geometric features to Action Unit scores ∈ [0, 1].
        Returns zero-valued AUScores if not yet calibrated.
        """
        if not self.is_calibrated:
            return AUScores()

        cfg = self.cfg
        b   = self._base     # Shorthand for baselines

        # ── AU43 / AU7 — Eye squinting ────────────────────────────────────────
        # EAR drops from baseline when patient squints in pain.
        ear_drop_l = (b["ear_l"] - ear_l) / (cfg.ear_delta_max + 1e-8)
        ear_drop_r = (b["ear_r"] - ear_r) / (cfg.ear_delta_max + 1e-8)
        au43 = float(np.clip((ear_drop_l + ear_drop_r) / 2.0, 0.0, 1.0))

        # ── AU4 — Brow lowering ───────────────────────────────────────────────
        # Brow-eye gap decreases when brows furrow toward midline.
        brow_drop = (b["brow"] - brow) / (cfg.brow_delta_max + 1e-8)
        au4 = float(np.clip(brow_drop, 0.0, 1.0))

        # Bilateral bonus: genuine pain expression is symmetrical.
        # Unilateral brow lowering is more likely skepticism/thinking.
        # Award a bonus only when au4 is already significant (> 0.35).
        bilateral_bonus = float(np.clip(au4 - 0.35, 0.0, 0.65)) if au4 > 0.35 else 0.0

        # ── AU9 — Nose wrinkling ──────────────────────────────────────────────
        # Nostril wings spread slightly when levator labii activates.
        nose_spread = (nose - b["nose"]) / (cfg.nose_delta_max + 1e-8)
        au9 = float(np.clip(nose_spread, 0.0, 1.0))

        # ── AU25 — Lip parting ────────────────────────────────────────────────
        # Vertical lip gap increases when lips part in pain/distress.
        lip_open = (mouth_v - b["mouth_v"]) / (cfg.mouth_v_max + 1e-8)
        au25 = float(np.clip(lip_open, 0.0, 1.0))

        # ── AU20 — Grimace (horizontal lip stretch) ───────────────────────────
        # Pain grimace widens the mouth horizontally without opening vertically.
        # Metric: the mouth_v / mouth_h ratio drops vs neutral (wider, not taller).
        ratio_base    = b["mouth_v"] / (b["mouth_h"] + 1e-8)
        ratio_current = mouth_v      / (mouth_h      + 1e-8)
        grimace_shift = (ratio_base - ratio_current) / (cfg.grimace_ratio_max + 1e-8)
        au20 = float(np.clip(grimace_shift, 0.0, 1.0))

        return AUScores(
            au4_brow_lowering = round(au4,       3),
            au43_eye_closure  = round(au43,      3),
            au9_nose_wrinkle  = round(au9,       3),
            au20_grimace      = round(au20,      3),
            au25_lip_part     = round(au25,      3),
            bilateral_bonus   = round(bilateral_bonus, 3),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # HELPERS  (private)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _level_label(score: float) -> str:
        """Map 0–10 score to clinical pain level label."""
        if score < 2.0:  return "none"
        if score < 4.0:  return "mild"
        if score < 6.0:  return "moderate"
        if score < 8.0:  return "severe"
        return "extreme"


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI INTEGRATION HELPER
# ══════════════════════════════════════════════════════════════════════════════

class MultiPatientScorerPool:
    """
    Thread-safe pool of FaceMeshPainScorer instances keyed by session_id.

    Use this in your FastAPI endpoint when processing video frames:

        pool = MultiPatientScorerPool()

        @app.post("/triage/vision")
        async def vision_endpoint(session_id: str, frame: ...) -> dict:
            scorer = pool.get_or_create(session_id)
            result = scorer.score(landmarks)
            return result.to_api_dict()

        @app.delete("/triage/session/{session_id}")
        async def end_session(session_id: str):
            pool.remove(session_id)
    """

    def __init__(self, config: Optional[PainScoringConfig] = None):
        self._cfg = config or PainScoringConfig()
        self._scorers: dict[str, FaceMeshPainScorer] = {}

    def get_or_create(self, session_id: str) -> FaceMeshPainScorer:
        if session_id not in self._scorers:
            self._scorers[session_id] = FaceMeshPainScorer(self._cfg)
        return self._scorers[session_id]

    def remove(self, session_id: str):
        self._scorers.pop(session_id, None)

    def __len__(self):
        return len(self._scorers)


# ══════════════════════════════════════════════════════════════════════════════
# QUICK DEMO  (run this file directly to verify the scorer)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("FaceMeshPainScorer — synthetic smoke test")
    print("=" * 50)

    import mediapipe as mp # type: ignore
    import cv2 # type: ignore


    mp_face_mesh = mp.solutions.face_mesh.FaceMesh
    face_mesh    = mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5,
    )

    scorer = FaceMeshPainScorer()
    cap    = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No camera found — run from a machine with a webcam.")
    else:
        print("Press Q to quit. Calibrating for first ~1.5s (look neutral)...")
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                lm  = results.multi_face_landmarks[0].landmark
                res = scorer.score(lm)

                status = (
                    f"Calibrating {res.calibration_progress*100:.0f}%"
                    if not res.is_calibrated
                    else f"Pain: {res.pain_score:4.1f}/10  [{res.pain_level}]  "
                         f"AU4={res.au_scores.au4_brow_lowering:.2f}  "
                         f"AU43={res.au_scores.au43_eye_closure:.2f}  "
                         f"AU9={res.au_scores.au9_nose_wrinkle:.2f}"
                )
                print(f"\r{status}", end="", flush=True)
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 200, 0) if res.pain_score < 5 else (0, 0, 255), 2)

            cv2.imshow("Pain Scorer Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print(f"\nSession peak: {scorer.get_session_peak():.1f}  "
              f"mean: {scorer.get_session_mean():.1f}")
        cap.release()
        cv2.destroyAllWindows()
