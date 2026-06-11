"""
triage/fusion_engine.py
=======================
Weighted multi-modal triage classification: Level A / B / C

Inputs  → VisionAnalysis  (FaceMesh pain score + YOLO flags)
         AudioAnalysis   (Whisper transcript + ChatGPT symptom extraction)
Output  → FusionResult   (level, confidence, rationale, primary triggers)

Scoring overview
────────────────
  combined_score = (pain_component + condition_component + symptom_component)
                   clamped to [0, 10]

  Level C  ≥ 7.5   (or any instant-escalation flag)
  Level B  ≥ 4.5
  Level A  < 4.5

Weights are tunable via TriageConfig; defaults reflect ED practice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TriageConfig:
    """All thresholds and weights in one place. Tune without touching logic."""

    # Component weights (must sum ≤ 10 per component so combined ≤ 10 raw)
    pain_weight: float       = 0.35   # FaceMesh AU pain score contributes 35%
    audio_pain_weight: float = 0.25   # Patient self-report contributes 25%
    condition_weight: float  = 0.25   # YOLO detections contribute 25%
    symptom_weight: float    = 0.15   # Keyword severity contributes 15%

    # Pain score normalisation (raw scores from FaceMesh / patient are 0–10)
    pain_scale: float = 10.0

    # YOLO condition scores (per detection, additive, uncapped before weight)
    condition_scores: dict[str, float] = field(default_factory=lambda: {
        "bleeding":        10.0,   # Visible blood → always escalate
        "unconscious":     10.0,   # No response   → always escalate
        "wheelchair":       6.0,   # Mobility issue → significant
        "crutches":         4.0,
        "oxygen_mask":      8.0,
        "stretcher":        8.0,
        "pale_skin":        5.0,
        "sweating":         4.5,
        "distress_posture": 5.5,
    })

    # Keyword severity buckets
    critical_keywords: list[str] = field(default_factory=lambda: [
        "chest pain", "cannot breathe", "difficulty breathing", "shortness of breath",
        "stroke", "seizure", "unconscious", "not responding", "heart attack",
        "severe bleeding", "anaphylaxis", "allergic reaction", "overdose",
    ])
    urgent_keywords: list[str] = field(default_factory=lambda: [
        "vomiting blood", "high fever", "severe headache", "confusion",
        "cannot walk", "severe pain", "dizziness", "fainting", "abdominal pain",
        "back pain", "weakness", "numbness",
    ])
    routine_keywords: list[str] = field(default_factory=lambda: [
        "mild pain", "cough", "cold", "sore throat", "minor cut",
        "prescription refill", "follow-up", "check-up",
    ])

    # Keyword score contribution per match
    keyword_score_critical: float = 10.0
    keyword_score_urgent: float   =  6.0
    keyword_score_routine: float  =  1.0

    # Decision thresholds
    level_c_threshold: float = 7.5
    level_b_threshold: float = 4.5

    # Instant-escalation flags (regardless of score)
    instant_c_conditions: list[str] = field(default_factory=lambda: [
        "bleeding", "unconscious",
    ])
    instant_c_keywords: list[str] = field(default_factory=lambda: [
        "chest pain", "cannot breathe", "difficulty breathing",
        "shortness of breath", "stroke", "seizure",
        "heart attack", "severe bleeding", "anaphylaxis", "overdose",
    ])

    # Self-report pain threshold that alone triggers Level C
    self_report_instant_c: float = 9.0


# ─────────────────────────────────────────────────────────────────────────────
# Input / Output models (mirrors your Pydantic models from models/)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VisionInput:
    pain_score: float                           # FaceMesh AU score 0–10
    detected_conditions: dict[str, bool]        # {"bleeding": False, ...}
    confidence: float = 1.0                     # Vision pipeline confidence

@dataclass
class AudioInput:
    self_reported_pain: Optional[float]         # Patient-stated 0–10 or None
    reported_symptoms: list[str]                # ["chest_pain", "dyspnea"]
    chief_complaint: str                        # "Chest pain with onset 2h ago"
    severity_keywords: list[str] = field(default_factory=list)  # Pre-extracted
    confidence: float = 1.0

@dataclass
class FusionResult:
    triage_level: str                           # "A", "B", or "C"
    combined_score: float                       # 0–10 final score
    confidence: float                           # 0–1
    decision_rationale: str                     # Human-readable explanation
    primary_triggers: list[str]                 # Top reasons for the decision
    component_scores: dict[str, float]          # {"pain": 3.2, "conditions": ...}
    instant_escalation: bool = False            # Bypassed scoring


# ─────────────────────────────────────────────────────────────────────────────
# Fusion Engine
# ─────────────────────────────────────────────────────────────────────────────

class TriageFusionEngine:
    """
    Stateless engine: one call to classify() per session.
    Inject a custom TriageConfig to adjust weights for your clinical context.
    """

    def __init__(self, config: Optional[TriageConfig] = None):
        self.config = config or TriageConfig()

    # ── Public entry point ────────────────────────────────────────────────────

    def classify(
        self,
        vision: Optional[VisionInput],
        audio: Optional[AudioInput],
    ) -> FusionResult:
        """
        Fuse vision + audio signals into a triage level.

        Either input may be None (e.g. camera failure, mic refusal) —
        the engine degrades gracefully and flags reduced confidence.
        """
        triggers: list[str] = []
        rationale_parts: list[str] = []

        # ── 1. Instant escalation checks (fast path) ──────────────────────────
        escalation = self._check_instant_escalation(vision, audio)
        if escalation:
            trigger_list, reason = escalation
            return FusionResult(
                triage_level="C",
                combined_score=10.0,
                confidence=0.99,
                decision_rationale=f"⚡ INSTANT ESCALATION — {reason}",
                primary_triggers=trigger_list,
                component_scores={},
                instant_escalation=True,
            )

        # ── 2. Component scoring ───────────────────────────────────────────────
        pain_score, pain_reason         = self._score_pain(vision, audio)
        condition_score, cond_reason    = self._score_conditions(vision)
        symptom_score, symptom_reason   = self._score_symptoms(audio)

        component_scores = {
            "pain":       round(pain_score,      2),
            "conditions": round(condition_score, 2),
            "symptoms":   round(symptom_score,   2),
        }

        # ── 3. Weighted combination ────────────────────────────────────────────
        cfg = self.config
        raw = (
            pain_score      * cfg.pain_weight
            + condition_score * cfg.condition_weight
            + symptom_score   * cfg.symptom_weight
        ) * 10   # normalise weights → 0–10 output range

        # Clamp to [0, 10]
        combined_score = round(max(0.0, min(10.0, raw)), 2)

        # ── 4. Level assignment ───────────────────────────────────────────────
        if combined_score >= cfg.level_c_threshold:
            level = "C"
        elif combined_score >= cfg.level_b_threshold:
            level = "B"
        else:
            level = "A"

        # ── 5. Confidence ─────────────────────────────────────────────────────
        vision_conf = vision.confidence if vision else 0.5
        audio_conf  = audio.confidence  if audio  else 0.5
        avg_conf    = (vision_conf + audio_conf) / 2

        # Penalise missing modalities
        if vision is None:
            avg_conf *= 0.75
        if audio is None:
            avg_conf *= 0.75

        # ── 6. Rationale string ───────────────────────────────────────────────
        parts = [p for p in [pain_reason, cond_reason, symptom_reason] if p]
        parts.append(f"Combined score: {combined_score}/10 → LEVEL {level}")
        rationale = " | ".join(parts)

        # ── 7. Primary triggers ───────────────────────────────────────────────
        primary_triggers = self._extract_primary_triggers(
            vision, audio, component_scores, level
        )

        logger.info(
            "Triage fusion complete",
            extra={
                "level": level,
                "score": combined_score,
                "components": component_scores,
            },
        )

        return FusionResult(
            triage_level=level,
            combined_score=combined_score,
            confidence=round(avg_conf, 2),
            decision_rationale=rationale,
            primary_triggers=primary_triggers,
            component_scores=component_scores,
        )

    # ── Instant escalation ────────────────────────────────────────────────────

    def _check_instant_escalation(
        self,
        vision: Optional[VisionInput],
        audio: Optional[AudioInput],
    ) -> Optional[tuple[list[str], str]]:
        """
        Returns (triggers, reason) if instant escalation applies, else None.
        These conditions skip weighted scoring entirely.
        """
        cfg = self.config
        triggers = []

        # Vision-based instant escalation
        if vision:
            for cond in cfg.instant_c_conditions:
                if vision.detected_conditions.get(cond):
                    triggers.append(f"{cond}_detected")

        # Keyword-based instant escalation
        if audio:
            complaint_lower = audio.chief_complaint.lower()
            all_text = complaint_lower + " " + " ".join(audio.reported_symptoms).lower()
            for kw in cfg.instant_c_keywords:
                if kw in all_text:
                    triggers.append(f"keyword:{kw}")

        # Self-report pain ≥ threshold
        if audio and audio.self_reported_pain is not None:
            if audio.self_reported_pain >= cfg.self_report_instant_c:
                triggers.append(
                    f"self_reported_pain_{audio.self_reported_pain}/10"
                )

        if triggers:
            reason = (
                f"Critical indicators detected: {', '.join(triggers[:3])}"
            )
            return triggers, reason

        return None

    # ── Pain scoring ──────────────────────────────────────────────────────────

    def _score_pain(
        self,
        vision: Optional[VisionInput],
        audio: Optional[AudioInput],
    ) -> tuple[float, str]:
        """
        Returns (pain_raw_0_to_1, rationale_string).
        Fuses FaceMesh pain score with patient self-report.
        """
        cfg = self.config
        parts = []
        scores = []
        weights = []

        if vision:
            norm = vision.pain_score / cfg.pain_scale
            scores.append(norm)
            weights.append(cfg.pain_weight)
            parts.append(f"FaceMesh pain: {vision.pain_score:.1f}/10")

        if audio and audio.self_reported_pain is not None:
            norm = audio.self_reported_pain / cfg.pain_scale
            scores.append(norm)
            weights.append(cfg.audio_pain_weight)
            parts.append(f"Self-reported pain: {audio.self_reported_pain}/10")

        if not scores:
            return 0.0, "Pain: no data"

        total_weight = sum(weights)
        fused = sum(s * w for s, w in zip(scores, weights)) / total_weight
        return round(fused, 3), " | ".join(parts)

    # ── Condition scoring ─────────────────────────────────────────────────────

    def _score_conditions(
        self,
        vision: Optional[VisionInput],
    ) -> tuple[float, str]:
        """
        Returns (condition_raw_0_to_1, rationale_string).
        Maps detected YOLO flags to additive scores, then normalises.
        """
        if not vision:
            return 0.0, "Conditions: no vision data"

        cfg = self.config
        total = 0.0
        detected = []

        for condition, flagged in vision.detected_conditions.items():
            if flagged and condition in cfg.condition_scores:
                score = cfg.condition_scores[condition]
                total += score
                detected.append(f"{condition}({score})")

        # Normalise: max possible condition score is 10 (worst single condition)
        # We cap at 10 before normalising so multiple conditions don't wrap
        normalised = min(total, 10.0) / 10.0

        if not detected:
            return 0.0, "Conditions: none detected"

        detail = ", ".join(detected)
        return round(normalised, 3), f"YOLO detections: {detail}"

    # ── Symptom keyword scoring ───────────────────────────────────────────────

    def _score_symptoms(
        self,
        audio: Optional[AudioInput],
    ) -> tuple[float, str]:
        """
        Returns (symptom_raw_0_to_1, rationale_string).
        Matches chief complaint + symptom list against severity keyword buckets.
        """
        if not audio:
            return 0.0, "Symptoms: no audio data"

        cfg = self.config
        all_text = (
            audio.chief_complaint.lower()
            + " "
            + " ".join(audio.reported_symptoms).lower()
            + " "
            + " ".join(audio.severity_keywords).lower()
        )

        max_score = 0.0
        matched_keywords: list[str] = []

        for kw in cfg.critical_keywords:
            if kw in all_text:
                max_score = max(max_score, cfg.keyword_score_critical)
                matched_keywords.append(f"critical:'{kw}'")

        for kw in cfg.urgent_keywords:
            if kw in all_text:
                max_score = max(max_score, cfg.keyword_score_urgent)
                matched_keywords.append(f"urgent:'{kw}'")

        for kw in cfg.routine_keywords:
            if kw in all_text:
                max_score = max(max_score, cfg.keyword_score_routine)
                matched_keywords.append(f"routine:'{kw}'")

        normalised = max_score / 10.0

        if not matched_keywords:
            return 0.0, "Symptoms: no severity keywords matched"

        top = ", ".join(matched_keywords[:3])
        return round(normalised, 3), f"Severity keywords: [{top}]"

    # ── Primary triggers ──────────────────────────────────────────────────────

    def _extract_primary_triggers(
        self,
        vision: Optional[VisionInput],
        audio: Optional[AudioInput],
        component_scores: dict[str, float],
        level: str,
    ) -> list[str]:
        """Return up to 5 human-readable triggers that drove the decision."""
        triggers: list[str] = []

        if vision:
            for cond, flagged in vision.detected_conditions.items():
                if flagged:
                    triggers.append(f"{cond}_detected")
            if vision.pain_score >= 7.0:
                triggers.append(f"high_vision_pain_score_{vision.pain_score:.1f}")

        if audio:
            if audio.self_reported_pain and audio.self_reported_pain >= 7.0:
                triggers.append(f"self_reported_pain_{audio.self_reported_pain}")
            if audio.reported_symptoms:
                triggers.extend(audio.reported_symptoms[:2])

        # Add the dominant component
        if component_scores:
            top = max(component_scores, key=component_scores.get)
            triggers.append(f"primary_signal:{top}({component_scores[top]:.2f})")

        return triggers[:5]


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI integration helper
# ─────────────────────────────────────────────────────────────────────────────

def build_fusion_inputs_from_db(
    vision_row: dict,
    audio_row: dict,
) -> tuple[Optional[VisionInput], Optional[AudioInput]]:
    """
    Convert raw DB rows (from Supabase) into typed fusion inputs.
    Call this in your POST /triage/decision handler before classify().
    """
    vision = None
    if vision_row:
        vision = VisionInput(
            pain_score=float(vision_row.get("pain_score", 0)),
            detected_conditions=vision_row.get("detected_conditions", {}),
            confidence=float(vision_row.get("confidence", 1.0)),
        )

    audio = None
    if audio_row:
        audio = AudioInput(
            self_reported_pain=audio_row.get("self_reported_pain"),
            reported_symptoms=audio_row.get("reported_symptoms", []),
            chief_complaint=audio_row.get("chief_complaint", ""),
            severity_keywords=audio_row.get("severity_keywords", []),
            confidence=float(audio_row.get("confidence", 1.0)),
        )

    return vision, audio


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    engine = TriageFusionEngine()

    # Test case 1: Level C — chest pain patient in wheelchair
    vision_c = VisionInput(
        pain_score=7.4,
        detected_conditions={"wheelchair": True, "bleeding": False, "unconscious": False},
    )
    audio_c = AudioInput(
        self_reported_pain=8.0,
        reported_symptoms=["chest_pain", "dyspnea"],
        chief_complaint="Chest pain with onset 2 hours ago, cannot breathe well",
    )
    result_c = engine.classify(vision_c, audio_c)
    print(f"Test C: Level={result_c.triage_level} Score={result_c.combined_score}")
    print(f"  Rationale: {result_c.decision_rationale}\n")

    # Test case 2: Level B — fall, moderate pain
    vision_b = VisionInput(
        pain_score=5.2,
        detected_conditions={"crutches": True},
    )
    audio_b = AudioInput(
        self_reported_pain=5.0,
        reported_symptoms=["knee_pain", "swelling"],
        chief_complaint="Fell down stairs, knee pain and swelling",
    )
    result_b = engine.classify(vision_b, audio_b)
    print(f"Test B: Level={result_b.triage_level} Score={result_b.combined_score}")
    print(f"  Rationale: {result_b.decision_rationale}\n")

    # Test case 3: Level A — routine check-up
    vision_a = VisionInput(
        pain_score=1.5,
        detected_conditions={},
    )
    audio_a = AudioInput(
        self_reported_pain=2.0,
        reported_symptoms=["mild_cough"],
        chief_complaint="Routine follow-up, mild cough for 3 days",
    )
    result_a = engine.classify(vision_a, audio_a)
    print(f"Test A: Level={result_a.triage_level} Score={result_a.combined_score}")
    print(f"  Rationale: {result_a.decision_rationale}\n")
