"""
fusion_engine.py — Multimodal triage fusion for vision + audio signals.

Combines vitals, posture/fall data, and audio distress features into
Level A / B / C triage decisions with explainable reasons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class VitalsInput:
    spo2: float = 98.0
    heart_rate: float = 72.0
    signal_quality: float = 0.5


@dataclass
class PostureInput:
    status: str = "UNKNOWN"
    fall_detected: bool = False
    immobile_seconds: float = 0.0
    confidence: float = 0.0


@dataclass
class VisionInput:
    vitals: VitalsInput = field(default_factory=VitalsInput)
    posture: PostureInput = field(default_factory=PostureInput)


@dataclass
class AudioInput:
    breath_rate: float = 16.0
    cough_detected: bool = False
    distress_score: float = 0.0
    speech_clarity: float = 1.0


@dataclass
class PatientContext:
    name: str = "Unknown"
    age: int = 0
    conditions: List[str] = field(default_factory=list)


@dataclass
class TriageResult:
    triage_level: str
    urgency_score: float
    reasons: List[str]
    recommended_action: str
    decided_at: str


class FusionEngine:
    """
    Rule-based fusion with weighted urgency scoring.

    Level C triggers (any):
      - SpO₂ < 90
      - Fall + immobile ≥ 15 s
      - Distress ≥ 0.7
      - HR < 40 or > 140
      - Cough + SpO₂ < 92

    Level B triggers (any, if not C):
      - SpO₂ 90–93
      - Distress 0.4–0.7
      - Fall detected (recovered / short immobility)
      - Abnormal breath rate (< 10 or > 28)
    """

    SPO2_CRITICAL = 90.0
    SPO2_WARNING = 94.0
    HR_LOW = 40.0
    HR_HIGH = 140.0
    DISTRESS_CRITICAL = 0.7
    DISTRESS_WARNING = 0.4
    IMMOBILE_CRITICAL_SEC = 15.0

    def __init__(self) -> None:
        pass

    @staticmethod
    def _parse_vision(data: Optional[Dict[str, Any]]) -> VisionInput:
        if not data:
            return VisionInput()
        vitals_raw = data.get("vitals") or {}
        posture_raw = data.get("posture") or {}
        return VisionInput(
            vitals=VitalsInput(
                spo2=float(vitals_raw.get("spo2", 98.0)),
                heart_rate=float(vitals_raw.get("heart_rate", 72.0)),
                signal_quality=float(vitals_raw.get("signal_quality", 0.5)),
            ),
            posture=PostureInput(
                status=str(posture_raw.get("status", "UNKNOWN")),
                fall_detected=bool(posture_raw.get("fall_detected", False)),
                immobile_seconds=float(posture_raw.get("immobile_seconds", 0.0)),
                confidence=float(posture_raw.get("confidence", 0.0)),
            ),
        )

    @staticmethod
    def _parse_audio(data: Optional[Dict[str, Any]]) -> AudioInput:
        if not data:
            return AudioInput()
        return AudioInput(
            breath_rate=float(data.get("breath_rate", 16.0)),
            cough_detected=bool(data.get("cough_detected", False)),
            distress_score=float(data.get("distress_score", 0.0)),
            speech_clarity=float(data.get("speech_clarity", 1.0)),
        )

    @staticmethod
    def _parse_context(data: Optional[Dict[str, Any]]) -> PatientContext:
        if not data:
            return PatientContext()
        return PatientContext(
            name=str(data.get("name", "Unknown")),
            age=int(data.get("age", 0)),
            conditions=list(data.get("conditions") or []),
        )

    def decide(
        self,
        vision: Optional[Dict[str, Any]] = None,
        audio: Optional[Dict[str, Any]] = None,
        patient_context: Optional[Dict[str, Any]] = None,
    ) -> TriageResult:
        v = self._parse_vision(vision)
        a = self._parse_audio(audio)
        ctx = self._parse_context(patient_context)

        reasons: List[str] = []
        score = 0.0

        spo2 = v.vitals.spo2
        hr = v.vitals.heart_rate
        level_c_triggers = 0
        level_b_triggers = 0

        # ── SpO₂ ──────────────────────────────────────────────────────────────
        if spo2 < self.SPO2_CRITICAL:
            reasons.append(f"SpO₂ critically low ({spo2:.1f}%)")
            score += 0.35
            level_c_triggers += 1
        elif spo2 < self.SPO2_WARNING:
            reasons.append(f"SpO₂ below normal ({spo2:.1f}%)")
            score += 0.15
            level_b_triggers += 1

        # ── Heart rate ────────────────────────────────────────────────────────
        if hr < self.HR_LOW or hr > self.HR_HIGH:
            reasons.append(f"Abnormal heart rate ({hr:.0f} bpm)")
            score += 0.25
            level_c_triggers += 1
        elif hr < 50 or hr > 120:
            reasons.append(f"Heart rate outside optimal range ({hr:.0f} bpm)")
            score += 0.10
            level_b_triggers += 1

        # ── Fall / immobility ─────────────────────────────────────────────────
        if v.posture.fall_detected and v.posture.immobile_seconds >= self.IMMOBILE_CRITICAL_SEC:
            reasons.append(
                f"Fall detected with {v.posture.immobile_seconds:.0f}s immobility"
            )
            score += 0.30
            level_c_triggers += 1
        elif v.posture.fall_detected:
            reasons.append("Fall detected — monitoring recovery")
            score += 0.15
            level_b_triggers += 1

        # ── Audio distress ────────────────────────────────────────────────────
        if a.distress_score >= self.DISTRESS_CRITICAL:
            reasons.append(f"High distress score ({a.distress_score:.2f})")
            score += 0.25
            level_c_triggers += 1
        elif a.distress_score >= self.DISTRESS_WARNING:
            reasons.append(f"Elevated distress ({a.distress_score:.2f})")
            score += 0.12
            level_b_triggers += 1

        if a.cough_detected and spo2 < 92:
            reasons.append("Cough with low SpO₂ — possible respiratory distress")
            score += 0.20
            level_c_triggers += 1
        elif a.cough_detected:
            reasons.append("Cough detected")
            score += 0.05
            level_b_triggers += 1

        if a.breath_rate < 10 or a.breath_rate > 28:
            reasons.append(f"Abnormal breath rate ({a.breath_rate:.0f}/min)")
            score += 0.12
            level_b_triggers += 1

        # ── Age / comorbidity boost ───────────────────────────────────────────
        if ctx.age >= 80 and level_c_triggers > 0:
            score = min(1.0, score + 0.05)
        if any(c.lower() in ("copd", "heart failure", "chf") for c in ctx.conditions):
            if spo2 < 92:
                score = min(1.0, score + 0.08)

        # ── Assign level ──────────────────────────────────────────────────────
        if level_c_triggers > 0:
            triage_level = "C"
        elif level_b_triggers > 0:
            triage_level = "B"
        else:
            triage_level = "A"
            if not reasons:
                reasons.append("All vitals within normal range")

        urgency_score = round(min(1.0, max(0.0, score)), 2)
        action = self._recommended_action(triage_level, reasons, ctx)
        decided_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        return TriageResult(
            triage_level=triage_level,
            urgency_score=urgency_score,
            reasons=reasons,
            recommended_action=action,
            decided_at=decided_at,
        )

    @staticmethod
    def _recommended_action(level: str, reasons: List[str], ctx: PatientContext) -> str:
        name = ctx.name if ctx.name != "Unknown" else "Patient"
        if level == "C":
            if any("immobility" in r.lower() for r in reasons):
                return f"Immediate clinical review for {name} — possible post-fall emergency"
            if any("spo" in r.lower() for r in reasons):
                return f"Immediate oxygen assessment for {name} — critical SpO₂"
            return f"Immediate clinical review for {name} — critical vitals detected"
        if level == "B":
            return f"Increase monitoring frequency for {name} — moderate concern"
        return f"Continue routine monitoring for {name}"
