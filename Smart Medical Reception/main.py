"""
main.py â€” FastAPI backend for the Health Monitor hackathon stack.

Endpoints:
  GET  /health
  POST /api/vision
  POST /api/audio
  POST /api/triage

Run:  uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from facemesh import VisionAnalyzer
from fusion_engine import FusionEngine
from speech_transcriber import Transcriber

# â”€â”€ Optional Supabase persistence â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

_supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as exc:
        print(f"[WARN] Supabase client not available: {exc}")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# â”€â”€ Pydantic models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class VisionRequest(BaseModel):
    patient_id: str
    image_base64: str
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VitalsOut(BaseModel):
    spo2: float
    heart_rate: float
    signal_quality: float


class PostureOut(BaseModel):
    status: str
    confidence: float
    fall_detected: bool
    immobile_seconds: float


class VisionResponse(BaseModel):
    patient_id: str
    vitals: VitalsOut
    posture: PostureOut
    faces_detected: int
    processed_at: str


class AudioFeatures(BaseModel):
    breath_rate: float = 16.0
    cough_detected: bool = False
    distress_score: float = 0.0
    speech_clarity: float = 1.0


class AudioRequest(BaseModel):
    patient_id: str
    audio_features: AudioFeatures
    timestamp: Optional[str] = None


class AudioResponse(BaseModel):
    patient_id: str
    audio_features: AudioFeatures
    stored_at: str


class TranscriptSegmentOut(BaseModel):
    start: float
    end: float
    text: str
    language: str


class TranscriptionResponse(BaseModel):
    patient_id: Optional[str] = None
    language: Optional[str]
    transcript: str
    segments: List[TranscriptSegmentOut]
    processed_at: str


class TriageRequest(BaseModel):
    patient_id: str
    vision: Optional[Dict[str, Any]] = None
    audio: Optional[Dict[str, Any]] = None
    patient_context: Optional[Dict[str, Any]] = None


class TriageResponse(BaseModel):
    patient_id: str
    triage_level: str
    urgency_score: float
    reasons: List[str]
    recommended_action: str
    decided_at: str


# â”€â”€ App setup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
app = FastAPI(
    title="Health Monitor API",
    description="Multimodal vision + audio triage for hackathon demo",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vision_analyzer = VisionAnalyzer()
fusion_engine = FusionEngine()

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "auto")
WHISPER_LANGUAGES = {
    lang.strip().lower()
    for lang in os.getenv("WHISPER_LANGUAGES", "ja,en").split(",")
    if lang.strip()
}
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "1"))
_speech_transcriber: Optional[Transcriber] = None


def _get_speech_transcriber() -> Transcriber:
    global _speech_transcriber
    if _speech_transcriber is None:
        _speech_transcriber = Transcriber(
            model_name=WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
            language_whitelist=WHISPER_LANGUAGES,
            beam_size=WHISPER_BEAM_SIZE,
        )
    return _speech_transcriber


# â”€â”€ Supabase helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _store_vitals(patient_id: str, result, processed_at: str) -> None:
    if not _supabase:
        return
    _supabase.table("vital_readings").insert({
        "patient_id": patient_id,
        "spo2": result.spo2,
        "heart_rate": result.heart_rate,
        "signal_quality": result.signal_quality,
        "posture_status": result.posture_status,
        "fall_detected": result.fall_detected,
        "immobile_seconds": result.immobile_seconds,
        "faces_detected": result.faces_detected,
        "recorded_at": processed_at,
    }).execute()
    _supabase.table("patients").update({
        "last_seen_at": processed_at,
    }).eq("id", patient_id).execute()


def _store_audio(patient_id: str, features: AudioFeatures, stored_at: str) -> None:
    if not _supabase:
        return
    _supabase.table("audio_readings").insert({
        "patient_id": patient_id,
        "breath_rate": features.breath_rate,
        "cough_detected": features.cough_detected,
        "distress_score": features.distress_score,
        "speech_clarity": features.speech_clarity,
        "recorded_at": stored_at,
    }).execute()


def _store_triage(patient_id: str, triage) -> None:
    if not _supabase:
        return
    _supabase.table("triage_events").insert({
        "patient_id": patient_id,
        "triage_level": triage.triage_level,
        "urgency_score": triage.urgency_score,
        "reasons": triage.reasons,
        "recommended_action": triage.recommended_action,
        "decided_at": triage.decided_at,
    }).execute()
    _supabase.table("patients").update({
        "current_triage_level": triage.triage_level,
        "urgency_score": triage.urgency_score,
        "last_seen_at": triage.decided_at,
    }).eq("id", patient_id).execute()


def _fetch_latest_vision(patient_id: str) -> Optional[Dict[str, Any]]:
    if not _supabase:
        return None
    resp = (
        _supabase.table("vital_readings")
        .select("*")
        .eq("patient_id", patient_id)
        .order("recorded_at", desc=True)
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    if not rows:
        return None
    row = rows[0]
    return {
        "vitals": {
            "spo2": row.get("spo2", 98),
            "heart_rate": row.get("heart_rate", 72),
            "signal_quality": row.get("signal_quality", 0.5),
        },
        "posture": {
            "status": row.get("posture_status", "UNKNOWN"),
            "fall_detected": row.get("fall_detected", False),
            "immobile_seconds": row.get("immobile_seconds", 0),
            "confidence": 0.8,
        },
    }


def _fetch_latest_audio(patient_id: str) -> Optional[Dict[str, Any]]:
    if not _supabase:
        return None
    resp = (
        _supabase.table("audio_readings")
        .select("*")
        .eq("patient_id", patient_id)
        .order("recorded_at", desc=True)
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    if not rows:
        return None
    row = rows[0]
    return {
        "breath_rate": row.get("breath_rate", 16),
        "cough_detected": row.get("cough_detected", False),
        "distress_score": row.get("distress_score", 0),
        "speech_clarity": row.get("speech_clarity", 1),
    }


def _fetch_patient_context(patient_id: str) -> Optional[Dict[str, Any]]:
    if not _supabase:
        return None
    resp = _supabase.table("patients").select("*").eq("id", patient_id).limit(1).execute()
    rows = resp.data or []
    if not rows:
        return None
    row = rows[0]
    return {
        "name": row.get("name", "Unknown"),
        "age": row.get("age", 0),
        "conditions": row.get("conditions") or [],
    }


# â”€â”€ Routes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "version": "1.0.0",
        "supabase_connected": _supabase is not None,
    }


@app.post("/api/vision", response_model=VisionResponse)
def process_vision(req: VisionRequest):
    """Analyze a camera frame â€” FaceMesh r-PPG vitals + YOLO fall detection."""
    try:
        result = vision_analyzer.analyze_base64(req.image_base64, req.patient_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Vision processing failed: {exc}") from exc

    processed_at = _now_iso()
    _store_vitals(req.patient_id, result, processed_at)

    return VisionResponse(
        patient_id=req.patient_id,
        vitals=VitalsOut(
            spo2=result.spo2,
            heart_rate=result.heart_rate,
            signal_quality=result.signal_quality,
        ),
        posture=PostureOut(
            status=result.posture_status,
            confidence=result.posture_confidence,
            fall_detected=result.fall_detected,
            immobile_seconds=result.immobile_seconds,
        ),
        faces_detected=result.faces_detected,
        processed_at=processed_at,
    )


@app.post("/api/audio", response_model=AudioResponse)
def process_audio(req: AudioRequest):
    """Ingest pre-extracted audio features (breath rate, distress, cough)."""
    stored_at = _now_iso()
    _store_audio(req.patient_id, req.audio_features, stored_at)
    return AudioResponse(
        patient_id=req.patient_id,
        audio_features=req.audio_features,
        stored_at=stored_at,
    )


@app.post("/api/audio/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    patient_id: Optional[str] = Form(default=None),
):
    """Transcribe an uploaded audio file with local faster-whisper."""
    suffix = Path(audio_file.filename or "audio.wav").suffix or ".wav"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(await audio_file.read())

        transcriber = _get_speech_transcriber()
        segments, language = transcriber.transcribe_file(str(tmp_path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Audio transcription failed: {exc}") from exc
    finally:
        if tmp_path:
            tmp_path.unlink(missing_ok=True)

    processed_at = _now_iso()
    transcript = " ".join(segment.text for segment in segments).strip()
    return TranscriptionResponse(
        patient_id=patient_id,
        language=language,
        transcript=transcript,
        segments=[
            TranscriptSegmentOut(
                start=segment.start,
                end=segment.end,
                text=segment.text,
                language=segment.language,
            )
            for segment in segments
        ],
        processed_at=processed_at,
    )

@app.post("/api/triage", response_model=TriageResponse)
def process_triage(req: TriageRequest):
    """Fuse vision + audio signals and assign triage level A / B / C."""
    vision = req.vision or _fetch_latest_vision(req.patient_id)
    audio = req.audio or _fetch_latest_audio(req.patient_id)
    context = req.patient_context or _fetch_patient_context(req.patient_id)

    if not vision and not audio:
        raise HTTPException(
            status_code=400,
            detail="No vision or audio data provided and none found in database",
        )

    triage = fusion_engine.decide(
        vision=vision,
        audio=audio,
        patient_context=context,
    )
    _store_triage(req.patient_id, triage)

    return TriageResponse(
        patient_id=req.patient_id,
        triage_level=triage.triage_level,
        urgency_score=triage.urgency_score,
        reasons=triage.reasons,
        recommended_action=triage.recommended_action,
        decided_at=triage.decided_at,
    )

