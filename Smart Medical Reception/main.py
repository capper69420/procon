"""Unified FastAPI backend for the MedAI kiosk.

The API keeps the original hackathon endpoints alive while adding a session
contract for the polished React kiosk:

- patient registration/listing
- measurement start/frame/stop
- heart-rate, SpO2, and face-status retrieval
- English/Japanese speech transcription
- assessment, room assignment, and results retrieval

Run:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from facemesh import VisionAnalyzer
from fusion_engine import FusionEngine
from speech_transcriber import Transcriber


BASE_DIR = Path(__file__).resolve().parent
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(25 * 1024 * 1024)))


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file(BASE_DIR / ".env")
_load_env_file(BASE_DIR / "frontend" / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL") or os.getenv("VITE_SUPABASE_URL", "")
SUPABASE_KEY = (
    os.getenv("SUPABASE_KEY")
    or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    or os.getenv("VITE_SUPABASE_ANON_KEY", "")
)

_supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client

        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as exc:
        print(f"[WARN] Supabase client not available: {exc}")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_id() -> str:
    return str(uuid.uuid4())


def _db_warn(label: str, exc: Exception) -> None:
    print(f"[WARN] Supabase {label} failed: {exc}")


def _db_insert(table: str, payload: Dict[str, Any], label: str) -> None:
    if not _supabase:
        return
    try:
        _supabase.table(table).insert(payload).execute()
    except Exception as exc:
        _db_warn(label, exc)


def _db_update(table: str, payload: Dict[str, Any], key: str, value: str, label: str) -> None:
    if not _supabase:
        return
    try:
        _supabase.table(table).update(payload).eq(key, value).execute()
    except Exception as exc:
        _db_warn(label, exc)


class PatientCreate(BaseModel):
    id: Optional[str] = None
    name: str = "Kiosk Patient"
    name_ja: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    sex: Optional[str] = None
    blood_type: Optional[str] = None
    room: Optional[str] = None
    symptoms: Optional[str] = None
    conditions: List[str] = Field(default_factory=list)
    emergency_contact: Optional[str] = None


class PatientOut(PatientCreate):
    id: str
    current_triage_level: str = "A"
    urgency_score: float = 0.0
    created_at: str
    updated_at: str
    latest_vitals: Optional[Dict[str, Any]] = None
    latest_audio: Optional[Dict[str, Any]] = None
    latest_triage: Optional[Dict[str, Any]] = None


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


class MeasurementStartRequest(BaseModel):
    patient_id: Optional[str] = None
    patient: Optional[PatientCreate] = None
    language: str = "ja"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MeasurementStartResponse(BaseModel):
    session_id: str
    patient: PatientOut
    status: str
    started_at: str


class MeasurementFrameRequest(BaseModel):
    image_base64: str
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MeasurementStopResponse(BaseModel):
    session_id: str
    patient_id: str
    status: str
    stopped_at: str
    latest_vitals: Optional[Dict[str, Any]] = None
    latest_triage: Optional[Dict[str, Any]] = None


class MetricResponse(BaseModel):
    patient_id: str
    session_id: Optional[str] = None
    metric: str
    value: Optional[float]
    unit: str
    signal_quality: Optional[float] = None
    status: str
    recorded_at: Optional[str] = None


class FaceStatusResponse(BaseModel):
    patient_id: str
    session_id: Optional[str] = None
    detected: bool
    faces_detected: int
    posture: Optional[PostureOut] = None
    recorded_at: Optional[str] = None


class AssessmentRequest(BaseModel):
    session_id: Optional[str] = None
    patient_id: Optional[str] = None
    vision: Optional[Dict[str, Any]] = None
    audio: Optional[Dict[str, Any]] = None
    transcript: Optional[str] = None
    patient_context: Optional[Dict[str, Any]] = None


class AssessmentResponse(BaseModel):
    session_id: Optional[str] = None
    patient_id: str
    triage: TriageResponse
    findings: List[str]
    summary: List[str]


class RoomAssignmentRequest(BaseModel):
    session_id: Optional[str] = None
    patient_id: Optional[str] = None
    assessment: Optional[Dict[str, Any]] = None


class RoomAssignmentResponse(BaseModel):
    session_id: Optional[str] = None
    patient_id: str
    department: str
    department_ja: str
    doctor: str
    room: str
    queue_position: str
    est_wait_min: int
    confidence_score: int
    inputs_count: int
    latency_sec: float
    model: str
    assigned_at: str


app = FastAPI(
    title="MedAI Kiosk API",
    description="Unified healthcare kiosk API for vision, speech, triage, and routing",
    version="2.0.0",
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins or ["*"],
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

_patients: Dict[str, Dict[str, Any]] = {}
_latest_vision: Dict[str, Dict[str, Any]] = {}
_latest_audio: Dict[str, Dict[str, Any]] = {}
_latest_triage: Dict[str, Dict[str, Any]] = {}
_sessions: Dict[str, Dict[str, Any]] = {}
_assignments: Dict[str, Dict[str, Any]] = {}


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


def _as_patient_out(patient: Dict[str, Any]) -> PatientOut:
    pid = patient["id"]
    enriched = {
        **patient,
        "latest_vitals": _latest_vision.get(pid),
        "latest_audio": _latest_audio.get(pid),
        "latest_triage": _latest_triage.get(pid),
    }
    return PatientOut(**enriched)


def _ensure_patient(patient_id: str) -> PatientOut:
    patient = _patients.get(patient_id)
    if not patient and _supabase:
        try:
            resp = _supabase.table("patients").select("*").eq("id", patient_id).limit(1).execute()
            rows = resp.data or []
            if rows:
                row = rows[0]
                now = row.get("created_at") or _now_iso()
                patient = {
                    "id": row["id"],
                    "name": row.get("name") or "Kiosk Patient",
                    "age": row.get("age"),
                    "conditions": row.get("conditions") or [],
                    "room": row.get("room"),
                    "current_triage_level": row.get("current_triage_level") or "A",
                    "urgency_score": row.get("urgency_score") or 0.0,
                    "created_at": now,
                    "updated_at": row.get("updated_at") or now,
                }
                _patients[patient_id] = patient
        except Exception as exc:
            print(f"[WARN] Supabase patient fetch failed: {exc}")
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return _as_patient_out(patient)


def _create_patient(payload: PatientCreate) -> PatientOut:
    now = _now_iso()
    patient_id = payload.id or _new_id()
    patient: Dict[str, Any] = {
        **payload.model_dump(exclude_none=True),
        "id": patient_id,
        "current_triage_level": "A",
        "urgency_score": 0.0,
        "created_at": now,
        "updated_at": now,
    }
    _patients[patient_id] = patient

    if _supabase:
        try:
            db_payload = {
                "id": patient_id,
                "name": patient["name"],
                "room": patient.get("room"),
                "age": patient.get("age"),
                "conditions": patient.get("conditions") or [],
                "current_triage_level": "A",
                "urgency_score": 0.0,
            }
            _supabase.table("patients").upsert(db_payload).execute()
        except Exception as exc:
            print(f"[WARN] Supabase patient upsert failed: {exc}")

    return _as_patient_out(patient)


def _vision_payload(patient_id: str, result: Any, processed_at: str) -> Dict[str, Any]:
    return {
        "patient_id": patient_id,
        "vitals": {
            "spo2": result.spo2,
            "heart_rate": result.heart_rate,
            "signal_quality": result.signal_quality,
        },
        "posture": {
            "status": result.posture_status,
            "confidence": result.posture_confidence,
            "fall_detected": result.fall_detected,
            "immobile_seconds": result.immobile_seconds,
        },
        "faces_detected": result.faces_detected,
        "processed_at": processed_at,
    }


def _store_vitals(patient_id: str, result: Any, processed_at: str) -> Dict[str, Any]:
    payload = _vision_payload(patient_id, result, processed_at)
    _latest_vision[patient_id] = payload
    if patient_id in _patients:
        _patients[patient_id]["updated_at"] = processed_at

    if _supabase:
        try:
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
            _supabase.table("patients").update({"last_seen_at": processed_at}).eq("id", patient_id).execute()
        except Exception as exc:
            print(f"[WARN] Supabase vital insert failed: {exc}")

    return payload


def _store_audio(patient_id: str, features: AudioFeatures, stored_at: str) -> Dict[str, Any]:
    payload = {
        "patient_id": patient_id,
        "audio_features": features.model_dump(),
        "stored_at": stored_at,
    }
    _latest_audio[patient_id] = features.model_dump()
    if patient_id in _patients:
        _patients[patient_id]["updated_at"] = stored_at

    if _supabase:
        try:
            _supabase.table("audio_readings").insert({
                "patient_id": patient_id,
                "breath_rate": features.breath_rate,
                "cough_detected": features.cough_detected,
                "distress_score": features.distress_score,
                "speech_clarity": features.speech_clarity,
                "recorded_at": stored_at,
            }).execute()
        except Exception as exc:
            print(f"[WARN] Supabase audio insert failed: {exc}")

    return payload


def _store_triage(patient_id: str, triage: Any) -> Dict[str, Any]:
    payload = {
        "patient_id": patient_id,
        "triage_level": triage.triage_level,
        "urgency_score": triage.urgency_score,
        "reasons": triage.reasons,
        "recommended_action": triage.recommended_action,
        "decided_at": triage.decided_at,
    }
    _latest_triage[patient_id] = payload
    if patient_id in _patients:
        _patients[patient_id].update({
            "current_triage_level": triage.triage_level,
            "urgency_score": triage.urgency_score,
            "updated_at": triage.decided_at,
        })

    if _supabase:
        try:
            _supabase.table("triage_events").insert(payload).execute()
            _supabase.table("patients").update({
                "current_triage_level": triage.triage_level,
                "urgency_score": triage.urgency_score,
                "last_seen_at": triage.decided_at,
            }).eq("id", patient_id).execute()
        except Exception as exc:
            print(f"[WARN] Supabase triage insert failed: {exc}")

    return payload


def _fetch_latest_vision(patient_id: str) -> Optional[Dict[str, Any]]:
    if patient_id in _latest_vision:
        stored = _latest_vision[patient_id]
        return {"vitals": stored["vitals"], "posture": stored["posture"]}
    if not _supabase:
        return None
    try:
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
                "spo2": row.get("spo2", 0),
                "heart_rate": row.get("heart_rate", 0),
                "signal_quality": row.get("signal_quality", 0),
            },
            "posture": {
                "status": row.get("posture_status", "UNKNOWN"),
                "fall_detected": row.get("fall_detected", False),
                "immobile_seconds": row.get("immobile_seconds", 0),
                "confidence": 0.8,
            },
        }
    except Exception as exc:
        print(f"[WARN] Supabase latest vision fetch failed: {exc}")
        return None


def _fetch_latest_audio(patient_id: str) -> Optional[Dict[str, Any]]:
    if patient_id in _latest_audio:
        return _latest_audio[patient_id]
    if not _supabase:
        return None
    try:
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
    except Exception as exc:
        print(f"[WARN] Supabase latest audio fetch failed: {exc}")
        return None


def _fetch_patient_context(patient_id: str) -> Optional[Dict[str, Any]]:
    patient = _patients.get(patient_id)
    if patient:
        return {
            "name": patient.get("name", "Unknown"),
            "age": patient.get("age") or 0,
            "conditions": patient.get("conditions") or [],
            "symptoms": patient.get("symptoms"),
        }
    if not _supabase:
        return None
    try:
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
    except Exception as exc:
        print(f"[WARN] Supabase patient context fetch failed: {exc}")
        return None


def _session_or_404(session_id: str) -> Dict[str, Any]:
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Measurement session not found")
    return session


def _patient_id_from_request(session_id: Optional[str], patient_id: Optional[str]) -> str:
    if patient_id:
        return patient_id
    if session_id:
        return str(_session_or_404(session_id)["patient_id"])
    raise HTTPException(status_code=400, detail="session_id or patient_id is required")


def _run_triage(
    patient_id: str,
    vision: Optional[Dict[str, Any]] = None,
    audio: Optional[Dict[str, Any]] = None,
    patient_context: Optional[Dict[str, Any]] = None,
) -> TriageResponse:
    vision_data = vision or _fetch_latest_vision(patient_id)
    audio_data = audio or _fetch_latest_audio(patient_id)
    context = patient_context or _fetch_patient_context(patient_id)

    if not vision_data and not audio_data:
        raise HTTPException(
            status_code=400,
            detail="No vision or audio data provided and none found for this patient",
        )

    triage = fusion_engine.decide(
        vision=vision_data,
        audio=audio_data,
        patient_context=context,
    )
    payload = _store_triage(patient_id, triage)
    return TriageResponse(**payload)


async def _transcribe_upload(
    audio_file: UploadFile,
    patient_id: Optional[str],
    language: Optional[str] = None,
) -> TranscriptionResponse:
    suffix = Path(audio_file.filename or "audio.wav").suffix or ".wav"
    tmp_path: Optional[Path] = None
    try:
        content = await audio_file.read()
        if len(content) > MAX_AUDIO_BYTES:
            raise HTTPException(status_code=413, detail="Audio file is too large")
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(content)

        transcriber = _get_speech_transcriber()
        segments, detected_language = transcriber.transcribe_file(str(tmp_path), language=language)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Audio transcription failed: {exc}") from exc
    finally:
        if tmp_path:
            tmp_path.unlink(missing_ok=True)

    processed_at = _now_iso()
    transcript = " ".join(segment.text for segment in segments).strip()
    return TranscriptionResponse(
        patient_id=patient_id,
        language=detected_language or language,
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


def _build_assignment(
    patient_id: str,
    session_id: Optional[str] = None,
    assessment: Optional[Dict[str, Any]] = None,
) -> RoomAssignmentResponse:
    triage = assessment or _latest_triage.get(patient_id) or {}
    level = triage.get("triage_level", "A")
    reasons = " ".join(triage.get("reasons") or []).lower()
    patient = _patients.get(patient_id, {})
    symptoms = str(patient.get("symptoms") or "").lower()

    if level == "C":
        department, department_ja, doctor, room, wait = "Emergency Medicine", "救急科", "Emergency Team", "ER-01", 0
    elif "spo" in reasons or "cough" in reasons or "breath" in reasons or "respir" in symptoms:
        department, department_ja, doctor, room, wait = "Respiratory Medicine", "呼吸器内科", "On-call Physician", "305 - 3F East Wing", 8
    elif "heart" in reasons or "cardio" in symptoms:
        department, department_ja, doctor, room, wait = "Cardiology", "循環器内科", "On-call Physician", "212 - 2F West Wing", 12
    else:
        department, department_ja, doctor, room, wait = "General Medicine", "総合内科", "On-call Physician", "201 - 2F Central", 15

    confidence = 96 if level == "C" else 88 if level == "B" else 76
    active_count = len([p for p in _patients.values() if p.get("current_triage_level") == level])
    response = RoomAssignmentResponse(
        session_id=session_id,
        patient_id=patient_id,
        department=department,
        department_ja=department_ja,
        doctor=doctor,
        room=room,
        queue_position=f"#{max(active_count, 1)}",
        est_wait_min=wait,
        confidence_score=confidence,
        inputs_count=sum(bool(x) for x in [_latest_vision.get(patient_id), _latest_audio.get(patient_id), triage, patient]),
        latency_sec=0.1,
        model="MedAI Rule Routing v1",
        assigned_at=_now_iso(),
    )
    _assignments[patient_id] = response.model_dump()
    return response


@app.get("/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "version": "2.0.0",
        "supabase_connected": _supabase is not None,
        "patients_cached": len(_patients),
        "sessions_active": len([s for s in _sessions.values() if s["status"] == "active"]),
    }


@app.post("/api/patients", response_model=PatientOut)
def create_patient(patient: PatientCreate) -> PatientOut:
    return _create_patient(patient)


@app.get("/api/patients")
def list_patients() -> Dict[str, List[PatientOut]]:
    if _supabase:
        try:
            resp = _supabase.table("patients").select("*").order("updated_at", desc=True).limit(100).execute()
            for row in resp.data or []:
                pid = row["id"]
                _patients.setdefault(pid, {
                    "id": pid,
                    "name": row.get("name") or "Kiosk Patient",
                    "age": row.get("age"),
                    "room": row.get("room"),
                    "conditions": row.get("conditions") or [],
                    "current_triage_level": row.get("current_triage_level") or "A",
                    "urgency_score": row.get("urgency_score") or 0.0,
                    "created_at": row.get("created_at") or _now_iso(),
                    "updated_at": row.get("updated_at") or _now_iso(),
                })
        except Exception as exc:
            print(f"[WARN] Supabase patient list failed: {exc}")

    order = {"C": 0, "B": 1, "A": 2}
    patients = sorted(
        (_as_patient_out(patient) for patient in _patients.values()),
        key=lambda p: (order.get(p.current_triage_level, 3), -p.urgency_score),
    )
    return {"patients": patients}


@app.get("/api/patients/{patient_id}", response_model=PatientOut)
def get_patient(patient_id: str) -> PatientOut:
    return _ensure_patient(patient_id)


@app.post("/api/vision", response_model=VisionResponse)
def process_vision(req: VisionRequest) -> VisionResponse:
    try:
        result = vision_analyzer.analyze_base64(req.image_base64, req.patient_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Vision processing failed: {exc}") from exc

    processed_at = _now_iso()
    payload = _store_vitals(req.patient_id, result, processed_at)
    return VisionResponse(**payload)


@app.post("/api/audio", response_model=AudioResponse)
def process_audio(req: AudioRequest) -> AudioResponse:
    stored_at = _now_iso()
    payload = _store_audio(req.patient_id, req.audio_features, stored_at)
    return AudioResponse(**payload)


@app.post("/api/audio/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    patient_id: Optional[str] = Form(default=None),
) -> TranscriptionResponse:
    return await _transcribe_upload(audio_file, patient_id)


@app.post("/api/speech/en/transcribe", response_model=TranscriptionResponse)
async def transcribe_english(
    audio_file: UploadFile = File(...),
    patient_id: Optional[str] = Form(default=None),
) -> TranscriptionResponse:
    return await _transcribe_upload(audio_file, patient_id, language="en")


@app.post("/api/speech/ja/transcribe", response_model=TranscriptionResponse)
async def transcribe_japanese(
    audio_file: UploadFile = File(...),
    patient_id: Optional[str] = Form(default=None),
) -> TranscriptionResponse:
    return await _transcribe_upload(audio_file, patient_id, language="ja")


@app.post("/api/triage", response_model=TriageResponse)
def process_triage(req: TriageRequest) -> TriageResponse:
    return _run_triage(req.patient_id, req.vision, req.audio, req.patient_context)


@app.post("/api/measurements/start", response_model=MeasurementStartResponse)
def start_measurement(req: MeasurementStartRequest) -> MeasurementStartResponse:
    if req.patient_id:
        patient = _ensure_patient(req.patient_id)
    else:
        patient = _create_patient(req.patient or PatientCreate())

    started_at = _now_iso()
    session_id = _new_id()
    _sessions[session_id] = {
        "id": session_id,
        "patient_id": patient.id,
        "status": "active",
        "language": req.language,
        "metadata": req.metadata,
        "started_at": started_at,
        "updated_at": started_at,
    }
    return MeasurementStartResponse(
        session_id=session_id,
        patient=patient,
        status="active",
        started_at=started_at,
    )


@app.post("/api/measurements/{session_id}/frame", response_model=VisionResponse)
def process_measurement_frame(session_id: str, req: MeasurementFrameRequest) -> VisionResponse:
    session = _session_or_404(session_id)
    if session["status"] != "active":
        raise HTTPException(status_code=409, detail="Measurement session is not active")
    patient_id = str(session["patient_id"])
    response = process_vision(VisionRequest(
        patient_id=patient_id,
        image_base64=req.image_base64,
        timestamp=req.timestamp,
        metadata=req.metadata,
    ))
    session["updated_at"] = response.processed_at
    return response


@app.post("/api/measurements/{session_id}/stop", response_model=MeasurementStopResponse)
def stop_measurement(session_id: str) -> MeasurementStopResponse:
    session = _session_or_404(session_id)
    patient_id = str(session["patient_id"])
    stopped_at = _now_iso()
    latest_triage: Optional[Dict[str, Any]] = None
    if _fetch_latest_vision(patient_id) or _fetch_latest_audio(patient_id):
        latest_triage = _run_triage(patient_id).model_dump()
    session.update({"status": "stopped", "stopped_at": stopped_at, "updated_at": stopped_at})
    return MeasurementStopResponse(
        session_id=session_id,
        patient_id=patient_id,
        status="stopped",
        stopped_at=stopped_at,
        latest_vitals=_latest_vision.get(patient_id),
        latest_triage=latest_triage,
    )


@app.get("/api/measurements/{session_id}/heart-rate", response_model=MetricResponse)
def get_heart_rate(session_id: str) -> MetricResponse:
    session = _session_or_404(session_id)
    patient_id = str(session["patient_id"])
    latest = _latest_vision.get(patient_id)
    vitals = latest.get("vitals") if latest else {}
    value = vitals.get("heart_rate")
    return MetricResponse(
        session_id=session_id,
        patient_id=patient_id,
        metric="heart_rate",
        value=value if value and value > 0 else None,
        unit="bpm",
        signal_quality=vitals.get("signal_quality"),
        status="ready" if value and value > 0 else "collecting",
        recorded_at=latest.get("processed_at") if latest else None,
    )


@app.get("/api/measurements/{session_id}/spo2", response_model=MetricResponse)
def get_spo2(session_id: str) -> MetricResponse:
    session = _session_or_404(session_id)
    patient_id = str(session["patient_id"])
    latest = _latest_vision.get(patient_id)
    vitals = latest.get("vitals") if latest else {}
    value = vitals.get("spo2")
    return MetricResponse(
        session_id=session_id,
        patient_id=patient_id,
        metric="spo2",
        value=value if value and value > 0 else None,
        unit="%",
        signal_quality=vitals.get("signal_quality"),
        status="ready" if value and value > 0 else "collecting",
        recorded_at=latest.get("processed_at") if latest else None,
    )


@app.get("/api/measurements/{session_id}/face-status", response_model=FaceStatusResponse)
def get_face_status(session_id: str) -> FaceStatusResponse:
    session = _session_or_404(session_id)
    patient_id = str(session["patient_id"])
    latest = _latest_vision.get(patient_id)
    posture = latest.get("posture") if latest else None
    return FaceStatusResponse(
        session_id=session_id,
        patient_id=patient_id,
        detected=bool(latest and latest.get("faces_detected", 0) > 0),
        faces_detected=int(latest.get("faces_detected", 0)) if latest else 0,
        posture=PostureOut(**posture) if posture else None,
        recorded_at=latest.get("processed_at") if latest else None,
    )


@app.post("/api/assessment", response_model=AssessmentResponse)
@app.post("/api/assessments", response_model=AssessmentResponse)
def create_assessment(req: AssessmentRequest) -> AssessmentResponse:
    patient_id = _patient_id_from_request(req.session_id, req.patient_id)
    triage = _run_triage(patient_id, req.vision, req.audio, req.patient_context)
    findings = triage.reasons
    summary = [
        f"Triage level {triage.triage_level}",
        f"Urgency score {round(triage.urgency_score * 100)} / 100",
        triage.recommended_action,
    ]
    return AssessmentResponse(
        session_id=req.session_id,
        patient_id=patient_id,
        triage=triage,
        findings=findings,
        summary=summary,
    )


@app.post("/api/assignments", response_model=RoomAssignmentResponse)
def create_assignment(req: RoomAssignmentRequest) -> RoomAssignmentResponse:
    patient_id = _patient_id_from_request(req.session_id, req.patient_id)
    return _build_assignment(patient_id, req.session_id, req.assessment)


@app.get("/api/results/{session_id}")
def get_results(session_id: str) -> Dict[str, Any]:
    session = _session_or_404(session_id)
    patient_id = str(session["patient_id"])
    return {
        "session": session,
        "patient": _as_patient_out(_patients[patient_id]).model_dump() if patient_id in _patients else None,
        "latest_vision": _latest_vision.get(patient_id),
        "latest_audio": _latest_audio.get(patient_id),
        "latest_triage": _latest_triage.get(patient_id),
        "assignment": _assignments.get(patient_id),
        "generated_at": _now_iso(),
    }
