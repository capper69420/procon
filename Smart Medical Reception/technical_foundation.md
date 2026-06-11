# Health Monitor — Technical Foundation

## System Overview

Multimodal patient monitoring for hackathon demos: **vision** (webcam r-PPG + fall detection), **audio** (breathing/distress features), and a **fusion engine** that assigns triage levels. A FastAPI backend exposes REST endpoints; a React doctor dashboard subscribes to Supabase real-time updates.

```
┌─────────────┐     POST /api/vision      ┌──────────────┐
│ Edge Client │ ─────────────────────────▶│              │
│ (camera /   │     POST /api/audio       │   FastAPI    │──▶ Supabase
│  mic)       │ ─────────────────────────▶│   main.py    │    (PostgreSQL)
└─────────────┘     POST /api/triage      └──────┬───────┘
                                                  │
                     ┌────────────────────────────┼────────────────────────────┐
                     │                            │                            │
               facemesh.py                 fusion_engine.py              DoctorDashboard.jsx
            (FaceMesh + r-PPG)           (multimodal triage)            (real-time Level C UI)
```

## Triage Levels

| Level | Color  | Criteria (any trigger) |
|-------|--------|------------------------|
| **A** | Green  | SpO₂ ≥ 94, HR 50–120, no fall, distress < 0.3 |
| **B** | Yellow | SpO₂ 90–93, abnormal HR, minor distress, fall recovered |
| **C** | Red    | SpO₂ < 90, fall + immobile ≥ 15 s, distress ≥ 0.7, HR < 40 or > 140 |

Level **C** patients appear at the top of the doctor dashboard with pulsing red styling and live vitals.

---

## API Contracts

Base URL: `http://localhost:8000`

### `GET /health`

Health check.

**Response 200**
```json
{ "status": "ok", "version": "1.0.0" }
```

---

### `POST /api/vision`

Process a single camera frame (base64 JPEG/PNG) and return vitals + posture signals.

**Request**
```json
{
  "patient_id": "550e8400-e29b-41d4-a716-446655440000",
  "image_base64": "<base64-encoded image bytes>",
  "timestamp": "2026-06-11T14:30:00Z",
  "metadata": { "camera_id": "room-101", "fps": 30 }
}
```

**Response 200**
```json
{
  "patient_id": "550e8400-e29b-41d4-a716-446655440000",
  "vitals": {
    "spo2": 96.2,
    "heart_rate": 74.0,
    "signal_quality": 0.82
  },
  "posture": {
    "status": "STANDING",
    "confidence": 0.91,
    "fall_detected": false,
    "immobile_seconds": 0.0
  },
  "faces_detected": 1,
  "processed_at": "2026-06-11T14:30:01Z"
}
```

**Side effect:** Inserts a row into `vital_readings` when `SUPABASE_URL` is configured.

---

### `POST /api/audio`

Ingest pre-extracted audio features (from an edge mic pipeline or mock data).

**Request**
```json
{
  "patient_id": "550e8400-e29b-41d4-a716-446655440000",
  "audio_features": {
    "breath_rate": 22.0,
    "cough_detected": false,
    "distress_score": 0.15,
    "speech_clarity": 0.85
  },
  "timestamp": "2026-06-11T14:30:00Z"
}
```

**Response 200**
```json
{
  "patient_id": "550e8400-e29b-41d4-a716-446655440000",
  "audio_features": {
    "breath_rate": 22.0,
    "cough_detected": false,
    "distress_score": 0.15,
    "speech_clarity": 0.85
  },
  "stored_at": "2026-06-11T14:30:01Z"
}
```

**Side effect:** Inserts a row into `audio_readings` when Supabase is configured.

---

### `POST /api/triage`

Fuse the latest (or inline) vision + audio signals and assign a triage level.

**Request**
```json
{
  "patient_id": "550e8400-e29b-41d4-a716-446655440000",
  "vision": {
    "vitals": { "spo2": 88.5, "heart_rate": 110, "signal_quality": 0.7 },
    "posture": { "status": "FALLEN", "fall_detected": true, "immobile_seconds": 18 }
  },
  "audio": {
    "breath_rate": 28,
    "cough_detected": true,
    "distress_score": 0.8,
    "speech_clarity": 0.3
  },
  "patient_context": { "name": "Jane Doe", "age": 78, "conditions": ["COPD"] }
}
```

If `vision` / `audio` are omitted, the server loads the most recent readings from Supabase.

**Response 200**
```json
{
  "patient_id": "550e8400-e29b-41d4-a716-446655440000",
  "triage_level": "C",
  "urgency_score": 0.94,
  "reasons": [
    "SpO₂ critically low (88.5%)",
    "Fall detected with 18s immobility",
    "High distress score (0.80)"
  ],
  "recommended_action": "Immediate clinical review — possible respiratory distress post-fall",
  "decided_at": "2026-06-11T14:30:02Z"
}
```

**Side effect:** Upserts `patients.current_triage_level` and inserts into `triage_events`.

---

## Doctor Dashboard Visual Spec

- Dark theme (`#0f1419` background)
- **Level C** cards: red border pulse, large patient name, SpO₂ / HR / distress badges
- **Level B**: amber accent
- **Level A**: muted green
- Real-time via Supabase `postgres_changes` on `patients` and `triage_events`
- Sort order: C → B → A, then by `urgency_score` descending

---

## Environment Variables

| Variable | Used by | Description |
|----------|---------|-------------|
| `SUPABASE_URL` | backend + frontend | Supabase project URL |
| `SUPABASE_KEY` | backend | Service role or anon key |
| `VITE_SUPABASE_URL` | frontend | Same URL (Vite prefix) |
| `VITE_SUPABASE_ANON_KEY` | frontend | Anon key for browser client |
