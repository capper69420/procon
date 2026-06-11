# Smart Medical Reception Kiosk
## Technical Foundation — Kosen Procon
### AI Triage & Task-Shifting System

---

## SECTION 1: ARCHITECTURE DECISION — FastAPI vs Express.js

### ✅ VERDICT: FastAPI (Python) — No Contest

For this specific project, Express.js would be a significant strategic mistake. Here's the engineering rationale:

| Factor | FastAPI (Python) ✅ | Express.js (Node.js) ❌ |
|--------|-------------------|------------------------|
| **AI Integration** | Same process, direct import | Requires IPC, subprocess, or microservice |
| **OpenCV / MediaPipe / YOLO** | `import cv2, mediapipe, ultralytics` — native | Would need to spawn Python child process |
| **Whisper API call** | Direct `openai` SDK in Python | Must proxy through HTTP to a Python server |
| **Pydantic Validation** | Auto-validates & serializes AI model output | Manual validation schemas (Zod/Joi) |
| **Auto API Docs** | Swagger UI at `/docs` — free, zero config | Requires swagger-jsdoc setup |
| **WebSocket (Live Dashboard)** | `FastAPI.WebSocket` + `asyncio` native | `ws` or `socket.io` — good but adds deps |
| **Async Processing** | `asyncio` + `BackgroundTasks` built-in | `async/await` but no built-in task queue |
| **Dev Speed (14 days)** | One language for AI + API + logic | Two languages, two repos to maintain |
| **Contest Presentation** | "We built the AI and API in one clean Python codebase" | "We have a Node proxy that calls our Python AI server" |

### Why This Matters for Your Architecture

With Express.js, your pipeline would be:
```
Camera → Python AI Script → [HTTP call] → Node.js API → Database
```
That's a cross-language HTTP hop in your critical path, adding latency and debugging complexity.

With FastAPI:
```
Camera → FastAPI endpoint (runs AI inline) → Database
```
Your AI/ML engineer and Backend engineer work in the same codebase, share the same Pydantic models, and deploy one service.

---

## SECTION 2: DATABASE SCHEMA

### Choice: Supabase (PostgreSQL) over Firebase

| | Supabase ✅ | Firebase |
|--|------------|---------|
| **Real-time** | ✅ Row-level subscriptions | ✅ Native |
| **Querying** | ✅ Full SQL + JOINs | ❌ Limited, no joins |
| **Data relationships** | ✅ Foreign keys, referential integrity | ❌ Manual |
| **Auto REST API** | ✅ PostgREST (zero config) | ❌ Custom only |
| **Contest Impressiveness** | "We used PostgreSQL with a proper relational schema" | "We used Firebase" |
| **Analytics/Reporting** | ✅ SQL aggregations | ❌ Complex |
| **Free tier** | ✅ 500MB, unlimited API calls | ✅ Generous |

### Entity Relationship Overview

```
staff
  ↑
  |  (assigned_doctor_id, override_by)
  |
triage_sessions  ←—————————————————————————————→  alerts
  |                                                   |
  |——→ vision_analyses    (1:1 per session)           ↑
  |——→ audio_analyses     (1:1 per session)           |
  |——→ triage_decisions   (1:1 per session)     (target_staff_id)
  |——→ queue_entries      (1:1 per session)
```

### Key Design Decisions

**1. Session-centric model**: `triage_sessions` is the hub. Every other table has `session_id` FK. This maps exactly to your workflow: one patient interaction = one session.

**2. JSONB for AI output**: `yolo_detections`, `reported_symptoms`, `severity_keywords` are JSONB — flexible enough to evolve without migrations, but still queryable with `@>` operators.

**3. Flattened condition flags**: `detected_bleeding`, `detected_wheelchair` etc. are BOOLEAN columns *in addition to* the raw JSONB. This lets you write fast dashboard queries like:
```sql
SELECT * FROM vision_analyses WHERE detected_bleeding = TRUE;
```

**4. Audit trail**: `triage_decisions.fusion_weights` and `decision_rationale` store exactly how and why a level was assigned. Judges will love this — it shows your AI is explainable, not a black box.

**5. Real-time dashboard view**: The `active_triage_dashboard` view JOINs all tables and orders by Level C > B > A. One Supabase subscription to this view updates the Doctor's Dashboard instantly.

---

## SECTION 3: SYSTEM ARCHITECTURE DIAGRAM

```
┌─────────────────────────────────────────────────────┐
│                   KIOSK HARDWARE                     │
│  ┌──────────┐   ┌──────────────┐   ┌─────────────┐  │
│  │ Webcam   │   │  Microphone  │   │  Tablet/    │  │
│  │ (YOLO +  │   │  (Whisper)   │   │  Screen UI  │  │
│  │ FaceMesh)│   │              │   │  (React)    │  │
│  └────┬─────┘   └──────┬───────┘   └──────┬──────┘  │
└───────┼─────────────────┼─────────────────┼─────────┘
        │                 │                 │
        ▼                 ▼                 │
┌───────────────────────────────────────┐  │
│         FASTAPI BACKEND (Python)      │  │
│                                       │  │
│  ┌─────────────┐  ┌─────────────────┐ │  │
│  │ /triage/    │  │  /audio/        │ │  │
│  │ vision      │  │  analyze        │ │  │
│  │             │  │                 │ │  │
│  │ • FaceMesh  │  │ • Whisper API   │ │  │
│  │ • YOLOv8    │  │ • ChatGPT API   │ │  │
│  │ • PainScore │  │ • EHR Builder   │ │  │
│  └──────┬──────┘  └───────┬─────────┘ │  │
│         │                 │           │  │
│         ▼                 ▼           │  │
│  ┌───────────────────────────────┐    │  │
│  │     TRIAGE FUSION ENGINE      │    │  │
│  │  (Combines Vision + Audio     │    │  │
│  │   → Level A / B / C)          │    │  │
│  └──────────────┬────────────────┘    │  │
│                 │                     │  │
│  ┌──────────────▼────────────────┐    │  │
│  │      WebSocket Manager        │◄───┼──┘
│  │  (Real-time push to Doctor    │    │
│  │   Dashboard on Level C)       │    │
│  └──────────────┬────────────────┘    │
└─────────────────┼─────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│              SUPABASE (PostgreSQL)                   │
│  triage_sessions │ vision_analyses │ audio_analyses  │
│  triage_decisions│ queue_entries   │ alerts          │
└────────────────────────┬────────────────────────────┘
                         │  Real-time subscription
                         ▼
┌─────────────────────────────────────────────────────┐
│           DOCTOR'S DASHBOARD (React.js)              │
│  • Live patient queue (Level C → B → A order)        │
│  • EHR summary per patient                           │
│  • Pain score + detected conditions                  │
│  • One-click acknowledge / override triage level     │
└─────────────────────────────────────────────────────┘
```

---

## SECTION 4: 14-DAY SPRINT PLAN

### Day-by-Day Breakdown

| Days | Phase | AI/ML Engineer | Backend Engineer | Frontend Engineer | PM/QA |
|------|-------|----------------|-----------------|-------------------|-------|
| 1–2 | Setup | Env setup, YOLO/FaceMesh port | FastAPI scaffold, Supabase schema deploy | React app init, component plan | Kanban setup, API contract |
| 3–4 | Vision | FaceMesh pain scorer | `POST /triage/vision` endpoint | Kiosk Camera UI component | Vision module test cases |
| 5–6 | Audio | Whisper + ChatGPT integration | `POST /triage/audio` endpoint | Voice recording UI (elderly-friendly) | Audio module test cases |
| 7–8 | Fusion | Triage fusion algorithm (Level A/B/C logic) | Fusion engine + DB writes | Triage result animation UI | End-to-end integration test |
| 9–10 | Dashboard | - | WebSocket alerts, queue API | Doctor's Dashboard (live updates) | Dashboard test with mock data |
| 11–12 | Integration | Fine-tune pain scoring | Full pipeline test | Polish UI, multilingual labels | Bug bash |
| 13 | Demo Prep | Camera setup for demo | Deploy (Render/Railway) | Demo walk-through UI polish | Script the 40-min demo |
| 14 | Buffer | - | Hotfixes | Final UI tweaks | QA sign-off |

### Critical Path
```
Vision Module (Days 3-4)
        ↓
Audio Module (Days 5-6)     ← These MUST complete before Day 7
        ↓
Triage Fusion (Days 7-8)    ← Core demo feature
        ↓
Doctor Dashboard (Days 9-10)← Contest finale centerpiece
```

---

## SECTION 5: FASTAPI PROJECT STRUCTURE

```
smart_reception/
├── main.py                    # FastAPI app entry point
├── requirements.txt
├── .env                       # API keys (never commit)
│
├── api/
│   ├── routes/
│   │   ├── triage.py          # POST /triage/vision, /triage/audio, /triage/decision
│   │   ├── queue.py           # GET/PATCH /queue
│   │   ├── alerts.py          # POST /alerts, WebSocket /ws/dashboard
│   │   └── staff.py           # GET /staff
│   └── deps.py                # Supabase client injection
│
├── models/
│   ├── session.py             # Pydantic: TriageSession, TriageLevel
│   ├── vision.py              # Pydantic: VisionAnalysis, YOLODetection
│   ├── audio.py               # Pydantic: AudioAnalysis, EHRSummary
│   └── decision.py            # Pydantic: TriageDecision, FusionResult
│
├── services/
│   ├── vision/
│   │   ├── facemesh_scorer.py # Your existing FaceMesh pain scoring code
│   │   ├── yolo_detector.py   # Your existing YOLO detection code
│   │   └── vision_pipeline.py # Orchestrates both, returns VisionAnalysis
│   ├── audio/
│   │   ├── whisper_client.py  # Whisper API wrapper
│   │   ├── llm_summarizer.py  # ChatGPT EHR builder
│   │   └── audio_pipeline.py  # Orchestrates both, returns AudioAnalysis
│   └── triage/
│       ├── fusion_engine.py   # Level A/B/C classification logic
│       └── alert_dispatcher.py# Sends real-time alerts
│
└── db/
    ├── supabase_client.py     # Supabase Python client setup
    └── queries.py             # Reusable DB query functions
```

---

## SECTION 6: KEY API CONTRACTS

### `POST /triage/vision`
```json
Request:
{
  "session_id": "uuid",
  "frame_base64": "base64_encoded_jpeg..."
}

Response:
{
  "pain_score": 7.4,
  "detected_conditions": {
    "bleeding": false,
    "wheelchair": true,
    "unconscious": false
  },
  "face_action_units": { "AU4": 0.82, "AU9": 0.6 },
  "processing_ms": 120
}
```

### `POST /triage/audio`
```json
Request:
{
  "session_id": "uuid",
  "audio_base64": "base64_encoded_wav..."
}

Response:
{
  "detected_language": "mn",
  "transcript_original": "Цээж өвдөж байна, 2 цагийн өмнөөс...",
  "transcript_english": "My chest hurts, since 2 hours ago...",
  "chief_complaint": "Chest pain with onset 2 hours ago",
  "reported_symptoms": ["chest_pain", "dyspnea"],
  "self_reported_pain": 8,
  "ehr_summary": "Patient presents with acute chest pain...",
  "processing_ms": 1840
}
```

### `POST /triage/decision`
```json
Request:
{
  "session_id": "uuid"
}

Response:
{
  "triage_level": "C",
  "confidence": 0.94,
  "routing_destination": "emergency_room_1",
  "decision_rationale": "YOLO: wheelchair (0.88). FaceMesh pain: 7.4/10. Patient self-reports 8/10 chest pain (2h). Severity keywords: ['chest pain', 'cannot breathe']. Combined score: 9.1 → LEVEL C",
  "primary_triggers": ["wheelchair_detected", "high_pain_score", "chest_pain_keyword"]
}
```

### `WebSocket /ws/dashboard`
```json
// Server pushes on every Level C decision:
{
  "event": "new_level_c_patient",
  "session_id": "uuid",
  "arrived_at": "2025-06-11T09:23:11Z",
  "chief_complaint": "Chest pain with onset 2 hours ago",
  "pain_score": 7.4,
  "detected_conditions": ["wheelchair"],
  "routing_destination": "emergency_room_1",
  "queue_number": 3
}
```

---

## NEXT STEPS (Ask me for any of these)

1. **FaceMesh Pain Scoring Algorithm** — Convert AU values to a calibrated 0–10 pain score
2. **Triage Fusion Engine** — The exact weighted formula for A/B/C classification
3. **React Doctor's Dashboard** — Component code with live Supabase subscription
4. **Voice-First Kiosk UI** — Large-text, voice-activated React components for elderly patients
5. **Whisper + ChatGPT EHR Builder** — Prompt engineering for structured medical summaries
6. **Pitch Deck Structure** — 10-slide Procon presentation with impact metrics
