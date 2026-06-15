# Health Monitor â€” Full Stack

Multimodal patient monitoring: **vision** (FaceMesh r-PPG + fall detection), **audio** distress features, **fusion triage**, and a **React doctor dashboard** with Supabase real-time updates.

## Quick Start

### 1. Database (Supabase)

1. Create a free project at [supabase.com](https://supabase.com)
2. Open **SQL Editor** â†’ paste and run `schema.sql`
3. Copy your project URL and anon key

### 2. Backend (FastAPI)

```powershell
cd C:\Users\tsah-05\Downloads\health-monitor
.\env12\Scripts\activate
pip install -r requirements.txt

# Set env vars (or copy .env.example)
$env:SUPABASE_URL = "https://your-project.supabase.co"
$env:SUPABASE_KEY = "your-anon-key"

uvicorn main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

### 3. Frontend (React)

```powershell
cd frontend
npm install

# Create frontend/.env with VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY
npm run dev
```

Dashboard: http://localhost:5173

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/api/vision` | Analyze base64 camera frame |
| POST | `/api/audio` | Ingest audio features |
| POST | `/api/audio/transcribe` | Transcribe uploaded audio with local faster-whisper |
| POST | `/api/triage` | Fuse signals â†’ Level A/B/C |

See `technical_foundation.md` for full request/response contracts.

## Project Structure

```
health-monitor/
â”œâ”€â”€ main.py                  # FastAPI server
â”œâ”€â”€ facemesh.py              # FaceMesh + r-PPG + fall detection
â”œâ”€â”€ fusion_engine.py         # Multimodal triage fusion
â”œâ”€â”€ health_monitor.py        # Original local webcam app (unchanged)
â”œâ”€â”€ schema.sql               # Supabase PostgreSQL setup
â”œâ”€â”€ technical_foundation.md  # Architecture + API spec
â””â”€â”€ frontend/
    â””â”€â”€ src/
        â”œâ”€â”€ DoctorDashboard.jsx
        â””â”€â”€ supabaseClient.js
```

## Demo Without Supabase

The dashboard runs in **demo mode** with mock Level C/B/A patients when Supabase env vars are missing. The API works standalone without persistence.

## Hackathon Demo Flow

1. Start backend + frontend
2. POST a webcam frame to `/api/vision` with a seed patient ID
3. POST mock audio to `/api/audio`
4. POST `/api/triage` â€” watch the dashboard update in real time

Example patient ID (Level C seed): `550e8400-e29b-41d4-a716-446655440003`

## Local Speech-to-Text

The `faster-whisper-stt` code is merged into the backend through `speech_transcriber.py`. Upload an audio file to `/api/audio/transcribe` as multipart form data using field name `audio_file`; optionally include `patient_id`.

Optional backend environment variables:

```powershell
$env:WHISPER_MODEL = "base"
$env:WHISPER_DEVICE = "auto"
$env:WHISPER_COMPUTE_TYPE = "auto"
$env:WHISPER_LANGUAGES = "ja,en"
$env:WHISPER_BEAM_SIZE = "1"
```
