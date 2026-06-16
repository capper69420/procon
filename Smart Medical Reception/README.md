# Smart Medical Reception - Full Stack

Multimodal patient monitoring: **vision** (FaceMesh r-PPG + fall detection), **audio** distress features, **local speech-to-text** with faster-whisper, **fusion triage**, and a **React doctor dashboard** with optional Supabase real-time updates.

## Quick Start

Run the backend and frontend in two separate PowerShell windows.

### 1. Backend (FastAPI)

```powershell
cd "C:\Users\Administrator\Downloads\procon-main\procon-main\Smart Medical Reception"

python -m venv .venv
.\.venv\Scripts\activate

pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Backend URLs:

```text
http://localhost:8000
http://localhost:8000/docs
```

The first speech-to-text request may take extra time because faster-whisper downloads the selected model.

### 2. Frontend (React)

Open a second PowerShell window:

```powershell
cd "C:\Users\Administrator\Downloads\procon-main\procon-main\Smart Medical Reception\frontend"

npm install
npm run dev
```

Dashboard URL:

```text
http://localhost:5173
```

## Optional Supabase Setup

The app can run without Supabase in demo mode. To enable persistence and real-time dashboard updates:

1. Create a free project at [supabase.com](https://supabase.com)
2. Open **SQL Editor**, paste `schema.sql`, and run it
3. Copy your project URL and anon key
4. Set backend environment variables before starting `uvicorn`:

```powershell
$env:SUPABASE_URL = "https://your-project.supabase.co"
$env:SUPABASE_KEY = "your-anon-key"
```

5. Create `frontend/.env`:

```env
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key
```

## Speech-to-Text Test

After the backend is running, upload a real audio file:

```powershell
curl.exe -X POST "http://localhost:8000/api/audio/transcribe" `
  -F "audio_file=@C:\Users\Administrator\Downloads\test.wav" `
  -F "patient_id=550e8400-e29b-41d4-a716-446655440003"
```

Replace `C:\Users\Administrator\Downloads\test.wav` with your actual `.wav`, `.mp3`, or `.m4a` file path.

To find audio files in Downloads:

```powershell
Get-ChildItem C:\Users\Administrator\Downloads -Recurse -Include *.wav,*.mp3,*.m4a
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/api/vision` | Analyze base64 camera frame |
| POST | `/api/audio` | Ingest audio features |
| POST | `/api/audio/transcribe` | Transcribe uploaded audio with local faster-whisper |
| POST | `/api/triage` | Fuse signals to Level A/B/C |

See `technical_foundation.md` for full request/response contracts.

## Project Structure

```text
Smart Medical Reception/
├── main.py                  # FastAPI server
├── speech_transcriber.py    # faster-whisper transcription wrapper
├── facemesh.py              # FaceMesh + r-PPG + fall detection
├── fusion_engine.py         # Multimodal triage fusion
├── health_monitor.py        # Original local webcam app
├── schema.sql               # Supabase PostgreSQL setup
├── technical_foundation.md  # Architecture + API spec
├── requirements.txt         # Python dependencies
└── frontend/
    └── src/
        ├── api.js
        ├── App.jsx
        └── supabaseClient.js
```

## Demo Without Supabase

The dashboard runs in demo mode with mock Level C/B/A patients when Supabase environment variables are missing. The API works standalone without persistence.

## Hackathon Demo Flow

1. Start backend and frontend
2. POST a webcam frame to `/api/vision` with a seed patient ID
3. POST mock audio features to `/api/audio`, or upload speech to `/api/audio/transcribe`
4. POST `/api/triage` and watch the dashboard update

Example patient ID: `550e8400-e29b-41d4-a716-446655440003`

## Local Speech-to-Text Settings

The `faster-whisper-stt` code is merged into the backend through `speech_transcriber.py`. Optional backend environment variables:

```powershell
$env:WHISPER_MODEL = "base"
$env:WHISPER_DEVICE = "auto"
$env:WHISPER_COMPUTE_TYPE = "auto"
$env:WHISPER_LANGUAGES = "ja,en"
$env:WHISPER_BEAM_SIZE = "1"
```
