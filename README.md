# Standalone Speech-To-Text and Health Monitor

This folder contains only the two standalone tools from the original project. It does not use the FastAPI backend, React frontend, Supabase, or any dashboard code.

## Folder layout

```text
standalone_speech_health/
├── health_monitor/
│   ├── health_monitor.py
│   ├── requirements.txt
│   └── run_health_monitor.bat
└── speech_to_text/
    ├── speech_to_text.py
    ├── audio_capture.py
    ├── streamer.py
    ├── transcriber.py
    ├── requirements.txt
    └── run_speech_to_text.bat
```

## Speech-to-text

Install:

```powershell
cd speech_to_text
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Run microphone transcription:

```powershell
python speech_to_text.py
```

Transcribe an audio file:

```powershell
python speech_to_text.py --file "C:\path\to\audio.wav" --output transcript.txt
```

List microphones:

```powershell
python speech_to_text.py --list-devices
```

The first run may download the selected Whisper model.

## Health monitor

Install:

```powershell
cd health_monitor
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Run:

```powershell
python health_monitor.py --camera 0
```

If the webcam does not open, try:

```powershell
python health_monitor.py --camera 1
```

Controls:

```text
q  quit
s  toggle SpO2/HR panels
```
