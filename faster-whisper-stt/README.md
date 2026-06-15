# Real-Time Speech-to-Text (faster-whisper)

Local, low-latency microphone transcription for **Japanese** and **English** only. No translation, no cloud APIs.

## Requirements

- Python 3.10+
- Microphone
- Optional: NVIDIA GPU with CUDA 12 + cuDNN 9 for GPU inference

## Installation

```bash
cd faster-whisper-stt
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

First run downloads the Whisper model from Hugging Face (size depends on `--model`).

### GPU notes

- GPU uses `float16` automatically when `--device auto` detects CUDA.
- CPU uses `int8` automatically.
- If CUDA libraries are missing, use `--device cpu`.

## Usage

Start transcription (default `base` model):

```bash
python app.py
```

Press **Enter** to stop. **Ctrl+C** also exits.

### Examples

Fast CPU setup:

```bash
python app.py --model base --device cpu
```

Better accuracy on GPU:

```bash
python app.py --model distil-large-v3 --device cuda --compute-type float16
```

Lower latency (shorter chunks):

```bash
python app.py --model small --chunk-seconds 1.5 --beam-size 1
```

Save final transcript:

```bash
python app.py --model base -o transcript.txt
```

List microphones:

```bash
python app.py --list-devices
```

## CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `base` | Model name (`base`, `small`, `distil-large-v3`, ...) |
| `--device` | `auto` | `auto`, `cuda`, or `cpu` |
| `--compute-type` | `auto` | `float16` on CUDA, `int8` on CPU |
| `--chunk-seconds` | `2.0` | Streaming chunk length |
| `--overlap-seconds` | `0.3` | Overlap to reduce word cuts |
| `--beam-size` | `1` | Lower = faster |
| `-o`, `--output` | — | Save final transcript to file |

## Architecture

- `audio_capture.py` — continuous microphone input
- `transcriber.py` — faster-whisper chunk transcription with VAD
- `streamer.py` — chunk buffering, worker thread, live updates
- `app.py` — CLI wiring and configuration

## Behavior

- Auto-detects language per chunk; only `ja` and `en` are accepted.
- Other languages are ignored (not translated).
- VAD skips silent audio.
- Partial results print immediately with timestamps.
- Neighboring chunk overlap is merged to reduce duplicated words.
