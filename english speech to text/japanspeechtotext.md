KotobaMic JA Live Transcriber
KotobaMic JA Live Transcriber is a small real-time Japanese speech-to-text script that listens to the microphone, buffers short audio chunks, and transcribes them with the Kotoba Faster Whisper model.

The main script is:

C:\Users\tsah-05\OneDrive\Desktop\health_monitor\realtimespeachtotextja.py
What It Does
Captures live microphone audio at 16 kHz.
Collects audio into 2-second chunks.
Sends each chunk to faster-whisper.
Uses the Japanese language setting: language="ja".
Prints recognized Japanese text to the console as it is detected.
Requirements
The script expects these Python packages:

sounddevice
numpy
faster-whisper
It currently uses this model:

kotoba-tech/kotoba-whisper-v2.0-faster
The current script is configured for CUDA:

model = WhisperModel(
    "kotoba-tech/kotoba-whisper-v2.0-faster",
    device="cuda",
    compute_type="float16",
)
That means it needs a working NVIDIA GPU setup, including the CUDA runtime libraries required by faster-whisper.

Run
From the project folder:

cd C:\Users\tsah-05\OneDrive\Desktop\health_monitor
.\.venv311\Scripts\python.exe realtimespeachtotextja.py
When it starts correctly, you should see:

Listening... Press Ctrl + C to stop.
Speak Japanese into the microphone. Transcribed text will print in the terminal.

To stop it, press:

Ctrl + C
CPU Fallback
If CUDA is not installed correctly, or you see an error like:

RuntimeError: Library cublas64_12.dll is not found or cannot be loaded
change the model setup to CPU mode:

model = WhisperModel(
    "kotoba-tech/kotoba-whisper-v2.0-faster",
    device="cpu",
    compute_type="int8",
)
CPU mode is slower, but it avoids CUDA runtime problems.

Troubleshooting
If no text appears:

Make sure the correct microphone is selected in Windows.
Speak clearly for at least a few seconds.
Check that the app has microphone permission.
Confirm the model has finished downloading from Hugging Face.
If the script fails on startup:

Verify the virtual environment exists at .venv311.
Check that sounddevice, numpy, and faster-whisper are installed.
If using CUDA, confirm the NVIDIA driver and CUDA libraries are available.
Notes
The first run may take longer because the Kotoba model can be downloaded and cached locally. Hugging Face may also show warnings if no HF_TOKEN is configured; the script can still work without one, but authenticated downloads may be faster and more reliable.
