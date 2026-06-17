import sounddevice as sd
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel

samplerate = 16000
block_duration = 0.5
chunk_duration = 2
channels = 1

frames_per_block = int(samplerate * block_duration)
frames_per_chunk = int(samplerate * chunk_duration)

audio_queue = queue.Queue()
audio_buffer = []

model = WhisperModel(
    "kotoba-tech/kotoba-whisper-v2.0-faster",
    device="cuda",
    compute_type="float16",
)

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def recorder():
    with sd.InputStream(samplerate=samplerate, channels=channels,
                        callback=audio_callback, blocksize=frames_per_block):
        print("Listening... Press Ctrl + C to stop. ", flush=True)
        while True:
            sd.sleep(100)
            
def transcriber():
    global audio_buffer
    while True:
        block = audio_queue.get()
        audio_buffer.append(block)

        total_frames = sum(len(b) for b in audio_buffer)
        if total_frames >= frames_per_chunk:
            audio_data = np.concatenate(audio_buffer)[:frames_per_chunk]
            audio_buffer = []

            audio_data = audio_data.flatten().astype(np.float32)

            segments, _ = model.transcribe(
                audio_data,
                language="ja",
                beam_size=1,
                condition_on_previous_text=False,
            )

            for segment in segments:
                print(segment.text, flush=True)

threading.Thread(target=recorder, daemon=True).start()
transcriber()
