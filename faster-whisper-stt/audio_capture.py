"""Continuous microphone capture for streaming transcription."""

from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd


class AudioCapture:
    """Capture mono float32 audio from the default microphone."""

    def __init__(self, sample_rate: int = 16000, block_duration: float = 0.1) -> None:
        self.sample_rate = sample_rate
        self.block_size = max(1, int(sample_rate * block_duration))
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            print(f"[audio] {status}", flush=True)
        self._queue.put(indata[:, 0].copy())

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.block_size,
            callback=self._callback,
        )
        self._stream.start()

    def read(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
