"""Chunk buffering, background transcription, and live transcript updates."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from audio_capture import AudioCapture
from transcriber import Transcriber, TranscriptSegment


@dataclass
class AudioChunk:
    audio: np.ndarray
    time_offset: float


def merge_overlap(previous: str, new: str) -> str:
    """Merge neighboring text, dropping duplicated overlap at chunk boundaries."""
    if not previous:
        return new
    if not new:
        return previous

    max_overlap = min(len(previous), len(new), 80)
    for size in range(max_overlap, 0, -1):
        if previous[-size:] == new[:size]:
            return previous + new[size:]
    return previous + (" " if previous and new else "") + new


class StreamingTranscriber:
    """Buffer mic audio, transcribe in a worker thread, emit live segments."""

    def __init__(
        self,
        capture: AudioCapture,
        transcriber: Transcriber,
        chunk_duration: float = 2.0,
        overlap_duration: float = 0.3,
        on_segment: Optional[Callable[[TranscriptSegment], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.capture = capture
        self.transcriber = transcriber
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.on_segment = on_segment or (lambda _seg: None)
        self.on_status = on_status or (lambda _msg: None)

        self._chunk_queue: queue.Queue[Optional[AudioChunk]] = queue.Queue(maxsize=4)
        self._result_queue: queue.Queue[tuple[list[TranscriptSegment], Optional[str]]] = (
            queue.Queue()
        )

        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._worker_thread: Optional[threading.Thread] = None

        self._buffer = np.array([], dtype=np.float32)
        self._timeline = 0.0
        self._last_emitted_end = 0.0
        self._last_text = ""
        self.final_transcript = ""

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return

        self._running = True
        self._buffer = np.array([], dtype=np.float32)
        self._timeline = 0.0
        self._last_emitted_end = 0.0
        self._last_text = ""
        self.final_transcript = ""

        self.capture.start()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._capture_thread.start()
        self._worker_thread.start()
        self.on_status("Listening... (speak in Japanese or English)")

    def stop(self) -> None:
        if not self._running:
            return

        self._running = False
        self._chunk_queue.put(None)

        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=10.0)

        self.capture.stop()
        self._drain_results()
        self.on_status("Stopped.")

    def poll(self) -> None:
        """Process completed transcription results on the main thread."""
        self._drain_results()

    def _drain_results(self) -> None:
        while True:
            try:
                segments, detected = self._result_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_segments(segments, detected)

    def _capture_loop(self) -> None:
        sample_rate = self.capture.sample_rate
        chunk_samples = int(sample_rate * self.chunk_duration)
        overlap_samples = int(sample_rate * self.overlap_duration)

        while self._running:
            block = self.capture.read(timeout=0.5)
            if block is None:
                continue

            self._buffer = np.concatenate((self._buffer, block))
            while len(self._buffer) >= chunk_samples:
                chunk = self._buffer[:chunk_samples]
                time_offset = self._timeline
                self._timeline += (chunk_samples - overlap_samples) / sample_rate
                self._buffer = self._buffer[chunk_samples - overlap_samples :]

                if not self._has_speech(chunk):
                    continue

                try:
                    self._chunk_queue.put(
                        AudioChunk(audio=chunk, time_offset=time_offset),
                        timeout=1.0,
                    )
                except queue.Full:
                    self.on_status("Transcription backlog; dropping chunk.")

        if self._buffer.size > 0 and self._has_speech(self._buffer):
            try:
                self._chunk_queue.put(
                    AudioChunk(audio=self._buffer.copy(), time_offset=self._timeline),
                    timeout=1.0,
                )
            except queue.Full:
                pass

    @staticmethod
    def _has_speech(audio: np.ndarray, threshold: float = 0.008) -> bool:
        return float(np.sqrt(np.mean(audio * audio))) >= threshold

    def _worker_loop(self) -> None:
        while True:
            item = self._chunk_queue.get()
            if item is None:
                break
            segments, detected = self.transcriber.transcribe_chunk(
                item.audio,
                item.time_offset,
            )
            self._result_queue.put((segments, detected))

    def _handle_segments(
        self,
        segments: list[TranscriptSegment],
        detected: Optional[str],
    ) -> None:
        if detected and detected not in self.transcriber.language_whitelist:
            return

        for segment in segments:
            if segment.end <= self._last_emitted_end + 0.05:
                continue

            merged = merge_overlap(self._last_text, segment.text)
            delta = merged[len(self._last_text) :].strip()
            if not delta:
                continue

            emitted = TranscriptSegment(
                start=segment.start,
                end=segment.end,
                text=delta,
                language=segment.language,
            )
            self._last_text = merged
            self._last_emitted_end = max(self._last_emitted_end, segment.end)
            self.final_transcript = merged
            self.on_segment(emitted)
