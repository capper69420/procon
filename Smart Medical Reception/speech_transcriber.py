"""faster-whisper transcription worker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Set

import numpy as np
from faster_whisper import WhisperModel


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    language: str


def resolve_device(device: str) -> tuple[str, str]:
    """Return (device, compute_type), auto-detecting CUDA when requested."""
    if device != "auto":
        return device, "float16" if device == "cuda" else "int8"

    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


class Transcriber:
    """Wrap WhisperModel for low-latency chunk transcription."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        compute_type: str = "auto",
        language_whitelist: Optional[Set[str]] = None,
        beam_size: int = 1,
    ) -> None:
        resolved_device, default_compute = resolve_device(device)
        if compute_type == "auto":
            compute_type = default_compute

        self.model = WhisperModel(
            model_name,
            device=resolved_device,
            compute_type=compute_type,
        )
        self.language_whitelist = language_whitelist or {"ja", "en"}
        self.beam_size = beam_size
        self.device = resolved_device
        self.compute_type = compute_type

    def transcribe_file(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> tuple[list[TranscriptSegment], Optional[str]]:
        """
        Transcribe a complete audio file. Returns segments and detected language.
        Segments are empty when language is not whitelisted.
        """
        options = {
            "task": "transcribe",
            "vad_filter": True,
            "vad_parameters": {"min_silence_duration_ms": 300},
            "condition_on_previous_text": False,
            "beam_size": self.beam_size,
        }
        if language:
            options["language"] = language

        segments_gen, info = self.model.transcribe(audio_path, **options)

        detected = info.language
        if detected not in self.language_whitelist:
            for _ in segments_gen:
                pass
            return [], detected

        results: list[TranscriptSegment] = []
        for segment in segments_gen:
            text = segment.text.strip()
            if not text:
                continue
            results.append(
                TranscriptSegment(
                    start=segment.start,
                    end=segment.end,
                    text=text,
                    language=detected,
                )
            )
        return results, detected
    def transcribe_chunk(
        self,
        audio: np.ndarray,
        time_offset: float,
    ) -> tuple[list[TranscriptSegment], Optional[str]]:
        """
        Transcribe one audio chunk. Returns segments and detected language.
        Segments are empty when language is not whitelisted.
        """
        if audio.size == 0:
            return [], None

        segments_gen, info = self.model.transcribe(
            audio,
            task="transcribe",
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            condition_on_previous_text=False,
            beam_size=self.beam_size,
        )

        detected = info.language
        if detected not in self.language_whitelist:
            # Drain generator so the worker finishes cleanly.
            for _ in segments_gen:
                pass
            return [], detected

        results: list[TranscriptSegment] = []
        for segment in segments_gen:
            text = segment.text.strip()
            if not text:
                continue
            results.append(
                TranscriptSegment(
                    start=time_offset + segment.start,
                    end=time_offset + segment.end,
                    text=text,
                    language=detected,
                )
            )
        return results, detected
