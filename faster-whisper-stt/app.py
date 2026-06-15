#!/usr/bin/env python3
"""Real-time Japanese/English speech-to-text using faster-whisper."""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

from audio_capture import AudioCapture
from streamer import StreamingTranscriber
from transcriber import Transcriber, TranscriptSegment

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "base"  # e.g. "distil-large-v3", "small", "base"
DEVICE = "auto"  # "auto", "cuda", or "cpu"
COMPUTE_TYPE = "auto"  # "auto", "float16", "int8", "int8_float16"
CHUNK_DURATION_SECONDS = 2.0
OVERLAP_DURATION_SECONDS = 0.3
SAMPLE_RATE = 16000
LANGUAGE_WHITELIST = {"ja", "en"}
BEAM_SIZE = 1


def format_segment(segment: TranscriptSegment) -> str:
    lang_tag = segment.language.upper()
    return f"[{segment.start:6.2f}s -> {segment.end:6.2f}s] ({lang_tag}) {segment.text}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time speech-to-text (Japanese and English only)."
    )
    parser.add_argument("--model", default=MODEL_NAME, help="Whisper model name")
    parser.add_argument(
        "--device",
        default=DEVICE,
        choices=["auto", "cuda", "cpu"],
        help="Inference device",
    )
    parser.add_argument(
        "--compute-type",
        default=COMPUTE_TYPE,
        help="Compute type (auto, float16, int8, int8_float16)",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=CHUNK_DURATION_SECONDS,
        help="Audio chunk length in seconds",
    )
    parser.add_argument(
        "--overlap-seconds",
        type=float,
        default=OVERLAP_DURATION_SECONDS,
        help="Overlap between consecutive chunks",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=SAMPLE_RATE,
        help="Microphone sample rate (Hz)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=BEAM_SIZE,
        help="Beam size (1 = fastest)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Optional path to save the final transcript",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List audio input devices and exit",
    )
    return parser.parse_args()


def list_input_devices() -> None:
    import sounddevice as sd

    print(sd.query_devices())


def run(args: argparse.Namespace) -> int:
    print(f"Loading model '{args.model}'...", flush=True)
    transcriber = Transcriber(
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language_whitelist=set(LANGUAGE_WHITELIST),
        beam_size=args.beam_size,
    )
    print(
        f"Ready on {transcriber.device} ({transcriber.compute_type}). "
        f"Languages: {', '.join(sorted(LANGUAGE_WHITELIST))}",
        flush=True,
    )

    capture = AudioCapture(sample_rate=args.sample_rate)
    streamer = StreamingTranscriber(
        capture=capture,
        transcriber=transcriber,
        chunk_duration=args.chunk_seconds,
        overlap_duration=args.overlap_seconds,
        on_segment=lambda seg: print(format_segment(seg), flush=True),
        on_status=lambda msg: print(f"\n>> {msg}", flush=True),
    )

    print("\nCommands:", flush=True)
    print("  Enter  -> stop transcription", flush=True)
    print("  Ctrl+C -> quit immediately\n", flush=True)

    stop_event = threading.Event()

    def wait_for_enter() -> None:
        try:
            input()
        except EOFError:
            pass
        stop_event.set()

    threading.Thread(target=wait_for_enter, daemon=True).start()
    streamer.start()

    try:
        while streamer.running and not stop_event.is_set():
            streamer.poll()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)
    finally:
        streamer.stop()
        streamer.poll()

    if streamer.final_transcript:
        print("\n--- Final transcript ---", flush=True)
        print(streamer.final_transcript, flush=True)

        if args.output:
            args.output.write_text(streamer.final_transcript + "\n", encoding="utf-8")
            print(f"\nSaved to {args.output}", flush=True)

    return 0


def main() -> int:
    args = parse_args()
    if args.list_devices:
        list_input_devices()
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
