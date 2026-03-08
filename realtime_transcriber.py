"""
Real-Time Voice-to-Text Transcription using OpenAI Whisper
============================================================
Captures audio from your microphone in real-time and transcribes
it on-the-fly using Whisper. Press Ctrl+C to stop.
"""

import argparse
import sys
import threading
import time
import queue

import numpy as np
import sounddevice as sd
import whisper


# ── Globals ──────────────────────────────────────────────────────
SAMPLE_RATE = 16000  # Whisper expects 16kHz mono audio


def list_audio_devices():
    """Print all available audio input devices."""
    print("\n Available Audio Input Devices:\n")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            marker = " <-- default" if i == sd.default.device[0] else ""
            print(f"  [{i}] {dev['name']}  (inputs: {dev['max_input_channels']}){marker}")
    print()


def realtime_transcribe(
    model_size: str = "base",
    language: str | None = None,
    chunk_duration: float = 5.0,
    device_index: int | None = None,
    energy_threshold: float = 0.01,
):
    """
    Continuously capture microphone audio and transcribe in real-time.

    Args:
        model_size: Whisper model size.
        language: Language code or None for auto-detect.
        chunk_duration: Seconds of audio per transcription chunk.
        device_index: Audio input device index (None = default mic).
        energy_threshold: Minimum RMS energy to consider as speech.
    """
    # ── Load model ───────────────────────────────────────────────
    print(f"Loading Whisper '{model_size}' model...")
    model = whisper.load_model(model_size)
    print("Model loaded.\n")

    # ── Audio state ──────────────────────────────────────────────
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    chunk_samples = int(SAMPLE_RATE * chunk_duration)

    def audio_callback(indata, frames, time_info, status):
        """Called by sounddevice for each audio block."""
        if status:
            print(f"  [audio warning] {status}", file=sys.stderr)
        audio_queue.put(indata[:, 0].copy())  # mono

    # ── Verify device ────────────────────────────────────────────
    try:
        dev_info = sd.query_devices(device_index, kind="input")
        print(f"Microphone: {dev_info['name']}")
    except Exception as e:
        print(f"Error selecting audio device: {e}", file=sys.stderr)
        list_audio_devices()
        sys.exit(1)

    print(f"Chunk duration: {chunk_duration}s | Language: {language or 'auto-detect'}")
    print("─" * 50)
    print("Speak now... (press Ctrl+C to stop)\n")

    # ── Start recording stream ───────────────────────────────────
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=int(SAMPLE_RATE * 0.5),  # 500ms blocks
        device=device_index,
        callback=audio_callback,
    )

    transcript_lines: list[str] = []

    try:
        with stream:
            buffer = np.empty(0, dtype=np.float32)

            while True:
                # Drain the queue into our buffer
                while not audio_queue.empty():
                    buffer = np.concatenate([buffer, audio_queue.get()])

                # Wait until we have enough audio for a chunk
                if len(buffer) < chunk_samples:
                    time.sleep(0.1)
                    continue

                # Grab exactly one chunk
                chunk = buffer[:chunk_samples]
                buffer = buffer[chunk_samples:]

                # Skip silence
                rms = np.sqrt(np.mean(chunk ** 2))
                if rms < energy_threshold:
                    continue

                # Transcribe
                transcribe_opts = {"fp16": False}
                if language:
                    transcribe_opts["language"] = language

                result = model.transcribe(
                    whisper.pad_or_trim(chunk.astype(np.float32)),
                    **transcribe_opts,
                )

                text = result["text"].strip()
                if text and text.lower() not in (
                    "you", "thank you.", "thanks for watching!",
                    "...", "", "thank you",
                ):
                    # Whisper sometimes hallucinates these on silence
                    timestamp = time.strftime("%H:%M:%S")
                    line = f"[{timestamp}] {text}"
                    print(line)
                    transcript_lines.append(line)

    except KeyboardInterrupt:
        print("\n\n─── Transcription stopped ───")

    # ── Save full transcript ─────────────────────────────────────
    if transcript_lines:
        out_file = "transcript_realtime.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("\n".join(transcript_lines) + "\n")
        print(f"\nFull transcript saved to: {out_file}")
    else:
        print("\nNo speech was detected.")


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Voice-to-Text using OpenAI Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python realtime_transcriber.py
  python realtime_transcriber.py --model small --language en
  python realtime_transcriber.py --chunk 3 --device 1
  python realtime_transcriber.py --list-devices
        """,
    )
    parser.add_argument(
        "--model", "-m",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--language", "-l",
        default=None,
        help="Language code (e.g., 'en'). Auto-detect if not set.",
    )
    parser.add_argument(
        "--chunk", "-c",
        type=float,
        default=5.0,
        help="Chunk duration in seconds (default: 5). Lower = faster but less accurate.",
    )
    parser.add_argument(
        "--device", "-d",
        type=int,
        default=None,
        help="Audio input device index. Use --list-devices to see options.",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.01,
        help="Energy threshold to filter silence (default: 0.01).",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit.",
    )

    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    realtime_transcribe(
        model_size=args.model,
        language=args.language,
        chunk_duration=args.chunk,
        device_index=args.device,
        energy_threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
