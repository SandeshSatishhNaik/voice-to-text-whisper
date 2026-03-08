"""
Voice-to-Text Transcription using OpenAI Whisper
==================================================
A CLI tool to transcribe audio/video files to text using Whisper.
Supports multiple audio formats and Whisper model sizes.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import whisper


SUPPORTED_FORMATS = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma",
    ".aac", ".opus", ".mp4", ".mkv", ".avi", ".webm"
}

MODEL_SIZES = ["tiny", "base", "small", "medium", "large"]


def validate_file(filepath: str) -> Path:
    """Validate that the input file exists and has a supported format."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format '{path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )
    return path


def transcribe_audio(
    filepath: str,
    model_size: str = "base",
    language: str | None = None,
    output_format: str = "txt",
    output_dir: str | None = None,
    verbose: bool = False,
) -> dict:
    """
    Transcribe an audio/video file to text.

    Args:
        filepath: Path to the audio/video file.
        model_size: Whisper model size (tiny, base, small, medium, large).
        language: Language code (e.g., 'en', 'es'). None for auto-detect.
        output_format: Output format - 'txt', 'srt', or 'json'.
        output_dir: Directory to save output. Defaults to same as input file.
        verbose: Print progress information.

    Returns:
        dict with keys: text, segments, language
    """
    audio_path = validate_file(filepath)

    if model_size not in MODEL_SIZES:
        raise ValueError(f"Invalid model size '{model_size}'. Choose from: {MODEL_SIZES}")

    if verbose:
        print(f"Loading Whisper '{model_size}' model...")

    model = whisper.load_model(model_size)

    if verbose:
        print(f"Transcribing: {audio_path.name}")

    start_time = time.time()

    transcribe_options = {}
    if language:
        transcribe_options["language"] = language

    result = model.transcribe(str(audio_path), **transcribe_options)

    elapsed = time.time() - start_time

    if verbose:
        detected_lang = result.get("language", "unknown")
        print(f"Detected language: {detected_lang}")
        print(f"Transcription completed in {elapsed:.1f}s")

    # Save output
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = audio_path.parent

    stem = audio_path.stem

    if output_format == "txt":
        out_file = out_dir / f"{stem}.txt"
        out_file.write_text(result["text"].strip(), encoding="utf-8")
    elif output_format == "srt":
        out_file = out_dir / f"{stem}.srt"
        srt_content = generate_srt(result["segments"])
        out_file.write_text(srt_content, encoding="utf-8")
    elif output_format == "json":
        import json
        out_file = out_dir / f"{stem}.json"
        output_data = {
            "text": result["text"].strip(),
            "language": result.get("language", "unknown"),
            "segments": [
                {
                    "id": seg["id"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip(),
                }
                for seg in result["segments"]
            ],
        }
        out_file.write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    if verbose:
        print(f"Output saved: {out_file}")

    return result


def generate_srt(segments: list) -> str:
    """Generate SRT subtitle content from Whisper segments."""
    srt_lines = []
    for i, seg in enumerate(segments, start=1):
        start = format_timestamp_srt(seg["start"])
        end = format_timestamp_srt(seg["end"])
        text = seg["text"].strip()
        srt_lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(srt_lines)


def format_timestamp_srt(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def main():
    parser = argparse.ArgumentParser(
        description="Voice-to-Text Transcription using OpenAI Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transcriber.py audio.mp3
  python transcriber.py audio.wav --model medium --language en
  python transcriber.py video.mp4 --format srt --output ./subtitles
  python transcriber.py meeting.m4a --format json --verbose
        """,
    )
    parser.add_argument("file", help="Path to audio/video file to transcribe")
    parser.add_argument(
        "--model", "-m",
        choices=MODEL_SIZES,
        default="base",
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--language", "-l",
        default=None,
        help="Language code (e.g., 'en', 'es'). Auto-detect if not specified.",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["txt", "srt", "json"],
        default="txt",
        help="Output format (default: txt)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory (default: same as input file)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress information",
    )

    args = parser.parse_args()

    try:
        result = transcribe_audio(
            filepath=args.file,
            model_size=args.model,
            language=args.language,
            output_format=args.format,
            output_dir=args.output,
            verbose=args.verbose,
        )
        if not args.verbose:
            print(result["text"].strip())
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Transcription failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
