# Voice-to-Text Transcription (Whisper)

A Python CLI tool for transcribing audio and video files to text using [OpenAI Whisper](https://github.com/openai/whisper).

## Features

- Transcribe audio/video files to text using OpenAI Whisper
- Multiple output formats: **TXT**, **SRT** (subtitles), **JSON**
- Support for many audio/video formats: MP3, WAV, M4A, FLAC, OGG, MP4, MKV, and more
- Choose from 5 model sizes: `tiny`, `base`, `small`, `medium`, `large`
- Auto-detect or manually specify language
- CLI-based — easy to use from terminal

## Supported Formats

**Audio:** `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.wma`, `.aac`, `.opus`  
**Video:** `.mp4`, `.mkv`, `.avi`, `.webm`

## Setup

### Prerequisites

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) installed and available in PATH

### Installation

```bash
# Clone the repository
git clone https://github.com/SandeshSatishhNaik/voice-to-text-whisper.git
cd voice-to-text-whisper

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python transcriber.py audio.mp3
```

### Options

| Option | Description |
|--------|-------------|
| `--model`, `-m` | Model size: `tiny`, `base`, `small`, `medium`, `large` (default: `base`) |
| `--language`, `-l` | Language code, e.g., `en`, `es`, `hi` (auto-detect if omitted) |
| `--format`, `-f` | Output format: `txt`, `srt`, `json` (default: `txt`) |
| `--output`, `-o` | Output directory (default: same as input) |
| `--verbose`, `-v` | Show progress info |

### Examples

```bash
# Transcribe with medium model and English language
python transcriber.py audio.wav --model medium --language en

# Generate SRT subtitles
python transcriber.py video.mp4 --format srt --output ./subtitles

# Transcribe with verbose output and JSON format
python transcriber.py meeting.m4a --format json --verbose

# Use the large model for best accuracy
python transcriber.py lecture.mp3 --model large --verbose
```

## Model Sizes

| Model | Parameters | Speed | Accuracy | VRAM |
|-------|-----------|-------|----------|------|
| `tiny` | 39M | Fastest | Lowest | ~1 GB |
| `base` | 74M | Fast | Good | ~1 GB |
| `small` | 244M | Medium | Better | ~2 GB |
| `medium` | 769M | Slow | High | ~5 GB |
| `large` | 1550M | Slowest | Best | ~10 GB |

## License

MIT
