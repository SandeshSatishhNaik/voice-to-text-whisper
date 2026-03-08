# Voice-to-Text Transcription (Whisper)

A Python CLI tool for **real-time voice-to-text** and file-based transcription using [OpenAI Whisper](https://github.com/openai/whisper).

## Features

- **Real-time microphone transcription** — speak and see text appear live
- Transcribe audio/video files to text using OpenAI Whisper
- Multiple output formats: **TXT**, **SRT** (subtitles), **JSON**
- Support for many audio/video formats: MP3, WAV, M4A, FLAC, OGG, MP4, MKV, and more
- Choose from 5 model sizes: `tiny`, `base`, `small`, `medium`, `large`
- Auto-detect or manually specify language
- Automatic silence detection — skips quiet segments
- Saves full real-time transcript to file on exit
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

### Real-Time Voice-to-Text (Microphone)

```bash
# Start real-time transcription (speak into your mic)
python realtime_transcriber.py

# Use a faster tiny model with English
python realtime_transcriber.py --model tiny --language en

# Shorter chunks for faster response (3 seconds)
python realtime_transcriber.py --chunk 3 --language en

# List available microphones
python realtime_transcriber.py --list-devices

# Use a specific microphone device
python realtime_transcriber.py --device 1
```

Press **Ctrl+C** to stop. The full transcript is saved to `transcript_realtime.txt`.

### Real-Time Options

| Option | Description |
|--------|-------------|
| `--model`, `-m` | Model size: `tiny`, `base`, `small`, `medium`, `large` (default: `base`) |
| `--language`, `-l` | Language code (e.g., `en`). Auto-detect if omitted |
| `--chunk`, `-c` | Chunk duration in seconds (default: 5). Lower = faster |
| `--device`, `-d` | Audio input device index |
| `--threshold`, `-t` | Energy threshold to filter silence (default: 0.01) |
| `--list-devices` | List available audio input devices |

---

### File-Based Transcription

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
