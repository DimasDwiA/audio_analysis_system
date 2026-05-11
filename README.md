# 🎙 Audio Analysis System

An **agentic AI system** for deep audio quality analysis powered by **FFmpeg** and **Google Gemini**.
Supports single-file analysis, batch processing, live microphone recording, and exposes a full REST API for team collaboration.

---

## Table of Contents

- [Architecture](#architecture)
- [Tech Stack & Key Decisions](#tech-stack--key-decisions)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [How to Run the Agent](#how-to-run-the-agent)
- [Full CLI](#full-cli)
- [REST API Usage](#rest-api-usage)
- [Example Output](#example-output)
- [Configuration](#configuration)
- [How the Agent Works](#how-the-agent-works)
- [Extending the System](#extending-the-system)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Entry Points                                │
│  ┌──────────────┐  ┌───────────────┐  ┌────────────┐  ┌──────────┐   │
│  │ run_agent.py │  │  main.py CLI  │  │ FastAPI    │  │  demo.py │   │
│  │ (interactive)│  │  (Click+Rich) │  │ REST API   │  │ (no key) │   │
│  └──────┬───────┘  └──────┬────────┘  └─────┬──────┘  └────┬─────┘   │
└─────────┼─────────────────┼────────────────┼───────────────┼─────────┘
          └─────────────────┴────────────────┘               │
                            │                                │
          ┌─────────────────▼────────────────┐               │
          │         AudioAnalysisAgent       │ ◄── Agentic   │  
          │  (Gemini function-calling loop)  │  Orchestrator │  
          └──────┬──────┬──────┬────────┬────┘               │
                 │      │      │        │                    │
       ┌─────────▼┐  ┌──▼──┐  ┌▼────┐  ┌▼──────────┐         │
       │ Metadata │  │Qual.│  │Rules│  │  Report   │         │
       │ Extractor│  │Anal.│  │Eng. │  │ Generator │◄────────┘
       └────┬─────┘  └──┬──┘  └──┬──┘  └─────┬─────┘
            │           │        │           │
       ┌────▼───────────▼────────▼─────┐     │
       │         FFmpegRunner          │   ┌─▼─────────────┐
       │  (ffprobe + ffmpeg filters)   │   │ GeminiClient  │
       └───────────────────────────────┘   │ (google-genai)│
                                           └───────────────┘
```

### Data Flow

```
Audio File
    │
    ▼
[1] ffprobe JSON         → duration, bitrate, sample rate, channels, codec
[2] silencedetect filter → silence segments with start/end/duration
[3] volumedetect filter  → mean dB, peak dB, histogram
[4] astats filter        → peak level, clipping detection
[5] astats filter        → RMS, crest factor, noise floor, dynamic range
    │
    ▼
[6] Rules Engine         → deterministic issue list + quality score (0–100) + grade A–F
    │
    ▼
[7] Gemini LLM           → executive summary + recommended actions
    │
    ▼
[8] JSON Report          → structured output to disk / API / CLI
```

---

## Tech Stack & Key Decisions

| Component | Technology | Decision Rationale |
|-----------|-----------|-------------------|
| Audio Analysis | **FFmpeg / ffprobe** | Industry standard, supports all formats, rich filter ecosystem |
| LLM | **Google Gemini 1.5 Flash** | Free tier with generous quotas; function-calling support |
| LLM SDK | **google-genai** | Latest official SDK (replaces the deprecated google-generativeai) |
| Agentic Loop | **Gemini Function Calling** | Agent decides which tools to invoke, enabling dynamic workflows |
| Hybrid Analysis | **Rules + LLM** | Rules for precision/reproducibility; LLM for human-readable prose |
| API | **FastAPI** | Async, auto-docs (Swagger/ReDoc), type-safe |
| CLI | **Click + Rich** | Beautiful terminal output; scriptable |
| Recording | **sounddevice + soundfile** | Cross-platform, NumPy-compatible |
| Validation | **Pydantic v2** | Strict schema enforcement for API I/O |

### Why Hybrid (Rules + LLM)?

Pure LLM analysis risks hallucinating values.  Pure rules-based analysis is brittle and produces robotic output.  My approach:

- **Rules Engine** (deterministic): Detects and classifies issues with exact dB values and timestamps.  Never wrong about the numbers.
- **LLM** (Gemini): Translates those exact findings into professional, human-readable summaries and prioritised recommendations.

---

## Project Structure

```
audio_analysis_system/
├── run_agent.py                 # ★ Main entry point – run the agent from here
├── main.py                      # Full CLI (Click + Rich)
├── demo.py                      # Demo without an API key (generate synthetic WAV)
├── config.py                    # Centralised settings (pydantic-settings)
├── requirements.txt
├── .env.example                 # Environment template
│
├── agent/
│   ├── audio_agent.py           # ★ Agentic orchestrator (Gemini loop)
│   └── tool_definitions.py      # Gemini FunctionDeclaration schemas
│
├── analyzers/
│   ├── ffmpeg_runner.py         # Low-level ffmpeg/ffprobe subprocess wrapper
│   ├── metadata_extractor.py    # ffprobe → structured metadata
│   ├── quality_analyzer.py      # silencedetect / volumedetect / astats
│   └── rules_engine.py          # Deterministic issue detection + scoring
│
├── llm/
│   ├── gemini_client.py         # Gemini API wrapper (completions + chat)
│   └── prompts.py               # All prompt templates
│
├── processor/
│   └── batch_processor.py       # Multi-file orchestration
│
├── recorder/
│   └── live_recorder.py         # Microphone capture (sounddevice)
│
├── reports/
│   └── report_generator.py      # Final JSON assembly + LLM insight call
│
├── api/
│   ├── app.py                   # FastAPI app factory
│   ├── routes/
│   │   ├── analysis.py          # POST /analyze/file, /upload, /record
│   │   └── batch.py             # POST /batch/analyze
│   └── schemas/
│       └── models.py            # Pydantic request/response models
│
├── interface/
│   └── cli.py                   # Click + Rich CLI commands
│
└── utils/
    ├── logger.py                # Rich + file logging
    └── helpers.py               # Shared utilities
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- FFmpeg installed and in PATH

```bash
# Ubuntu / Debian
sudo apt-get install ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Windows (Chocolatey)
choco install ffmpeg
```

### Installation

```bash
# 1. Clone / download the project
cd audio_analysis_system

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env and set your GEMINI_API_KEY
# Get a free key at: https://aistudio.google.com
```

### Verify Installation

```bash
python main.py --version
python main.py devices          # List microphone inputs

# Jalankan demo (tidak butuh API key)
python demo.py
```

---

## How to Run the Agent

> **`run_agent.py`** is the easiest and most comprehensive way to run the system.
> It supports 6 operating modes to suit your needs.

### Mode 1 – Automatic Demo (no file required, no API key required)

The fastest way to try out the system. It will automatically generate a synthetic WAV file:

```bash
python run_agent.py
```

### Mode 2 – Analyze Your Files (without an API key)

Runs only FFmpeg + the Rules Engine. No internet connection required:

```bash
python run_agent.py recording.wav
python run_agent.py /path/to/podcast.mp3 --no-llm
```

### Mode 3 – Full Pipeline + AI Insights (with Gemini API key)

After setting `GEMINI_API_KEY` in `.env`, the system automatically enables LLM insights:

```bash
# Make sure .env is set: GEMINI_API_KEY=AIzaSy...
python run_agent.py recording.wav
```

The output will include:
- Full FFmpeg analysis (metadata, silence, volume, clipping, statistics)
- Rules-based quality score (0–100) and grade (A–F)
- **AI Executive Summary** from Gemini
- **Prioritized action recommendations**
- **Overall verdict**

### Mode 4 – Full Agentic Loop (Gemini leads everything)

In this mode, Gemini decides on its own which tools to call, in what order, and when to stop:

```bash
# Requires GEMINI_API_KEY
python run_agent.py recording.wav --agent
```

The log will display every function call dynamically decided by the model:
```
Iteration 1/20 – model issued 1 function call(s).
  ▶ Tool: extract_audio_metadata         args: ['file_path']
Iteration 2/20 – model issued 1 function call(s).
  ▶ Tool: detect_silence_segments        args: ['file_path']
...
```

### Mode 5 – Batch Entire Folder

```bash
python run_agent.py /records/folder/ --batch

# Example with a specific folder
python run_agent.py /home/user/podcast_episodes/ --batch
```

The output consists of a summary table for each file plus Gemini's batch verdict.

### Mode 6 – REST API for Teams

```bash
python main.py serve
# → Swagger UI: http://localhost:8000/docs
```

### Mode Table

| Mode | Command | API Key | Who is in charge |
|------|---------|---------|---------------------|
| Automatic Demo | `python run_agent.py` | ❌ | Python Code |
| File without LLM | `python run_agent.py file.wav --no-llm` | ❌ | Python Code |
| File + AI Insights | `python run_agent.py file.wav` | ✅ | Code + Gemini (insights) |
| **Full Agent** | `python run_agent.py file.wav --agent` | ✅ | **Full Gemini** |
| Batch folder | `python run_agent.py /folder/ --batch` | ✅ optional | Python code |
| REST API | `python main.py serve` | ✅ optional | HTTP clients / team |

---

## CLI Usage

`main.py` provides a more comprehensive CLI with dedicated subcommands.

### Analyse a Single File

```bash
python main.py analyze path/to/audio.wav
```

```bash
# Save report to output/ and display in terminal
python main.py analyze recording.mp3 --save

# Output raw JSON only (pipe-friendly)
python main.py analyze recording.wav --json-only | jq '.audio_quality'
```

### Batch Analysis (Directory)

```bash
# Analyse all audio files in a directory
python main.py batch /path/to/recordings/

# Custom file patterns
python main.py batch /recordings/ --pattern "*.wav,*.mp3"

# Skip LLM batch summary (faster)
python main.py batch /recordings/ --no-summary
```

### Live Recording + Analysis

```bash
# Record 60 seconds from default mic then analyse
python main.py record --duration 60

# Use a specific device
python main.py devices                    # List devices first
python main.py record --duration 30 --device 2
```

### Start the API Server

```bash
python main.py serve
# → http://localhost:8000/docs  (Swagger UI)

# Dev mode with auto-reload
python main.py serve --reload --port 8000
```

---

## REST API Usage

### API Base URL

```
http://localhost:8000
```

### Health Check

```bash
curl http://localhost:8000/health
```

### Analyse a File (server-side path)

```bash
curl -X POST http://localhost:8000/analyze/file \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/absolute/path/to/audio.wav", "save_report": true}'
```

### Upload and Analyse

```bash
curl -X POST http://localhost:8000/analyze/upload \
  -F "file=@/local/path/to/audio.wav"
```

### Batch Analysis

```bash
curl -X POST http://localhost:8000/batch/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "file_paths": [
      "/recordings/session_001.wav",
      "/recordings/session_002.wav",
      "/recordings/session_003.mp3"
    ],
    "generate_batch_summary": true
  }'
```

### Record + Analyse (server microphone)

```bash
curl -X POST http://localhost:8000/analyze/record \
  -H "Content-Type: application/json" \
  -d '{"duration_seconds": 30}'
```

### Interactive Docs

Visit `http://localhost:8000/docs` for the full Swagger UI with request/response schemas.

---

## Example Output

### Single File JSON Report

```json
{
  "report_id": "18f4f3f9-fa8b-4e38-ba4f-881f2c802684",
  "generated_at": "2026-05-11T02:30:24.369421Z",
  "file_name": "audio_0.wav",
  "file_path": "C:\\Users\\Dimas Dwi Armaisya\\Downloads\\audio_0.wav",
  "duration_seconds": 4.464,
  "duration_hms": "00:00:04",
  "file_size_human": "418.0 KB",
  "format": "WAV / WAVE (Waveform Audio)",
  "audio_quality": {
    "silence_ratio": 0.0,
    "total_silence_seconds": 0,
    "silence_segment_count": 0,
    "clipping_detected": false,
    "avg_volume_db": -17.7,
    "peak_volume_db": -1.5,
    "noise_floor_db": null,
    "dynamic_range_db": 94.838782,
    "rms_level_db": -17.658129,
    "dc_offset": -0.000201,
    "sample_rate_hz": 48000,
    "channels": 1,
    "channel_layout": "unknown",
    "codec": "pcm_s16le",
    "bitrate_kbps": 768.1,
    "quality_score": 100,
    "quality_grade": "A"
  },
  "silence_segments": [],
  "issues": [],
  "rules_analysis": {
    "quality_score": 100,
    "quality_grade": "A",
    "summary_flags": [],
    "detailed_issues": []
  },
  "raw_metrics": {
    "metadata": {
      "file_name": "audio_0.wav",
      "file_path": "C:\\Users\\Dimas Dwi Armaisya\\Downloads\\audio_0.wav",
      "file_size_bytes": 428588,
      "file_size_human": "418.0 KB",
      "format_name": "WAV / WAVE (Waveform Audio)",
      "format_tags": {},
      "duration_seconds": 4.464,
      "duration_hms": "00:00:04",
      "overall_bitrate_bps": 768078,
      "overall_bitrate_kbps": 768.1,
      "codec_name": "pcm_s16le",
      "codec_long_name": "PCM signed 16-bit little-endian",
      "sample_rate_hz": 48000,
      "channels": 1,
      "channel_layout": "unknown",
      "bit_depth": 16,
      "stream_bitrate_bps": 768000
    },
    "silence_analysis": {
      "threshold_db": -30.0,
      "min_duration_s": 2.0,
      "segments": [],
      "total_silence_seconds": 0,
      "segment_count": 0
    },
    "volume_analysis": {
      "mean_volume_db": -17.7,
      "max_volume_db": -1.5,
      "histogram_db": [
        {
          "db": -1,
          "count": 8
        },
        {
          "db": -2,
          "count": 17
        },
        {
          "db": -3,
          "count": 236
        }
      ],
      "is_too_quiet": false,
      "low_volume_threshold_db": -40.0
    },
    "clipping_analysis": {
      "clipping_detected": false,
      "peak_level_db": null,
      "rms_level_db": null,
      "peak_count": 2560,
      "max_difference": 0.0,
      "clipping_threshold_db": -0.1
    },
    "audio_statistics": {
      "rms_level_db": -17.658129,
      "rms_peak_db": -9.190965,
      "rms_trough_db": -53.351974,
      "crest_factor_db": 6.432486,
      "flat_factor": 0.0,
      "dc_offset": -0.000201,
      "noise_floor_db": null,
      "dynamic_range_db": 94.838782,
      "is_noisy": false,
      "noise_floor_threshold_db": -50.0
    }
  },
  "llm_insights": {
    "executive_summary": "The audio file is in excellent technical condition with no detected clipping or silence. It maintains a healthy peak level of -1.5 dB, providing sufficient headroom while utilizing the available dynamic range effectively. The recording is technically sound and ready for production use without further processing.",
    "recommended_actions": [
      "Proceed with standard production workflows as no technical defects were identified.",
      "Verify the content's subjective quality through a listening test to ensure the performance meets creative requirements."
    ],
    "overall_verdict": "The audio file meets all professional standards and is approved for immediate use."
  }
}
```

### CLI Terminal Output

```
╭───────────────────────── Audio Analysis Report ─────────────────────────╮
│ audio_0.wav                                                             │
│ Duration: 00:00:04  Size: 418.0 KB  Format: WAV / WAVE (Waveform Audio) │
╰─────────────────────────────────────────────────────────────────────────╯
          Audio Quality Metrics           
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Metric             ┃ Value             ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ Quality Score      │ 100/100 (Grade A) │
│ Silence Ratio      │ 0.0%              │
│ Silence Segments   │ 0                 │
│ Clipping Detected  │ No                │
│ Avg Volume (dB)    │ -17.7             │
│ Peak Volume (dB)   │ -1.5              │
│ Noise Floor (dB)   │ N/A               │
│ Dynamic Range (dB) │ 94.8              │
│ Sample Rate        │ 48000 Hz          │
│ Channels           │ 1                 │
│ Codec              │ pcm_s16le         │
│ Bitrate            │ 768.1 kbps        │
└────────────────────┴───────────────────┘
╭─────────────────────────────────────────── Issues ────────────────────────────────────────────╮
│ ✓ No issues detected.                                                                         │
╰───────────────────────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────────── AI Executive Summary ─────────────────────────────────────╮
│ The audio file is in excellent technical condition with no detected clipping or silence. It   │
│ maintains a healthy peak level of -1.5 dB, providing sufficient headroom while utilizing the  │
│ available dynamic range effectively. The recording is technically sound and ready for         │
│ production use without further processing.                                                    │
╰───────────────────────────────────────────────────────────────────────────────────────────────╯
╭───────────────────────────────────── Recommended Actions ─────────────────────────────────────╮
│   1. Proceed with standard production workflows as no technical defects were identified.      │
│   2. Verify the content's subjective quality through a listening test to ensure the           │
│ performance meets creative requirements.                                                      │
╰───────────────────────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────────────────────── Overall Verdict ───────────────────────────────────────╮
│ The audio file meets all professional standards and is approved for immediate use.            │
╰───────────────────────────────────────────────────────────────────────────────────────────────╯

Report saved → output\audio_0_report.json
```

### Output Terminal (run_agent.py)

```
╔══════════════════════════════════════════════════════════════════╗
║      🎙  Audio Analysis Agent – Interactive Runner               ║
║      FFmpeg + Rules Engine + Gemini AI (hybrid pipeline)         ║
╚══════════════════════════════════════════════════════════════════╝

✅ Detected GEMINI_API_KEY  – mode LLM active

🤖 AGENTIC MODE – Gemini will determine which tool to call dynamically.

▶ File: audio_0.wav
───────────────────────────────────────────────────────────────────────────────────────────────
INFO     2026-05-11 11:46:07,670 | llm.gemini_client | INFO | Gemini client initialised (model:
         gemini-3.1-flash-lite)                                                                
INFO     2026-05-11 11:46:07,844 | agent.audio_agent | INFO |                                  
         ════════════════════════════════════════════════════════════                          
INFO     2026-05-11 11:46:07,846 | agent.audio_agent | INFO | AudioAnalysisAgent starting      
         analysis: C:\Users\Dimas Dwi Armaisya\Downloads\audio_0.wav                           
INFO     2026-05-11 11:46:07,847 | agent.audio_agent | INFO |                                  
         ════════════════════════════════════════════════════════════                                                                      
INFO     2026-05-11 11:46:48,129 | agent.audio_agent | INFO |   ▶ Tool: generate_final_report  
         args: ['file_path', 'all_analysis_data_json']                                         
INFO     2026-05-11 11:46:48,131 | reports.report_generator | INFO | Generating LLM insights   
         via Gemini…                                                                           
INFO     2026-05-11 11:46:48,133 | google_genai.models | INFO | AFC is enabled with max remote 
         calls: 10.                                                                            
INFO     2026-05-11 11:47:01,201 | httpx | INFO | HTTP Request: POST                           
         https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generate
         Content "HTTP/1.1 200 OK"                                                             
INFO     2026-05-11 11:47:01,207 | reports.report_generator | INFO | Report assembled for      
         'audio_0.wav' – score=100, grade=A, issues=0                                          
INFO     2026-05-11 11:47:05,577 | httpx | INFO | HTTP Request: POST                           
         https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generate
         Content "HTTP/1.1 200 OK"                                                             
INFO     2026-05-11 11:47:05,579 | agent.audio_agent | INFO | Agent completed in 8 iteration(s)
         – no more function calls.                                                             
INFO     2026-05-11 11:47:05,581 | agent.audio_agent | INFO | Final report extracted from agent
         state.                                                                                
╭─────────────────────────────────── Hasil Analisis Agent ────────────────────────────────────╮
│ audio_0.wav                                                                                 │
│ Duration: 00:00:04  Size: 418.0 KB                                                          │
╰─────────────────────────────────────────────────────────────────────────────────────────────╯
  Skor   : 100/100 (Grade A)
  Issues : 0
───────────────────────────────────────────────────────────────────────────────────────────────
╭────────────────────────────────── 🤖 AI Executive Summary ──────────────────────────────────╮
│ The audio file is in excellent condition, meeting professional standards for clarity and    │
│ technical integrity. It features a healthy peak level of -1.5 dB, ensuring maximum headroom │
│ without clipping. With a dynamic range of 94.8 dB and no detected silence or noise issues,  │
│ the recording is technically pristine.                                                      │
╰─────────────────────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────────── Recommended Actions ────────────────────────────────────╮
│   1. No immediate technical corrections are required as the file passed all quality checks. │
│   2. Proceed with standard production workflows, such as normalization or final mastering,  │
│ if the target delivery specifications require a different average loudness level.           │
╰─────────────────────────────────────────────────────────────────────────────────────────────╯
╭────────────────────────────────────────── Verdict ──────────────────────────────────────────╮
│ The audio file is of high quality and ready for immediate use.                              │
╰─────────────────────────────────────────────────────────────────────────────────────────────╯

💾 Report saved → output\audio_0_report.json
✅ Analysis done.
```

---

## Configuration

All settings are loaded from `.env`.  See `.env.example` for the full list.

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | *(required)* | Free at [aistudio.google.com](https://aistudio.google.com) |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model to use |
| `FFMPEG_PATH` | `ffmpeg` | Path to ffmpeg binary |
| `FFPROBE_PATH` | `ffprobe` | Path to ffprobe binary |
| `SILENCE_THRESHOLD_DB` | `-30` | dB below which audio is considered silent |
| `SILENCE_MIN_DURATION` | `2.0` | Minimum silence segment length (s) |
| `LOW_VOLUME_THRESHOLD_DB` | `-40` | Mean volume below this triggers a warning |
| `HIGH_SILENCE_RATIO_THRESHOLD` | `0.2` | Silence ratio above this (20%) triggers a warning |
| `NOISE_FLOOR_THRESHOLD_DB` | `-50` | Noise floor above this is flagged |
| `OUTPUT_DIR` | `./output` | Where JSON reports are saved |
| `TEMP_DIR` | `./temp` | Temporary files (uploads, recordings) |
| `API_HOST` | `0.0.0.0` | Bind address API server |
| `API_PORT` | `8000` | REST API listen port |
| `SAMPLE_RATE` | `44100` | Live recording sample rate (Hz) |
| `CHANNELS` | `1` | Live recording channels (1=mono) |
| `RECORD_CHUNK_SECONDS` | `30` | Durasi chunk rekaman streaming |

---

## How the Agent Works

The `AudioAnalysisAgent` drives an **agentic loop** using Gemini's function-calling capability:

```
User Request
     │
     ▼
Gemini receives task + 7 tool schemas
     │
     ▼  ← Agentic loop (max 20 iterations) ──────────────────┐
Model decides which tool to call                              │
     │                                                        │
     ▼                                                        │
Tool executes (FFmpeg command / Rules Engine)                 │
     │                                                        │
     ▼                                                        │
Result returned to model                                      │
     │                                                        │
     └──────────────── Model issues next tool call ──────────┘
     │
     ▼ (model issues no more function calls)
Final JSON report extracted from accumulated state
```

### Available Agent Tools

| Tool | FFmpeg Command | Purpose |
|------|---------------|---------|
| `extract_audio_metadata` | `ffprobe -print_format json` | Duration, codec, sample rate, channels |
| `detect_silence_segments` | `ffmpeg -af silencedetect` | Silence intervals with timestamps |
| `analyze_volume_levels` | `ffmpeg -af volumedetect` | Mean/peak dB, histogram |
| `detect_audio_clipping` | `ffmpeg -af astats=metadata=1` | Peak levels, clipping flag |
| `analyze_audio_statistics` | `ffmpeg -af astats` | RMS, crest factor, noise floor |
| `run_rules_based_analysis` | *(Python logic)* | Issue list, quality score, grade |
| `generate_final_report` | *(Gemini + Python)* | Assembled JSON + LLM narrative |

---

## Extending the System

### Add a New Analysis Tool

1. Add the FFmpeg analysis method to `analyzers/quality_analyzer.py`
2. Add the `FunctionDeclaration` to `agent/tool_definitions.py`
3. Add the handler method to `AudioAnalysisAgent._dispatch` in `agent/audio_agent.py`
4. Update `RulesEngine` in `analyzers/rules_engine.py` to use the new data
5. Update `ReportGenerator` to include the new field in the output

### Add a New API Endpoint

1. Create or update a file in `api/routes/`
2. Add the router to `api/app.py`
3. Add request/response schemas to `api/schemas/models.py`

### Use a Different LLM

Replace `llm/gemini_client.py` with any provider that supports:
- One-shot completions (`complete()`)
- Multi-turn function calling (`build_agent_model()`, `start_chat()`)

---