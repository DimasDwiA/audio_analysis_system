"""
run_agent.py – Cara termudah untuk menjalankan Audio Analysis Agent.

Mendukung 3 mode:
  1. Analisis file tunggal (dengan atau tanpa Gemini API key)
  2. Batch processing seluruh folder
  3. Demo otomatis (generate WAV sintetis, tidak butuh file apapun)

Cara pakai:
    python run_agent.py                          # mode demo otomatis
    python run_agent.py audio.wav                # analisis 1 file
    python run_agent.py audio.wav --no-llm       # hanya FFmpeg + rules
    python run_agent.py /folder/rekaman/ --batch # batch seluruh folder
"""

from __future__ import annotations

import json
import math
import os
import struct
import sys
import tempfile
import wave
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.table import Table

# Load .env before checking environment variables
from config import settings

console = Console()

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║      🎙  Audio Analysis Agent – Interactive Runner               ║
║      FFmpeg + Rules Engine + Gemini AI (hybrid pipeline)         ║
╚══════════════════════════════════════════════════════════════════╝
"""

GRADE_COLOR = {"A": "green", "B": "cyan", "C": "yellow", "D": "orange3", "F": "red"}
SEV_COLOR   = {"critical": "red", "warning": "yellow", "info": "blue"}

# Synthetic WAV helper (demo mode)
def _make_demo_wav(path: Path, duration: float = 12.0) -> Path:
    """Generate a WAV with a 440 Hz tone + 4-second silence in the middle."""
    sr = 44100
    total = int(duration * sr)
    s_start, s_end = int(3.5 * sr), int(7.5 * sr)   
    frames = []
    for i in range(total):
        if s_start <= i < s_end:
            frames.append(0)
        else:
            frames.append(int(0.45 * 32767 * math.sin(2 * math.pi * 440 * i / sr)))
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{total}h", *frames))
    return path

# Core: run the full pipeline (with or without LLM)
def run_pipeline(audio_path: Path, use_llm: bool = True) -> dict:
    """
    Execute the complete analysis pipeline on *audio_path*.

    - use_llm=True  → full agentic mode (AudioAnalysisAgent + Gemini)
    - use_llm=False → FFmpeg + Rules engine only (no API key required)
    """
    from analyzers.metadata_extractor import MetadataExtractor
    from analyzers.quality_analyzer import QualityAnalyzer
    from analyzers.rules_engine import RulesEngine
    from reports.report_generator import ReportGenerator

    console.print(f"\n[bold]▶ Starting the analysis:[/bold] [cyan]{audio_path.name}[/cyan]")
    console.print(Rule(style="dim"))

    # Step 1 
    console.print("[bold blue][1/5] Extracting metadata (ffprobe)…[/bold blue]")
    meta = MetadataExtractor().extract(audio_path)
    _print_meta_table(meta)

    # Step 2–4
    qa = QualityAnalyzer()

    console.print("\n[bold blue][2/5] Detecting silence (ffmpeg silencedetect)…[/bold blue]")
    silence = qa.detect_silence(audio_path, threshold_db=-30, min_duration=0.5)
    _print_silence(silence, meta.get("duration_seconds", 0))

    console.print("\n[bold blue][3/5] Analyzing volume (ffmpeg volumedetect)…[/bold blue]")
    volume = qa.analyze_volume(audio_path)
    _print_volume(volume)

    console.print("\n[bold blue][4/5] Detecting clipping + statistics (ffmpeg astats)…[/bold blue]")
    clipping = qa.detect_clipping(audio_path)
    stats    = qa.analyze_stats(audio_path)
    _print_quality(clipping, stats)

    # Step 5: Rules engine
    console.print("\n[bold blue][5/5] Rules-based analysis…[/bold blue]")
    all_data = dict(
        metadata=meta,
        silence_analysis=silence,
        volume_analysis=volume,
        clipping_analysis=clipping,
        audio_statistics=stats,
    )
    all_data["rules_analysis"] = RulesEngine().analyze(all_data)
    _print_rules(all_data["rules_analysis"])

    # LLM Gemini insights
    llm_client = None
    if use_llm:
        llm_client = _try_init_llm()

    report = ReportGenerator(llm_client=llm_client).build(audio_path, all_data)
    _print_insights(report.get("llm_insights", {}), use_llm=llm_client is not None)

    return report

# Agentic mode (full Gemini function-calling loop)
def run_agent(audio_path: Path) -> dict:
    from agent.audio_agent import AudioAnalysisAgent

    console.print(f"\n[bold magenta]🤖 AGENTIC MODE[/bold magenta] – "
                  f"Gemini will determine which tool to call dynamically.\n")
    console.print(f"[bold]▶ File:[/bold] [cyan]{audio_path.name}[/cyan]")
    console.print(Rule(style="dim"))

    agent = AudioAnalysisAgent()
    report = agent.analyze(audio_path)
    _print_final_report(report)
    return report

# Batch mode
def run_batch(folder: Path) -> dict:
    """Analisis semua file audio dalam sebuah folder."""
    from processor.batch_processor import BatchProcessor

    patterns = ["*.wav", "*.mp3", "*.m4a", "*.flac", "*.ogg", "*.aac"]
    files: list[Path] = []
    for p in patterns:
        files.extend(folder.glob(p))
    files = sorted(set(files))

    if not files:
        console.print(f"[yellow]Tidak ada file audio ditemukan di: {folder}[/yellow]")
        sys.exit(1)

    console.print(f"\n[bold]📂 Batch Mode:[/bold] {len(files)} file ditemukan di [cyan]{folder}[/cyan]")
    console.print(Rule(style="dim"))

    processor = BatchProcessor()
    report = processor.process_files([str(f) for f in files])
    _print_batch_summary(report)
    return report

# Rich display helpers
def _print_meta_table(meta: dict) -> None:
    t = Table(show_header=False, box=None, padding=(0, 3))
    for k, v in [
        ("File",        meta.get("file_name")),
        ("Format",      meta.get("format_name")),
        ("Duration",    meta.get("duration_hms")),
        ("Sample Rate", f"{meta.get('sample_rate_hz')} Hz"),
        ("Channels",    meta.get("channels")),
        ("Codec",       meta.get("codec_name")),
        ("Bitrate",     f"{meta.get('overall_bitrate_kbps')} kbps"),
        ("Size",        meta.get("file_size_human")),
    ]:
        t.add_row(f"[dim]{k}[/dim]", f"[cyan]{v}[/cyan]")
    console.print(t)

def _print_silence(silence: dict, duration: float) -> None:
    ratio = silence["total_silence_seconds"] / duration if duration else 0
    console.print(
        f"  Segmen : [yellow]{silence['segment_count']}[/yellow]  │  "
        f"Total   : [yellow]{silence['total_silence_seconds']:.1f}s[/yellow]  │  "
        f"Rasio   : [yellow]{ratio:.1%}[/yellow]"
    )
    for seg in silence.get("segments", []):
        console.print(
            f"  [dim]├[/dim] {seg['start_s']:.2f}s → {seg.get('end_s', '?')}s  "
            f"([yellow]{seg.get('duration_s', '?')}s[/yellow])"
        )

def _print_volume(volume: dict) -> None:
    quiet = volume.get("is_too_quiet", False)
    console.print(
        f"  Mean: [cyan]{volume.get('mean_volume_db', 'N/A')} dB[/cyan]  │  "
        f"Peak: [cyan]{volume.get('max_volume_db', 'N/A')} dB[/cyan]  │  "
        f"Too quiet: {'[red]YA[/red]' if quiet else '[green]Tidak[/green]'}"
    )

def _print_quality(clipping: dict, stats: dict) -> None:
    clipped = clipping.get("clipping_detected", False)
    console.print(
        f"  Clipping   : {'[red]TERDETEKSI![/red]' if clipped else '[green]Tidak ada[/green]'}  │  "
        f"Peak       : [cyan]{clipping.get('peak_level_db', 'N/A')} dBFS[/cyan]\n"
        f"  Noise floor: [cyan]{stats.get('noise_floor_db', 'N/A')} dB[/cyan]  │  "
        f"Dyn. range : [cyan]{stats.get('dynamic_range_db', 'N/A')} dB[/cyan]  │  "
        f"DC offset  : [cyan]{stats.get('dc_offset', 'N/A')}[/cyan]"
    )

def _print_rules(rules: dict) -> None:
    score = rules["quality_score"]
    grade = rules["quality_grade"]
    gc = GRADE_COLOR.get(grade, "white")
    console.print(
        f"  Skor  : [{gc}][bold]{score}/100[/bold][/{gc}]  │  "
        f"Grade : [{gc}][bold]{grade}[/bold][/{gc}]  │  "
        f"Flags : {', '.join(rules['summary_flags']) or '[dim]tidak ada[/dim]'}"
    )
    for issue in rules.get("issues", []):
        sev = issue.get("severity", "info")
        sc  = SEV_COLOR.get(sev, "white")
        console.print(
            f"  [{sc}][{sev.upper():8}][/{sc}] "
            f"[dim]{issue.get('category',''):10}[/dim] {issue.get('message','')}"
        )

def _print_insights(insights: dict, use_llm: bool = True) -> None:
    console.print(Rule(style="dim"))
    if not use_llm or not insights.get("executive_summary"):
        console.print(
            Panel(
                "[dim]Gemini API key not found – enable it with:\n"
                "  echo GEMINI_API_KEY=xxx >> .env\n"
                "  (free key at https://aistudio.google.com)[/dim]",
                title="[yellow]LLM Insights[/yellow]",
            )
        )
        return

    console.print(Panel(
        insights.get("executive_summary", ""),
        title="[bold yellow]🤖 AI Executive Summary[/bold yellow]",
    ))
    actions = insights.get("recommended_actions", [])
    if actions:
        action_txt = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(actions))
        console.print(Panel(action_txt, title="[bold yellow]Recommended Actions[/bold yellow]"))
    verdict = insights.get("overall_verdict", "")
    if verdict:
        console.print(Panel(f"[bold]{verdict}[/bold]", title="Verdict"))

def _print_final_report(report: dict) -> None:
    aq    = report.get("audio_quality", {})
    grade = aq.get("quality_grade", "?")
    score = aq.get("quality_score", 0)
    gc    = GRADE_COLOR.get(grade, "white")

    console.print(Panel(
        f"[bold]{report.get('file_name')}[/bold]\n"
        f"Duration: [cyan]{report.get('duration_hms')}[/cyan]  "
        f"Size: [cyan]{report.get('file_size_human')}[/cyan]",
        title="[bold blue]Hasil Analisis Agent[/bold blue]",
    ))
    console.print(
        f"  Skor   : [{gc}][bold]{score}/100 (Grade {grade})[/bold][/{gc}]\n"
        f"  Issues : [red]{len(report.get('issues', []))}[/red]"
    )
    _print_insights(report.get("llm_insights", {}), use_llm=True)

def _print_batch_summary(report: dict) -> None:
    console.print(Panel(
        f"Total  : [bold]{report['total_files']}[/bold]  │  "
        f"Success: [green]{report['processed']}[/green]  │  "
        f"Failed  : [red]{report['failed']}[/red]\n"
        f"Avg Score : [cyan]{report['avg_quality_score']}[/cyan]  │  "
        f"Min: [red]{report['min_quality_score']}[/red]  │  "
        f"Max: [green]{report['max_quality_score']}[/green]",
        title="[bold]📊 Batch Selesai[/bold]",
    ))
    for r in report["individual_reports"]:
        aq = r.get("audio_quality", {})
        grade = aq.get("quality_grade", "?")
        gc = GRADE_COLOR.get(grade, "white")
        console.print(
            f"  [{gc}]{grade}[/{gc}] {aq.get('quality_score','?'):3}/100  "
            f"[dim]{r.get('file_name','?')}[/dim]"
        )
    bs = report.get("batch_summary", {})
    if bs.get("overall_batch_verdict"):
        console.print(Panel(
            f"[bold]{bs['overall_batch_verdict']}[/bold]",
            title="[yellow]Batch Verdict[/yellow]",
        ))

def _try_init_llm():
    """Coba inisialisasi Gemini. Return None jika API key tidak ada."""
    try:
        from llm.gemini_client import GeminiClient
        return GeminiClient()
    except EnvironmentError:
        return None

# Entry point
def main() -> None:
    console.print(BANNER, highlight=False)

    args = sys.argv[1:]
    use_agent_mode = "--agent" in args
    use_no_llm     = "--no-llm" in args
    batch_mode     = "--batch" in args

    # Filter out flags
    positional = [a for a in args if not a.startswith("--")]

    # Check API key 
    has_key = bool(settings.GEMINI_API_KEY.strip())
    if has_key:
        console.print("[green]✅ Detected GEMINI_API_KEY  – mode LLM active[/green]")
    else:
        console.print(
            "[yellow]⚠️  GEMINI_API_KEY Not Found – "
            "running in FFmpeg + Rules only mode[/yellow]\n"
            "[dim]   To enable AI insights:\n"
            "   1. Go to https://aistudio.google.com → Get API key (FREE)\n"
            "   2. Add to the .env file: GEMINI_API_KEY=AIzaSy...\n"
            "   3. Rerun this script[/dim]"
        )

    # Batch mode
    if batch_mode:
        folder = Path(positional[0]) if positional else Path(".")
        report = run_batch(folder)
        _save_and_exit(report, "batch_report.json")
        return

    # Tentukan file audio
    if positional:
        audio_path = Path(positional[0])
        if not audio_path.exists():
            console.print(f"[red]❌ File tidak ditemukan: {audio_path}[/red]")
            sys.exit(1)
    else:
        console.print(
            "\n[bold yellow]Tidak ada file yang diberikan – "
            "membuat WAV demo otomatis (12 detik)…[/bold yellow]"
        )
        tmp = Path(tempfile.mktemp(suffix="_demo.wav"))
        audio_path = _make_demo_wav(tmp)
        console.print(f"[dim]WAV demo dibuat: {audio_path}[/dim]")

    # Mode Execution
    if use_agent_mode and has_key:
        # Full agentic loop
        report = run_agent(audio_path)
    else:
        if use_agent_mode and not has_key:
            console.print(
                "[yellow]⚠️  --agent membutuhkan GEMINI_API_KEY. "
                "Beralih ke pipeline mode.[/yellow]"
            )
        # Pipeline langsung (deterministic + optional LLM untuk insights)
        report = run_pipeline(audio_path, use_llm=has_key and not use_no_llm)

    _save_and_exit(report, f"{audio_path.stem}_report.json")

def _save_and_exit(report: dict, filename: str) -> None:
    out = Path("./output") / filename
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\n[dim]💾 Report saved → {out}[/dim]")
    console.print("[bold green]✅ Analysis done.[/bold green]")

if __name__ == "__main__":
    main()
