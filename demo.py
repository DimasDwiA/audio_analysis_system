"""
demo.py – Stand-alone demonstration of the Audio Analysis System.

This script can run WITHOUT a Gemini API key – it exercises the full
FFmpeg-based analysis and rules engine, then shows what LLM insights
would look like (using stub data when the key is absent).

Usage:
    python demo.py                         # auto-generates a test WAV
    python demo.py /path/to/audio.wav      # analyse an existing file
    python demo.py --batch /path/to/dir/   # batch demo
"""

from __future__ import annotations

import json
import math
import struct
import sys
import wave
import tempfile
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

console = Console()

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║         🎙  Audio Analysis System – Demo                     ║
║         FFmpeg + Gemini AI | Agentic Pipeline                ║
╚══════════════════════════════════════════════════════════════╝
"""

# Synthetic WAV generator
def generate_test_wav(output_path: Path, duration: float = 10.0) -> Path:
    """
    Create a synthetic WAV with a 440 Hz tone and an embedded silent section
    (simulates a realistic audio file with quality issues).
    """
    sr = 44100
    amp = 0.45
    n_frames = int(duration * sr)
    silence_start = int(3.0 * sr)
    silence_end = int(7.0 * sr)

    frames: list[int] = []
    for i in range(n_frames):
        if silence_start <= i < silence_end:
            sample = 0
        else:
            sample = int(amp * 32767 * math.sin(2 * math.pi * 440 * i / sr))
        frames.append(sample)

    with wave.open(str(output_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{n_frames}h", *frames))

    console.print(
        f"[dim]Generated synthetic WAV: {output_path.name} "
        f"({duration:.0f}s, 440 Hz tone + 4s silence)[/dim]"
    )
    return output_path

# Core demo runner
def run_demo(audio_path: Path) -> dict:
    """Run the full analysis pipeline (sans LLM) and display results."""
    from analyzers.metadata_extractor import MetadataExtractor
    from analyzers.quality_analyzer import QualityAnalyzer
    from analyzers.rules_engine import RulesEngine
    from reports.report_generator import ReportGenerator

    console.print(f"\n[bold]Analysing:[/bold] [cyan]{audio_path.name}[/cyan]")

    # Step 1: Metadata 
    console.print("\n[bold blue]Step 1/5 – Extracting metadata with ffprobe…[/bold blue]")
    meta = MetadataExtractor().extract(audio_path)

    meta_table = Table(show_header=False, box=None, padding=(0, 2))
    for k, v in [
        ("File", meta.get("file_name")),
        ("Format", meta.get("format_name")),
        ("Duration", meta.get("duration_hms")),
        ("Sample Rate", f"{meta.get('sample_rate_hz')} Hz"),
        ("Channels", meta.get("channels")),
        ("Codec", meta.get("codec_name")),
        ("Bitrate", f"{meta.get('overall_bitrate_kbps')} kbps"),
        ("Size", meta.get("file_size_human")),
    ]:
        meta_table.add_row(f"[dim]{k}[/dim]", str(v))
    console.print(meta_table)

    # Step 2: Silence
    console.print("\n[bold blue]Step 2/5 – Detecting silence segments…[/bold blue]")
    silence = QualityAnalyzer().detect_silence(audio_path, threshold_db=-30, min_duration=0.5)
    console.print(
        f"  Found [yellow]{silence['segment_count']}[/yellow] silence segment(s) │ "
        f"Total: [yellow]{silence['total_silence_seconds']:.1f}s[/yellow]"
    )
    for seg in silence["segments"]:
        console.print(
            f"    ├ {seg['start_s']:.2f}s → {seg.get('end_s', '?')}s "
            f"(duration: [yellow]{seg.get('duration_s', '?')}s[/yellow])"
        )

    # Step 3: Volume
    console.print("\n[bold blue]Step 3/5 – Analyzing volume levels…[/bold blue]")
    qa = QualityAnalyzer()
    volume = qa.analyze_volume(audio_path)
    console.print(
        f"  Mean: [cyan]{volume.get('mean_volume_db', 'N/A')} dB[/cyan]  │  "
        f"Peak: [cyan]{volume.get('max_volume_db', 'N/A')} dB[/cyan]  │  "
        f"Too quiet: {'[red]Yes[/red]' if volume.get('is_too_quiet') else '[green]No[/green]'}"
    )

    # Step 4: Clipping + Stats
    console.print("\n[bold blue]Step 4/5 – Checking for clipping and audio statistics…[/bold blue]")
    clipping = qa.detect_clipping(audio_path)
    stats = qa.analyze_stats(audio_path)
    console.print(
        f"  Clipping: {'[red]DETECTED[/red]' if clipping.get('clipping_detected') else '[green]None[/green]'}  │  "
        f"Peak: [cyan]{clipping.get('peak_level_db', 'N/A')} dBFS[/cyan]  │  "
        f"Noise floor: [cyan]{stats.get('noise_floor_db', 'N/A')} dB[/cyan]  │  "
        f"Dynamic range: [cyan]{stats.get('dynamic_range_db', 'N/A')} dB[/cyan]"
    )

    # Step 5: Rules Engine
    console.print("\n[bold blue]Step 5/5 – Applying rules-based analysis…[/bold blue]")
    all_data = {
        "metadata": meta,
        "silence_analysis": silence,
        "volume_analysis": volume,
        "clipping_analysis": clipping,
        "audio_statistics": stats,
    }
    rules = RulesEngine().analyze(all_data)
    all_data["rules_analysis"] = rules

    score = rules["quality_score"]
    grade = rules["quality_grade"]
    grade_colours = {"A": "green", "B": "cyan", "C": "yellow", "D": "orange3", "F": "red"}
    gc = grade_colours.get(grade, "white")

    console.print(
        f"  Quality Score: [{gc}][bold]{score}/100 (Grade {grade})[/bold][/{gc}]  │  "
        f"Issues: [red]{len(rules['issues'])}[/red]  │  "
        f"Flags: {', '.join(rules['summary_flags']) or '[dim]none[/dim]'}"
    )

    if rules["issues"]:
        console.print()
        for issue in rules["issues"]:
            sev = issue.get("severity", "info")
            sev_map = {"critical": "red", "warning": "yellow", "info": "blue"}
            sc = sev_map.get(sev, "white")
            console.print(
                f"  [{sc}][{sev.upper()}][/{sc}] "
                f"[dim]{issue.get('category', '')}[/dim] – {issue.get('message', '')}"
            )

    # Assemble report
    report = ReportGenerator(llm_client=None).build(audio_path, all_data)

    # Show what LLM insights would look like
    _show_llm_stub_insights(rules["issues"], score, grade)

    return report

def _show_llm_stub_insights(issues: list, score: int, grade: str) -> None:
    """Show a realistic example of what Gemini would produce."""
    has_silence = any(i.get("category") == "silence" for i in issues)
    has_clipping = any(i.get("category") == "clipping" for i in issues)
    has_volume = any(i.get("category") == "volume" for i in issues)

    if score >= 90:
        summary = (
            "This audio recording demonstrates excellent quality across all measured parameters. "
            "No significant issues were detected. The signal is clean, well-leveled, and free of "
            "artifacts. This file is production-ready."
        )
        actions = ["No corrective action required.", "Maintain current recording setup for future sessions."]
        verdict = "Excellent – ready for archival or transcription without modification."
    elif has_clipping:
        summary = (
            f"This audio (score: {score}/100, grade {grade}) contains critical clipping distortion. "
            "Clipping causes irreversible waveform damage and will degrade transcription accuracy. "
            "The recording gain was set too high at the source."
        )
        actions = [
            "Reduce recording input gain by 6–10 dB for future sessions.",
            "Apply a de-clipping algorithm (e.g., Adobe Audition's DeClipper) to recover partial audio.",
            "Re-record if the content is critical and clipping is severe.",
        ]
        verdict = "Critical issues detected – re-recording strongly recommended."
    elif has_silence:
        summary = (
            f"This audio (score: {score}/100, grade {grade}) is largely usable but contains "
            "notable silence segments. Long silences may indicate recording interruptions or "
            "equipment failures. Overall signal quality is acceptable."
        )
        actions = [
            "Review and annotate or trim the extended silence segments identified in the report.",
            "Verify the silence was intentional (e.g., natural pauses) vs. a recording failure.",
            "Monitor recording status in real-time during future sessions to catch dropouts early.",
            "Consider noise gating to automatically suppress very short silence artifacts.",
        ]
        verdict = "Mostly usable – trim silence segments before distribution or transcription."
    else:
        summary = (
            f"Audio quality is acceptable (score: {score}/100, grade {grade}) with minor issues "
            "that should be addressed before final delivery. Signal levels and frequency response "
            "are within acceptable ranges."
        )
        actions = [
            "Apply normalisation to bring levels to industry standard (-3 dBFS peak).",
            "Review flagged segments manually before archival.",
        ]
        verdict = "Acceptable quality – minor improvements recommended."

    stub_insights = {
        "executive_summary": summary,
        "recommended_actions": actions,
        "overall_verdict": verdict,
    }

    console.print(
        Panel(
            "[dim italic](Gemini API key not set – showing representative output)[/dim italic]\n\n"
            + stub_insights["executive_summary"],
            title="[bold yellow]AI Executive Summary[/bold yellow]",
        )
    )

    action_text = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(stub_insights["recommended_actions"]))
    console.print(Panel(action_text, title="[bold yellow]Recommended Actions[/bold yellow]"))
    console.print(Panel(f"[bold]{stub_insights['overall_verdict']}[/bold]", title="Overall Verdict"))

# Entry point
def main() -> None:
    console.print(BANNER, highlight=False)

    batch_mode = "--batch" in sys.argv
    if batch_mode:
        idx = sys.argv.index("--batch")
        dir_arg = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
        if not dir_arg:
            console.print("[red]Usage: python demo.py --batch /path/to/directory[/red]")
            sys.exit(1)

        dir_path = Path(dir_arg)
        files = list(dir_path.glob("*.wav")) + list(dir_path.glob("*.mp3"))
        if not files:
            console.print(f"[yellow]No audio files found in {dir_path}[/yellow]")
            sys.exit(1)

        console.print(f"[bold]Batch mode:[/bold] {len(files)} file(s) found in {dir_path}\n")
        for fp in files:
            run_demo(fp)
            console.print("─" * 60)
        return

    # Single-file mode
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        audio_path = Path(sys.argv[1])
        if not audio_path.exists():
            console.print(f"[red]File not found: {audio_path}[/red]")
            sys.exit(1)
    else:
        # Generate a synthetic test file
        console.print("[bold yellow]No audio file specified – generating a synthetic test WAV…[/bold yellow]")
        tmp = tempfile.NamedTemporaryFile(suffix="_demo.wav", delete=False)
        audio_path = generate_test_wav(Path(tmp.name), duration=12.0)

    report = run_demo(audio_path)

    # Show JSON snippet
    snippet = {
        "file_name": report.get("file_name"),
        "duration_seconds": report.get("duration_seconds"),
        "audio_quality": {
            k: v for k, v in report.get("audio_quality", {}).items()
            if k in ("silence_ratio", "clipping_detected", "avg_volume_db",
                     "quality_score", "quality_grade")
        },
        "issues": report.get("issues", []),
    }

    console.print(
        Panel(
            Syntax(json.dumps(snippet, indent=2), "json", theme="monokai"),
            title="[bold]JSON Report Excerpt[/bold]",
        )
    )

    out = Path("./output/demo_report.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    console.print(f"\n[dim]Full report saved → {out}[/dim]")
    console.print("\n[bold green]✓ Demo complete.[/bold green]")
    console.print(
        "[dim]To use the full system with Gemini AI insights:\n"
        "  1. Set GEMINI_API_KEY in .env\n"
        "  2. Run: python main.py analyze /path/to/audio.wav[/dim]"
    )

if __name__ == "__main__":
    main()