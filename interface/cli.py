"""
interface/cli.py – Rich terminal interface for the audio analysis system.

Commands
--------
  analyze   – Analyse a single audio file.
  batch     – Analyse multiple files in a directory or explicit list.
  record    – Record from the microphone then analyse.
  serve     – Start the REST API server.
  devices   – List audio input devices.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import click
from rich import print as rprint
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

console = Console()

# Helpers
def _grade_colour(grade: str) -> str:
    return {"A": "green", "B": "cyan", "C": "yellow", "D": "orange3", "F": "red"}.get(grade, "white")


def _severity_colour(sev: str) -> str:
    return {"critical": "red", "warning": "yellow", "info": "blue"}.get(sev, "white")


def _print_report(report: dict) -> None:
    """Render a single analysis report in the terminal."""
    aq = report.get("audio_quality", {})
    grade = aq.get("quality_grade", "?")
    score = aq.get("quality_score", 0)
    grade_col = _grade_colour(grade)

    # Header panel
    header = (
        f"[bold]{report.get('file_name', 'Unknown')}[/bold]\n"
        f"Duration: [cyan]{report.get('duration_hms', '?')}[/cyan]  "
        f"Size: [cyan]{report.get('file_size_human', '?')}[/cyan]  "
        f"Format: [cyan]{report.get('format', '?')}[/cyan]"
    )
    console.print(Panel(header, title="[bold blue]Audio Analysis Report[/bold blue]", expand=False))

    # Quality metrics table
    metrics = Table(title="Audio Quality Metrics", show_header=True, header_style="bold magenta")
    metrics.add_column("Metric", style="dim")
    metrics.add_column("Value")

    def _row(label: str, value, fmt: str = "{}") -> None:
        disp = fmt.format(value) if value is not None else "[dim]N/A[/dim]"
        metrics.add_row(label, str(disp))

    _row("Quality Score", f"[{grade_col}]{score}/100 (Grade {grade})[/{grade_col}]")
    _row("Silence Ratio", f"{aq.get('silence_ratio', 0):.1%}")
    _row("Silence Segments", aq.get("silence_segment_count"))
    _row("Clipping Detected", f"[red]YES[/red]" if aq.get("clipping_detected") else "[green]No[/green]")
    _row("Avg Volume (dB)", f"{aq.get('avg_volume_db'):.1f}" if aq.get("avg_volume_db") else "N/A")
    _row("Peak Volume (dB)", f"{aq.get('peak_volume_db'):.1f}" if aq.get("peak_volume_db") else "N/A")
    _row("Noise Floor (dB)", f"{aq.get('noise_floor_db'):.1f}" if aq.get("noise_floor_db") else "N/A")
    _row("Dynamic Range (dB)", f"{aq.get('dynamic_range_db'):.1f}" if aq.get("dynamic_range_db") else "N/A")
    _row("Sample Rate", f"{aq.get('sample_rate_hz', '?')} Hz")
    _row("Channels", aq.get("channels"))
    _row("Codec", aq.get("codec"))
    _row("Bitrate", f"{aq.get('bitrate_kbps', '?')} kbps")

    console.print(metrics)

    # Issues
    issues = report.get("rules_analysis", {}).get("detailed_issues", [])
    if issues:
        issue_table = Table(title="Detected Issues", show_header=True, header_style="bold red")
        issue_table.add_column("Severity", width=10)
        issue_table.add_column("Category", width=12)
        issue_table.add_column("Message")
        for issue in issues:
            sev = issue.get("severity", "info")
            col = _severity_colour(sev)
            issue_table.add_row(
                f"[{col}]{sev.upper()}[/{col}]",
                issue.get("category", ""),
                issue.get("message", ""),
            )
        console.print(issue_table)
    else:
        console.print(Panel("[green]✓ No issues detected.[/green]", title="Issues"))

    # LLM Insights
    insights = report.get("llm_insights", {})
    if insights.get("executive_summary"):
        console.print(Panel(
            insights["executive_summary"],
            title="[bold yellow]AI Executive Summary[/bold yellow]",
        ))

    actions = insights.get("recommended_actions", [])
    if actions:
        action_text = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(actions))
        console.print(Panel(action_text, title="[bold yellow]Recommended Actions[/bold yellow]"))

    verdict = insights.get("overall_verdict", "")
    if verdict:
        console.print(Panel(f"[bold]{verdict}[/bold]", title="Overall Verdict"))

# CLI group
@click.group()
@click.version_option("1.0.0")
def cli():
    """
    🎙 Audio Analysis System – Powered by FFmpeg + Gemini AI

    Analyse audio files, live recordings, or batches of files
    and get AI-generated quality reports.
    """

# Commands
@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--save", is_flag=True, default=True, help="Save JSON report to output/")
@click.option("--json-only", is_flag=True, help="Print raw JSON only (for piping)")
def analyze(file_path: str, save: bool, json_only: bool):
    """Analyse a single AUDIO_FILE and display the quality report."""
    from agent.audio_agent import AudioAnalysisAgent

    agent = AudioAnalysisAgent()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"Analyzing {Path(file_path).name}…", total=None)
        report = agent.analyze(file_path)

    if json_only:
        print(json.dumps(report, indent=2))
        return

    _print_report(report)

    if save:
        from config import settings
        out = settings.OUTPUT_DIR / f"{Path(file_path).stem}_report.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        console.print(f"\n[dim]Report saved → {out}[/dim]")

@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--pattern", default="*.wav,*.mp3,*.m4a,*.flac,*.ogg,*.aac", show_default=True)
@click.option("--no-summary", is_flag=True, help="Skip LLM batch summary")
@click.option("--json-only", is_flag=True)
def batch(input_path: str, pattern: str, no_summary: bool, json_only: bool):
    """
    Analyse all audio files in INPUT_PATH (directory) or a comma-separated
    list of file paths.
    """
    from processor.batch_processor import BatchProcessor

    path = Path(input_path)
    if path.is_dir():
        extensions = [p.strip() for p in pattern.split(",")]
        files: list[Path] = []
        for ext in extensions:
            files.extend(path.glob(ext))
        files = sorted(set(files))
    else:
        # Treat input as a single file
        files = [path]

    if not files:
        console.print(f"[yellow]No audio files found in {input_path}[/yellow]")
        sys.exit(1)

    console.print(f"[bold]Found {len(files)} file(s) to process.[/bold]")

    processor = BatchProcessor()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=not json_only,
    ) as progress:
        progress.add_task("Running batch analysis…", total=None)
        report = processor.process_files(
            [str(f) for f in files],
            generate_batch_summary=not no_summary,
        )

    if json_only:
        print(json.dumps(report, indent=2))
        return

    # Summary table
    console.print(Panel(
        f"Total: [bold]{report['total_files']}[/bold]  "
        f"Processed: [green]{report['processed']}[/green]  "
        f"Failed: [red]{report['failed']}[/red]  "
        f"Avg Score: [cyan]{report['avg_quality_score']}[/cyan]",
        title="[bold]Batch Results[/bold]",
    ))

    # Per-file summary table
    tbl = Table(title="Individual File Results", header_style="bold magenta")
    tbl.add_column("File")
    tbl.add_column("Score", justify="right")
    tbl.add_column("Grade", justify="center")
    tbl.add_column("Issues", justify="right")
    tbl.add_column("Clipping")

    for r in report["individual_reports"]:
        aq = r.get("audio_quality", {})
        grade = aq.get("quality_grade", "?")
        console.print(tbl.add_row(
            r.get("file_name", "?"),
            str(aq.get("quality_score", "?")),
            f"[{_grade_colour(grade)}]{grade}[/{_grade_colour(grade)}]",
            str(len(r.get("issues", []))),
            "[red]YES[/red]" if aq.get("clipping_detected") else "[green]No[/green]",
        ) or "")

    console.print(tbl)

    # Batch LLM summary
    bs = report.get("batch_summary", {})
    if bs.get("overall_batch_verdict"):
        console.print(Panel(bs["overall_batch_verdict"], title="[bold yellow]Batch Verdict[/bold yellow]"))

@cli.command()
@click.option("--duration", "-d", default=30, show_default=True, type=int, help="Recording duration (seconds)")
@click.option("--device", type=int, default=None, help="Input device index (see: devices command)")
@click.option("--json-only", is_flag=True)
def record(duration: int, device: Optional[int], json_only: bool):
    """Record audio from the microphone then analyse it."""
    from agent.audio_agent import AudioAnalysisAgent
    from recorder.live_recorder import LiveRecorder

    console.print(f"[bold yellow]Recording {duration}s from microphone…[/bold yellow]")
    recorder = LiveRecorder(device=device)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold red]● Recording… {task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(f"{duration}s", total=None)
        wav = recorder.record_fixed(duration)

    console.print(f"[green]Recording saved: {wav}[/green]")

    agent = AudioAnalysisAgent()
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Analysing…"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("", total=None)
        report = agent.analyze(wav)

    if json_only:
        print(json.dumps(report, indent=2))
    else:
        _print_report(report)

@cli.command()
def devices():
    """List available audio input devices."""
    from recorder.live_recorder import LiveRecorder

    devs = LiveRecorder.list_devices()
    if not devs:
        console.print("[yellow]No input devices found.[/yellow]")
        return

    tbl = Table(title="Available Audio Input Devices", header_style="bold magenta")
    tbl.add_column("Index", justify="right")
    tbl.add_column("Name")
    tbl.add_column("Channels", justify="right")
    tbl.add_column("Default Sample Rate", justify="right")

    for d in devs:
        tbl.add_row(
            str(d["index"]),
            d["name"],
            str(d["max_input_channels"]),
            f"{int(d['default_samplerate'])} Hz",
        )

    console.print(tbl)

@cli.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8000, show_default=True, type=int)
@click.option("--reload", is_flag=True, help="Auto-reload on file changes (dev mode)")
def serve(host: str, port: int, reload: bool):
    """Start the REST API server."""
    import uvicorn

    console.print(Panel(
        f"[bold green]Audio Analysis API[/bold green]\n"
        f"Listening on [cyan]http://{host}:{port}[/cyan]\n"
        f"Docs: [cyan]http://{host}:{port}/docs[/cyan]",
        title="Server Starting",
    ))
    uvicorn.run("api.app:app", host=host, port=port, reload=reload)
