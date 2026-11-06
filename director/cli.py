"""Antigravity Director CLI.
Interactive terminal interface and benchmark dashboard using Rich.
"""
import sys
import argparse
import asyncio
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn

from director.ollama_worker import OllamaWorker
from director.orchestrator import HybridDirector
from director.runner import BenchmarkRunner, BenchmarkSummary
from director.benchmark_tasks import BENCHMARK_TASKS
from director.config import CONFIG

console = Console()

def display_header():
    header_text = Text()
    header_text.append("⚡ ANTIGRAVITY DIRECTOR: HYBRID MODEL ENGINE\n", style="bold cyan")
    header_text.append("Frontier Model Director + Local Ollama MLX Workers\n", style="bold white")
    header_text.append("Metrics: Tokens Saved | Accuracy | Latency | Compute Resources", style="dim")
    console.print(Panel(header_text, border_style="cyan", expand=False))

async def cmd_models():
    """List available local Ollama models."""
    display_header()
    worker = OllamaWorker()
    try:
        models = await worker.list_models()
        table = Table(title="Installed Ollama Local Models", border_style="blue")
        table.add_column("Model Name", style="bold green")
        table.add_column("Default Target", style="cyan")
        
        for m in models:
            is_default = "★ [bold yellow]Active Worker[/bold yellow]" if m == CONFIG.default_local_model else ""
            table.add_row(m, is_default)
        console.print(table)
    except Exception as e:
        console.print(f"[bold red]Failed to reach Ollama at {CONFIG.ollama_base_url}:[/bold red] {e}")

async def cmd_run(prompt: str, local_model: str):
    """Run a single task interactively in Hybrid mode."""
    display_header()
    director = HybridDirector(local_model=local_model)
    console.print(f"[bold yellow]Task Prompt:[/bold yellow] {prompt}\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task1 = progress.add_task("[cyan]Director decomposing and routing to local worker...", total=None)
        res = await director.execute_hybrid(task_id="interactive_task", prompt=prompt)
        progress.update(task1, completed=True)

    console.print("\n[bold green]Result Output:[/bold green]")
    console.print(Panel(res.output, title="Worker Generated Output", border_style="green"))

    # Metrics Panel
    m = res.metrics
    metrics_table = Table(title="Telemetry & Resource Metrics", border_style="cyan")
    metrics_table.add_column("Metric", style="bold white")
    metrics_table.add_column("Value", style="bold yellow")

    metrics_table.add_row("Execution Wall Time", f"{m.wall_time_ms} ms")
    metrics_table.add_row("Frontier Tokens (Plan/Judge)", f"{m.frontier_input_tokens + m.frontier_output_tokens}")
    metrics_table.add_row("Local Worker Tokens Generated", f"{m.local_output_tokens} tokens")
    metrics_table.add_row("Local Generation Throughput", f"{m.local_tokens_per_sec} tokens/sec")
    metrics_table.add_row("Estimated Frontier Cost", f"${m.frontier_cost_usd:.6f}")
    metrics_table.add_row("System RAM (Start ➔ End)", f"{m.ram_used_gb_start} GB ➔ {m.ram_used_gb_end} GB (Δ {m.ram_delta_mb} MB)")
    metrics_table.add_row("Fallback Escalated?", "[red]Yes[/red]" if res.fallback_triggered else "[green]No (Worker Succeeded)[/green]")

    console.print(metrics_table)

async def cmd_benchmark(export_path: str = None, local_model: str = "auto"):
    """Run the side-by-side comparative benchmark suite."""
    display_header()
    target_model = None if local_model == "auto" else local_model
    director = HybridDirector(local_model=target_model)
    runner = BenchmarkRunner(director=director)

    mode_label = "Specialist Auto-Router (qwen2.5-coder / llama3.2 / gemma4)" if target_model is None else target_model
    console.print(f"[bold cyan]Running comparative benchmark across {len(BENCHMARK_TASKS)} tasks...[/bold cyan]")
    console.print(f"[bold yellow]Worker Mode:[/bold yellow] {mode_label}\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        t = progress.add_task("[yellow]Executing Frontier, Local, and Hybrid trials...", total=None)
        summary: BenchmarkSummary = await runner.run_all()
        progress.update(t, completed=True)

    # Render Task Breakdown Table
    table = Table(title="Benchmark Comparison Matrix", border_style="cyan")
    table.add_column("Task ID", style="bold white")
    table.add_column("Category", style="dim")
    table.add_column("Worker Model", style="bold magenta")
    table.add_column("Frontier Baseline", style="blue")
    table.add_column("Local Worker Baseline", style="cyan")
    table.add_column("Hybrid Director", style="green")
    table.add_column("Tokens Saved", style="bold yellow")
    table.add_column("Cost Saved", style="bold green")

    for c in summary.comparisons:
        worker_used = c.local_only.details.get("model", "local")
        f_status = "✓ Pass" if c.frontier_only.verified else "✗ Fail"
        l_status = "✓ Pass" if c.local_only.verified else "✗ Fail"
        h_status = "✓ Pass" if c.hybrid.verified else "✗ Fail"
        if c.hybrid.fallback_triggered:
            h_status += " (Fallback)"

        f_summary = f"{f_status} | {c.frontier_only.metrics.wall_time_ms}ms"
        l_summary = f"{l_status} | {c.local_only.metrics.wall_time_ms}ms"
        h_summary = f"{h_status} | {c.hybrid.metrics.wall_time_ms}ms"

        table.add_row(
            c.task.id,
            c.task.category,
            worker_used,
            f_summary,
            l_summary,
            h_summary,
            f"{c.tokens_saved} ({c.cost_saved_pct:.1f}%)",
            f"${c.cost_saved_usd:.5f}",
        )
    console.print(table)

    # Render Summary Panel
    summary_panel = Table.grid(padding=1)
    summary_panel.add_column(style="bold cyan")
    summary_panel.add_column(style="bold yellow")

    summary_panel.add_row("Frontier Baseline Accuracy:", f"{summary.frontier_accuracy_pct}%")
    summary_panel.add_row("Local Worker Baseline Accuracy:", f"{summary.local_accuracy_pct}%")
    summary_panel.add_row("Hybrid Orchestration Accuracy:", f"{summary.hybrid_accuracy_pct}%")
    summary_panel.add_row("Total Frontier Tokens (Baseline):", f"{summary.total_frontier_tokens_baseline:,}")
    summary_panel.add_row("Total Frontier Tokens (Hybrid):", f"{summary.total_frontier_tokens_hybrid:,}")
    summary_panel.add_row("Total Tokens Saved:", f"[bold green]{summary.total_tokens_saved:,} ({summary.overall_tokens_saved_pct}% reduction)[/bold green]")
    summary_panel.add_row("Baseline Cost:", f"${summary.total_cost_baseline_usd:.5f}")
    summary_panel.add_row("Hybrid Cost:", f"${summary.total_cost_hybrid_usd:.5f}")
    summary_panel.add_row("Total Savings:", f"[bold green]${summary.total_cost_saved_usd:.5f}[/bold green]")

    console.print("\n")
    console.print(Panel(summary_panel, title="Overall Benchmark Summary", border_style="green"))

    if export_path:
        with open(export_path, "w") as f:
            f.write(summary.model_dump_json(indent=2))
        console.print(f"[bold green]Exported telemetry report to:[/bold green] {export_path}")

def main():
    parser = argparse.ArgumentParser(description="Antigravity Hybrid Director & Benchmarking Framework")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # models command
    subparsers.add_parser("models", help="List installed Ollama local models")

    # run command
    run_parser = subparsers.add_parser("run", help="Run a single prompt in hybrid mode")
    run_parser.add_argument("prompt", type=str, help="Prompt or task to execute")
    run_parser.add_argument("--model", type=str, default=CONFIG.default_local_model, help="Local model name")

    # benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run comparative benchmark suite")
    bench_parser.add_argument("--export", type=str, default=None, help="Path to export JSON benchmark results")
    bench_parser.add_argument("--model", type=str, default="auto", help="Local model name or 'auto' for specialist routing")

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Launch the interactive Visual Web Dashboard")
    serve_parser.add_argument("--port", type=int, default=8501, help="Port to bind dashboard server (default: 8501)")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")

    args = parser.parse_args()

    if args.command == "models":
        asyncio.run(cmd_models())
    elif args.command == "run":
        asyncio.run(cmd_run(args.prompt, args.model))
    elif args.command == "benchmark":
        asyncio.run(cmd_benchmark(args.export, args.model))
    elif args.command == "serve":
        from director.server import start_server
        start_server(port=args.port, host=args.host)

if __name__ == "__main__":
    main()
