"""
cli.py – Command-line interface for the Interview Trainer Agent.
Provides an interactive, rich-formatted terminal experience.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
from typing import Optional, List

app = typer.Typer(
    name="interview-trainer",
    help="🎯 Interview Trainer Agent — Powered by IBM Granite + RAG",
    rich_markup_mode="rich",
)
console = Console()


# ── Banner ────────────────────────────────────────────────────────────────────

def print_banner():
    console.print(Panel.fit(
        "[bold blue]🎯 AICTE Interview Trainer Agent[/bold blue]\n"
        "[dim]Powered by [bold]IBM Granite[/bold] (watsonx.ai) + RAG[/dim]\n"
        "[dim]Personalised questions · Model answers · Prep strategies[/dim]",
        border_style="blue",
        padding=(1, 4),
    ))


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_agent_with_spinner():
    from src.agent import InterviewTrainerAgent
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("Initialising IBM Granite + RAG pipeline…", total=None)
        agent = InterviewTrainerAgent()
        agent.setup()
    return agent


def display_output(title: str, content: str):
    console.print(Panel(
        Markdown(content),
        title=f"[bold cyan]{title}[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))


# ── Commands ──────────────────────────────────────────────────────────────────

@app.command()
def prep(
    name: str = typer.Option(..., "--name", "-n", prompt="Your full name"),
    role: str = typer.Option(..., "--role", "-r", prompt="Target job role"),
    level: str = typer.Option(
        "Mid-Level", "--level", "-l",
        help="Fresher | Junior | Mid-Level | Senior | Lead",
    ),
    years: int = typer.Option(0, "--years", "-y", help="Years of experience"),
    skills: str = typer.Option("", "--skills", "-s", help="Comma-separated skills"),
    company: str = typer.Option("", "--company", "-c", help="Target company"),
    num_questions: int = typer.Option(10, "--questions", "-q", help="Number of questions"),
    resume: Optional[str] = typer.Option(None, "--resume", help="Path to resume file (PDF/DOCX/TXT)"),
):
    """
    🚀 Run the full interview preparation pipeline.
    Generates questions, preparation strategy, and behavioral coaching.
    """
    print_banner()

    skills_list = [s.strip() for s in skills.split(",") if s.strip()]

    agent = get_agent_with_spinner()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(f"Building profile for [bold]{name}[/bold]…", total=None)
        profile = agent.create_profile(
            name=name,
            job_role=role,
            experience_level=level,
            years_of_experience=years,
            skills=skills_list,
            resume_file=resume,
            target_company=company,
        )

    # Profile summary
    table = Table(box=box.ROUNDED, show_header=False, border_style="blue")
    table.add_column("Field", style="bold")
    table.add_column("Value", style="green")
    table.add_row("Candidate", profile.name)
    table.add_row("Target Role", profile.job_role)
    table.add_row("Experience", f"{profile.experience_level} ({profile.years_of_experience} yrs)")
    table.add_row("Skills", profile.skills_str[:80] or "Auto-detected")
    table.add_row("Company", profile.target_company or "Not specified")
    console.print("\n[bold]📋 Candidate Profile[/bold]")
    console.print(table)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Generating with IBM Granite…", total=None)
        results = agent.run_full_preparation(profile, num_questions=num_questions)

    mode = "🟢 IBM Granite LIVE" if results.get("ibm_granite_live") else "🟡 Demo Mode (set WATSONX_API_KEY for live)"
    console.print(f"\n[dim]{mode}[/dim]\n")

    display_output("📋 Interview Questions", results["question_set"])
    display_output("📅 7-Day Preparation Strategy", results["preparation_strategy"])
    display_output("🎭 Behavioral Assessment", results["behavioral_assessment"])

    # Save option
    if Confirm.ask("\n💾 Save results to a file?"):
        output_path = f"prep_{name.replace(' ', '_')}_{role.replace(' ', '_')}.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# Interview Prep: {name} — {role}\n\n")
            f.write("## Questions\n\n" + results["question_set"] + "\n\n")
            f.write("## Preparation Strategy\n\n" + results["preparation_strategy"] + "\n\n")
            f.write("## Behavioral Assessment\n\n" + results["behavioral_assessment"])
        console.print(f"[green]✅ Saved to {output_path}[/green]")


@app.command()
def evaluate(
    name: str = typer.Option(..., "--name", "-n", prompt="Your name"),
    role: str = typer.Option(..., "--role", "-r", prompt="Target role"),
    level: str = typer.Option("Mid-Level", "--level", "-l"),
):
    """
    ✅ Interactive answer evaluator — answer questions and get scored feedback.
    """
    print_banner()
    agent = get_agent_with_spinner()
    profile = agent.create_profile(name=name, job_role=role, experience_level=level)

    console.print(Panel(
        f"[bold]Evaluating answers for:[/bold] {name} → {role}",
        style="green",
    ))

    while True:
        question = Prompt.ask("\n[bold cyan]📌 Enter interview question[/bold cyan] (or 'quit')")
        if question.lower() in ("quit", "q", "exit"):
            break

        answer = Prompt.ask("[bold yellow]💬 Your answer[/bold yellow]")

        with Progress(SpinnerColumn(), TextColumn("Evaluating with IBM Granite…"), transient=True) as p:
            p.add_task("", total=None)
            ev = agent.evaluate_answer(profile, question, answer)

        if ev.score > 0:
            score_color = "green" if ev.score >= 7 else ("yellow" if ev.score >= 5 else "red")
            console.print(f"\n[{score_color}]Score: {ev.score}/10[/{score_color}]")

        display_output("📊 Feedback", ev.raw_feedback)

        if not Confirm.ask("Continue with another question?"):
            break


@app.command()
def questions(
    name: str = typer.Option(..., "--name", "-n", prompt="Your name"),
    role: str = typer.Option(..., "--role", "-r", prompt="Target role"),
    level: str = typer.Option("Mid-Level", "--level", "-l"),
    num: int = typer.Option(10, "--num", help="Number of questions"),
    skills: str = typer.Option("", "--skills"),
):
    """
    📋 Generate interview questions only.
    """
    print_banner()
    agent = get_agent_with_spinner()
    skills_list = [s.strip() for s in skills.split(",") if s.strip()]
    profile = agent.create_profile(name=name, job_role=role, experience_level=level, skills=skills_list)

    with Progress(SpinnerColumn(), TextColumn("Generating questions…"), transient=True) as p:
        p.add_task("", total=None)
        q_output = agent.generate_questions(profile, num_questions=num)

    display_output(f"📋 {num} Interview Questions — {role}", q_output)


@app.command()
def status():
    """
    📡 Show agent and IBM Granite connection status.
    """
    print_banner()
    agent = get_agent_with_spinner()
    s = agent.status

    table = Table(title="Agent Status", box=box.ROUNDED, border_style="blue")
    table.add_column("Component", style="bold")
    table.add_column("Status")

    table.add_row("Agent Ready", "✅ Yes" if s["ready"] else "❌ No")
    table.add_row("IBM Granite", "🟢 LIVE" if s["ibm_granite_live"] else "🟡 Demo Mode")
    table.add_row("Model ID", s["model_id"])
    vs = s["vectorstore_stats"]
    table.add_row("ChromaDB", "✅ Available" if vs.get("chroma_available") else "⚠️ Unavailable")
    table.add_row("Embedder", "✅ Available" if vs.get("embedder_available") else "⚠️ Unavailable")
    table.add_row("KB Chunks", str(vs.get("collection_count") or vs.get("in_memory_docs", 0)))

    console.print(table)

    if not s["ibm_granite_live"]:
        console.print(Panel(
            "[yellow]IBM Granite is running in Demo Mode.[/yellow]\n"
            "To enable live AI, set your credentials in [bold].env[/bold]:\n\n"
            "  WATSONX_API_KEY=your_key\n"
            "  WATSONX_PROJECT_ID=your_project_id\n\n"
            "Get free credentials at: [link]https://cloud.ibm.com/registration[/link]",
            title="⚙️ Setup Required",
            border_style="yellow",
        ))


if __name__ == "__main__":
    app()
