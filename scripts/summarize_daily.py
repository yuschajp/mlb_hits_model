#!/usr/bin/env python3
"""
summarize_daily.py

Reads docs/dashboard_data.json and generates a plain-English daily
briefing using a local Ollama model (no external API calls, no cost).

Usage:
    python3 scripts/summarize_daily.py
    python3 scripts/summarize_daily.py --model llama3.1:8b
    python3 scripts/summarize_daily.py --save

Requires Ollama running locally (`ollama serve` or `brew services start ollama`)
and a model already pulled (e.g. `ollama pull llama3.1:8b`).
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

DASHBOARD_PATH = Path("docs/dashboard_data.json")
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.1:8b"


def load_dashboard_data(path: Path) -> dict:
    if not path.exists():
        print(f"ERROR: {path} not found. Run this from the repo root.", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def build_prompt(data: dict) -> str:
    """
    Build a compact, structured prompt from the dashboard data.
    Only pulls in the fields that matter for a daily readout —
    keeps the prompt small so a 7B model handles it fast and reliably.
    """
    generated_at = data.get("generated_at", "unknown date")
    overview_rows = data.get("overview", {}).get("rows", [])
    picks = data.get("picks", {})

    lines = [f"Date: {generated_at}", "", "Model performance overview:"]
    for row in overview_rows:
        model = row.get("model", "unknown")
        today = row.get("today", 0)
        graded = row.get("graded", 0)
        metric = row.get("metric", "")
        value = row.get("value", "")
        baseline = row.get("baseline", "")
        edge_pct = row.get("edge_pct")
        verdict = row.get("verdict", "")
        note = row.get("note", "")

        edge_str = f"{edge_pct:.1f}% edge" if isinstance(edge_pct, (int, float)) else "edge n/a"
        line = (
            f"- {model}: {today} today, {graded} graded, "
            f"{metric}={value} (baseline {baseline}, {edge_str}), verdict: {verdict}"
        )
        if note:
            line += f" [{note}]"
        lines.append(line)

    n_singles = len(picks.get("singles", []))
    n_parlays = len(picks.get("parlays", []))
    n_rr = len(picks.get("round_robins", []))
    lines.append("")
    lines.append(
        f"Today's queued picks: {n_singles} singles, {n_parlays} parlays, "
        f"{n_rr} round robins."
    )

    prompt = (
        "You are a terse quant analyst summarizing a sports prediction "
        "platform's daily model performance for the platform's operator. "
        "Given the structured data below, write a short daily briefing "
        "(4-6 sentences). Call out: (1) which models are beating their "
        "baseline and by how much, (2) any model underperforming its "
        "baseline, (3) any model with zero graded results yet worth noting, "
        "and (4) whether there are any picks queued today. Be direct and "
        "factual, no hype, no bullet points, no preamble.\n\n"
        + "\n".join(lines)
    )
    return prompt


def call_ollama(prompt: str, model: str) -> str:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("response", "").strip()
    except Exception as e:
        print(f"ERROR calling Ollama: {e}", file=sys.stderr)
        print(
            "Is Ollama running? Try: brew services start ollama",
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Generate daily briefing via local LLM.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the summary to logs/daily_summary.log (appends, timestamped)",
    )
    args = parser.parse_args()

    data = load_dashboard_data(DASHBOARD_PATH)
    prompt = build_prompt(data)
    summary = call_ollama(prompt, args.model)

    print("=" * 60)
    print(f" Daily Briefing — {data.get('generated_at', 'unknown date')}")
    print("=" * 60)
    print(summary)
    print("=" * 60)

    if args.save:
        log_path = Path("logs/daily_summary.log")
        log_path.parent.mkdir(exist_ok=True)
        with open(log_path, "a") as f:
            f.write(f"\n[{data.get('generated_at')}]\n{summary}\n")
        print(f"\nSaved to {log_path}")


if __name__ == "__main__":
    main()
