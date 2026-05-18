from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import litellm

ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = Path(__file__).with_name("best_agent_prompt.txt")
RESULTS_PATH = ROOT / "results.tsv"
RUN_LOG = ROOT / "run.log"
MODEL = os.getenv("EVOLVE_MODEL", os.getenv("AGENT_MODEL", "openai/gpt-4.1-mini"))
API_BASE = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")

MUTATION_SYSTEM_PROMPT = """You are optimizing a system prompt for a restaurant management AI agent.
The agent runs a 30-day simulation. Score = net_profit - penalties. -100,000 = bankrupt.
Here is the current prompt and score history. Propose ONE targeted change to the prompt
that is most likely to improve the score. Explain your reasoning in one sentence, then
output only the full new prompt. Do not change the JSON output format instructions."""


@dataclass
class EvalResult:
    score: float
    output: str
    worst_scenarios: list[tuple[str, float]]


def ensure_results_file() -> None:
    if not RESULTS_PATH.exists():
        RESULTS_PATH.write_text("timestamp\tscore\tdescription\tprompt_hash\n", encoding="utf-8")


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def read_results() -> list[dict[str, str]]:
    ensure_results_file()
    rows: list[dict[str, str]] = []
    with RESULTS_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def best_score(rows: list[dict[str, str]]) -> float:
    if not rows:
        return 0.0
    return max(float(row["score"]) for row in rows if row.get("score"))


def append_result(score: float, description: str, current_hash: str) -> None:
    ensure_results_file()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with RESULTS_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([timestamp, f"{score:.2f}", description, current_hash])


def run_command(cmd: list[str], log_path: Path | None = None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    output = proc.stdout + proc.stderr
    if log_path is not None:
        log_path.write_text(output, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{output}")
    return output


def parse_final_score(output: str) -> float:
    match = re.search(r"Final score:\s*(-?\d+(?:\.\d+)?)", output)
    if match:
        return float(match.group(1))
    match = re.search(r"\*\*\* FINAL SCORE:\s*([-\d,]+(?:\.\d+)?) \*\*\*", output)
    if match:
        return float(match.group(1).replace(",", ""))
    raise ValueError("Could not parse final score from output")


def parse_scenario_table(output: str) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    capture = False
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("Scenario") and "Avg Score" in stripped:
            capture = True
            continue
        if capture and not stripped:
            break
        if capture and re.match(r"^-{5,}$", stripped):
            continue
        if capture and stripped.startswith("─"):
            break
        if capture and stripped and not stripped.startswith("Games played"):
            match = re.match(r"([A-Za-z_]+)\s+(-?\d+(?:\.\d+)?)", stripped)
            if match:
                rows.append((match.group(1), float(match.group(2))))
    return sorted(rows, key=lambda item: item[1])


def mutate_prompt(current_prompt: str, history: list[dict[str, str]], last_change: str) -> tuple[str, str]:
    history_text = "\n".join(
        f"{row['timestamp']}\tscore={row['score']}\tdesc={row['description']}\thash={row['prompt_hash']}"
        for row in history[-20:]
    ) or "No prior runs."
    response = litellm.completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": MUTATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Current prompt:\n{current_prompt}\n\n"
                    f"Recent score history:\n{history_text}\n\n"
                    f"What changed vs previous run:\n{last_change}\n"
                ),
            },
        ],
        temperature=0.4,
        max_tokens=2200,
        timeout=20,
        api_base=API_BASE,
    )
    content = (response.choices[0].message.content or "").strip()
    if content.startswith("```"):
        parts = content.split("\n", 1)
        content = parts[1] if len(parts) > 1 else content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
    lines = [line for line in content.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("Mutation response missing prompt body")
    description = lines[0].strip()
    new_prompt = "\n".join(lines[1:]).strip()
    return description, new_prompt


def maybe_commit(score: float) -> None:
    subprocess.run(["git", "add", "agents/best_agent_prompt.txt", "results.tsv"], cwd=ROOT, check=False)
    subprocess.run(
        ["git", "commit", "-m", f"evolve: improve prompt to {score:.2f}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def run_single_game(url: str, scenario: str, seed: int) -> EvalResult:
    output = run_command(
        [sys.executable, "-m", "agents.best_agent", "--scenario", scenario, "--seed", str(seed), "--url", url],
        log_path=RUN_LOG,
    )
    return EvalResult(score=parse_final_score(output), output=output, worst_scenarios=[])


def run_full_eval(url: str, scenarios: str, seeds: str, parallel: int) -> EvalResult:
    output = run_command([
        sys.executable,
        "-m",
        "agents.evaluate",
        "agents.best_agent",
        "--scenarios",
        scenarios,
        "--seeds",
        seeds,
        "--quiet",
        "--parallel",
        str(parallel),
        "--url",
        url,
    ])
    worst = parse_scenario_table(output)[:4]
    return EvalResult(score=parse_final_score(output), output=output, worst_scenarios=worst)


def evolve(args: argparse.Namespace) -> None:
    ensure_results_file()
    current_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    best_prompt = current_prompt
    previous_change = "Initial prompt."

    for iteration in range(1, args.iterations + 1):
        history = read_results()
        target_best = best_score(history)
        run_eval = iteration % args.full_eval_every == 0
        if run_eval:
            result = run_full_eval(args.url, args.eval_scenarios, args.eval_seeds, args.parallel)
            worst_text = ", ".join(f"{name}:{score:.0f}" for name, score in result.worst_scenarios) or "n/a"
            description = f"full_eval worst={worst_text}"
        else:
            result = run_single_game(args.url, args.scenario, args.seed)
            description = f"single_run {args.scenario} seed={args.seed}"

        current_hash = prompt_hash(current_prompt)
        append_result(result.score, description, current_hash)
        improved = result.score > target_best

        if improved:
            best_prompt = current_prompt
            maybe_commit(result.score)
        else:
            PROMPT_PATH.write_text(best_prompt, encoding="utf-8")
            current_prompt = best_prompt

        history = read_results()
        mutate_reason, next_prompt = mutate_prompt(current_prompt, history, previous_change)
        previous_change = mutate_reason
        current_prompt = next_prompt
        PROMPT_PATH.write_text(current_prompt, encoding="utf-8")

        print(
            f"iter={iteration} score={result.score:.2f} best_before={target_best:.2f} "
            f"improved={'yes' if improved else 'no'} hash={current_hash} change={mutate_reason}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous prompt evolution loop for best_agent")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--url", default=os.getenv("RESTBENCH_URL", "http://localhost:8001"))
    parser.add_argument("--scenario", default="baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval-scenarios",
        default="baseline,supply_crisis,tourist_season,renovation",
    )
    parser.add_argument("--eval-seeds", default="42,88,123")
    parser.add_argument("--full-eval-every", type=int, default=5)
    parser.add_argument("--parallel", type=int, default=4)
    args = parser.parse_args()
    evolve(args)


if __name__ == "__main__":
    main()
