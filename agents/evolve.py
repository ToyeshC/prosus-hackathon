"""AutoResearch-style prompt evolution loop for best_agent.

Pattern: Karpathy AutoResearch / FlorisFok/AutoResearchYC.
1. Eval current prompt across (scenarios x seeds).
2. Feed score table + worst-case failures into mutator LLM.
3. Mutator outputs a single targeted prompt change.
4. Re-eval. Keep prompt if improved else restore best.
5. Repeat until iterations exhausted or target score hit.

Improvements over v1:
- Defaults to full multi-seed eval (single-seed cherry picks were noisy).
- Mutator receives per-scenario (avg, min, max), worst-game summary lines.
- Tracks best multi-seed avg as the true objective.
- Saves prompt lineage to agents/prompt_history/vN.txt.
- Cross-iteration learnings written to agents/learnings.md and fed to mutator.
- Early stop on --target score.
- Mutation model overridable (e.g. anthropic/claude-sonnet-4-6).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import litellm

ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = Path(__file__).with_name("best_agent_prompt.txt")
RESULTS_PATH = ROOT / "results.tsv"
LEARNINGS_PATH = ROOT / "agents" / "learnings.md"
LINEAGE_DIR = ROOT / "agents" / "prompt_history"
RUN_LOG = ROOT / "run.log"

DEFAULT_MUTATION_MODEL = os.getenv(
    "EVOLVE_MODEL",
    os.getenv("AGENT_MODEL", "openai/gpt-4.1"),
)
API_BASE = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")

MUTATION_SYSTEM_PROMPT = """You are evolving a system prompt for a restaurant-management LLM agent.
The agent plays a 30-day simulation. Score = net_profit - penalties. Bankruptcy = -100,000.
You will see:
- The current prompt.
- A score history with description (scenario worst case, etc.).
- Per-scenario score breakdown for the latest run.
- Accumulated learnings (free-form notes you have written across iterations).

Rules:
1. Output ONE targeted change. Do not rewrite wholesale.
2. Your first line MUST be a short imperative sentence (<=120 chars) describing the change.
3. Then a blank line, then the full new prompt body.
4. Never alter the output-format instructions (must still tell the LLM to return JSON list of actions).
5. Prefer additions/clarifications targeting the worst-scoring scenario.
6. If recent attempts regressed, REVERT and try a different direction; do not stack regressions.
7. Keep prompt under ~3000 chars."""


@dataclass
class GameResult:
    scenario: str
    seed: int
    score: float
    status: str
    days: int
    profit: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "seed": self.seed,
            "score": self.score,
            "status": self.status,
            "days": self.days,
            "profit": self.profit,
        }


@dataclass
class EvalResult:
    avg_score: float
    games: list[GameResult]
    scenario_summary: dict[str, dict[str, float]] = field(default_factory=dict)

    def description(self) -> str:
        worst = sorted(self.games, key=lambda g: g.score)[:3]
        worst_text = ", ".join(f"{g.scenario}/{g.seed}:{g.score:.0f}" for g in worst)
        return f"eval avg={self.avg_score:.0f} worst=[{worst_text}]"

    def scenario_lines(self) -> str:
        lines = []
        for scenario, stats in sorted(self.scenario_summary.items()):
            lines.append(
                f"  {scenario:<20} avg={stats['avg']:>8.0f} "
                f"min={stats['min']:>8.0f} max={stats['max']:>8.0f}"
            )
        return "\n".join(lines)

    def worst_lines(self, n: int = 5) -> str:
        worst = sorted(self.games, key=lambda g: g.score)[:n]
        return "\n".join(
            f"  scenario={g.scenario:<18} seed={g.seed} score={g.score:>8.0f} "
            f"profit={g.profit:>8.0f} days={g.days} status={g.status}"
            for g in worst
        )


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
    scores = [float(row["score"]) for row in rows if row.get("score")]
    return max(scores) if scores else float("-inf")


def append_result(score: float, description: str, current_hash: str) -> None:
    ensure_results_file()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with RESULTS_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([timestamp, f"{score:.2f}", description, current_hash])


def run_command(cmd: list[str], log_path: Path | None = None, timeout: int = 1800) -> str:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=timeout,
    )
    output = proc.stdout + proc.stderr
    if log_path is not None:
        log_path.write_text(output, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{output[-2000:]}")
    return output


FINAL_SCORE_RE = re.compile(r"\*\*\* FINAL SCORE:\s*([-\d,]+(?:\.\d+)?)")
COMPLETED_LINE_RE = re.compile(
    r"Completed \d+/\d+:\s*(\S+)\s+seed=(\d+)\s+→\s+([-\d,]+(?:\.\d+)?)"
)


def parse_eval_output(output: str) -> EvalResult:
    games: list[GameResult] = []
    for line in output.splitlines():
        m = COMPLETED_LINE_RE.search(line)
        if m:
            scenario = m.group(1)
            seed = int(m.group(2))
            score = float(m.group(3).replace(",", ""))
            games.append(GameResult(
                scenario=scenario,
                seed=seed,
                score=score,
                status="ok" if score > -100_000 else "bankrupt",
                days=30 if score > -100_000 else 0,
                profit=score,
            ))
    final_match = FINAL_SCORE_RE.search(output)
    if final_match:
        avg = float(final_match.group(1).replace(",", ""))
    elif games:
        avg = sum(g.score for g in games) / len(games)
    else:
        raise ValueError("Could not parse eval output:\n" + output[-1500:])

    scenario_summary: dict[str, dict[str, float]] = {}
    by_scenario: dict[str, list[float]] = {}
    for g in games:
        by_scenario.setdefault(g.scenario, []).append(g.score)
    for scenario, scores in by_scenario.items():
        scenario_summary[scenario] = {
            "avg": sum(scores) / len(scores),
            "min": min(scores),
            "max": max(scores),
            "count": len(scores),
        }
    return EvalResult(avg_score=avg, games=games, scenario_summary=scenario_summary)


def run_full_eval(url: str, scenarios: str, seeds: str, parallel: int, team_name: str) -> EvalResult:
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
        "--team-name",
        team_name,
    ])
    RUN_LOG.write_text(output, encoding="utf-8")
    return parse_eval_output(output)


def load_learnings() -> str:
    if not LEARNINGS_PATH.exists():
        return "(no learnings yet)"
    text = LEARNINGS_PATH.read_text(encoding="utf-8")
    return text[-4000:] or "(empty)"


def append_learning(line: str) -> None:
    LEARNINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    with LEARNINGS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"- [{timestamp}] {line}\n")


def save_lineage(iteration: int, prompt_text: str, score: float | None) -> Path:
    LINEAGE_DIR.mkdir(parents=True, exist_ok=True)
    score_str = "init" if score is None else f"{score:.0f}"
    path = LINEAGE_DIR / f"v{iteration:03d}_score_{score_str}.txt"
    path.write_text(prompt_text, encoding="utf-8")
    return path


def mutate_prompt(
    current_prompt: str,
    history: list[dict[str, str]],
    last_change: str,
    latest_eval: EvalResult | None,
    model: str,
) -> tuple[str, str]:
    history_text = "\n".join(
        f"  {row['timestamp']}  score={row['score']}  desc={row['description'][:120]}"
        for row in history[-15:]
    ) or "(no prior runs)"
    if latest_eval is not None:
        eval_section = (
            f"Latest avg score: {latest_eval.avg_score:.0f}\n"
            f"Per-scenario:\n{latest_eval.scenario_lines()}\n"
            f"Worst games:\n{latest_eval.worst_lines()}\n"
        )
    else:
        eval_section = "(no eval yet)"

    user_msg = (
        f"=== Current prompt ===\n{current_prompt}\n\n"
        f"=== Score history (last 15) ===\n{history_text}\n\n"
        f"=== Last mutation description ===\n{last_change}\n\n"
        f"=== Latest evaluation ===\n{eval_section}\n\n"
        f"=== Accumulated learnings ===\n{load_learnings()}\n"
    )

    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": MUTATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.55,
        max_tokens=2500,
        timeout=60,
        api_base=API_BASE,
    )
    content = (response.choices[0].message.content or "").strip()
    if content.startswith("```"):
        parts = content.split("\n", 1)
        content = parts[1] if len(parts) > 1 else content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
    if "\n\n" in content:
        first_para, rest = content.split("\n\n", 1)
        description = first_para.strip().splitlines()[0].strip()
        new_prompt = rest.strip()
    else:
        lines = [line for line in content.splitlines() if line.strip()]
        if len(lines) < 2:
            raise ValueError("Mutation response missing body")
        description = lines[0].strip()
        new_prompt = "\n".join(lines[1:]).strip()
    return description[:250], new_prompt


def maybe_commit(score: float) -> None:
    subprocess.run(
        ["git", "add", "agents/best_agent_prompt.txt", "results.tsv",
         "agents/learnings.md", "agents/prompt_history"],
        cwd=ROOT,
        check=False,
    )
    subprocess.run(
        ["git", "commit", "-m", f"evolve: avg score {score:.2f}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def evolve(args: argparse.Namespace) -> None:
    ensure_results_file()
    current_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    best_prompt = current_prompt
    history_before = read_results()
    best_eval_score = best_score(history_before)
    if best_eval_score == float("-inf"):
        best_eval_score = float("-inf")
    previous_change = "(initial prompt)"
    last_eval: EvalResult | None = None

    save_lineage(0, current_prompt, None)

    for iteration in range(1, args.iterations + 1):
        print(f"\n=== iter {iteration}/{args.iterations} model={args.model} ===")
        result = run_full_eval(args.url, args.eval_scenarios, args.eval_seeds, args.parallel, args.team_name)
        last_eval = result
        current_hash = prompt_hash(current_prompt)
        append_result(result.avg_score, result.description(), current_hash)
        improved = result.avg_score > best_eval_score
        print(
            f"iter={iteration} avg={result.avg_score:.0f} "
            f"best_before={best_eval_score:.0f} improved={'YES' if improved else 'no'}"
        )
        print(result.scenario_lines())

        if improved:
            best_prompt = current_prompt
            best_eval_score = result.avg_score
            maybe_commit(result.avg_score)
            append_learning(
                f"KEEP @iter{iteration} avg={result.avg_score:.0f}: {previous_change}"
            )
        else:
            append_learning(
                f"REVERT @iter{iteration} avg={result.avg_score:.0f} (best={best_eval_score:.0f}): "
                f"{previous_change}"
            )
            PROMPT_PATH.write_text(best_prompt, encoding="utf-8")
            current_prompt = best_prompt

        save_lineage(iteration, current_prompt, result.avg_score)

        if best_eval_score >= args.target:
            print(f"\n*** target {args.target} reached. Best avg: {best_eval_score:.0f} ***")
            break

        if iteration == args.iterations:
            break

        try:
            history = read_results()
            mutate_reason, next_prompt = mutate_prompt(
                current_prompt, history, previous_change, last_eval, args.model
            )
        except Exception as exc:
            print(f"  mutation failed: {exc}")
            append_learning(f"mutation_error @iter{iteration}: {exc}")
            continue
        previous_change = mutate_reason
        current_prompt = next_prompt
        PROMPT_PATH.write_text(current_prompt, encoding="utf-8")
        print(f"  mutation: {mutate_reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoResearch prompt evolution loop")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--url", default=os.getenv("RESTBENCH_URL", "http://52.48.183.209:8001"))
    parser.add_argument(
        "--eval-scenarios",
        default="baseline,supply_crisis,tourist_season,renovation",
    )
    parser.add_argument("--eval-seeds", default="42,88,123")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--target", type=float, default=60000.0,
                        help="Stop early when best avg score >= target")
    parser.add_argument("--model", default=DEFAULT_MUTATION_MODEL,
                        help="LLM model for prompt mutation")
    parser.add_argument("--team-name", default="RelayEvolve")
    args = parser.parse_args()
    evolve(args)


if __name__ == "__main__":
    main()
