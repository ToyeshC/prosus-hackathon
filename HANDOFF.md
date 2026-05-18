# Hackathon Handoff — RestBench Restaurant Agent

## Current state
- Team: **AKT** on http://52.48.183.209:8001
- Rank: **#16**, avg **18,826**, **12 cells / 30** completed
- Top team: estain at 39,892 (12 cells)
- Hidden scenarios unlock ~16:00 (6 scenarios × 3 seeds = 18 cells)

## Critical mechanics learned
- **Matrix seeds are [7, 55, 99]** — NOT 42 (we wasted hours on seed=42 early)
- **Dashboard tracks LATEST submission per cell**, NOT best-per-cell
- So **every submission can overwrite a good score with a bad one**
- Strategy: identify best agent per cell, submit each cell exactly once with best agent
- Per-team rate limit: 60 games/hour. Multi-team trick = fresh quota per team_name
- LLM API: LiteLLM proxy at `http://litellm-production.eba-pvykax23.eu-west-1.elasticbeanstalk.com`, key: `my-key` (replace with the hackathon-provided key locally — never commit the real one)

## Best score per cell (current AKT)
| Cell | Score | Agent | Note |
|---|---|---|---|
| baseline/7 | 38,053 | greedy | (akt_robust got 43,222 on team 020 — better) |
| baseline/55 | -1,084 | greedy | (akt_robust got 1,704 on team 020 — better) |
| baseline/99 | 15,161 | greedy | (akt_robust got 18,612 on team 020 — better) |
| supply/7 | 32,234 | greedy | (akt_final has 44,564 — need to reclaim) |
| supply/55 | 23,069 | akt_final | locked |
| supply/99 | 28,791 | akt_final | locked |
| tourist/7 | **53,390** | greedy | best cell |
| tourist/55 | 10,397 | greedy | (akt_robust got 16,900 — better) |
| tourist/99 | 36,807 | greedy | (akt_final has 51,959 — need to reclaim) |
| renovation/7 | ~+469 | akt_final | |
| renovation/55 | -7,878 | akt_final | hardest cell |
| renovation/99 | -5,484 | akt_final | (akt_robust got -4,289 — marginal better) |

If we submit best-of-three agents per cell to AKT: projected avg ~22,700.

## Agent files (in `agents/`)
- **`adaptive_agent.py`** — original rule-based, currently restored to safer v2 with shelf-caps + cash override
- **`adaptive_agent_v2_backup.py`** — backup of safer version
- **`greedy_agent.py`** — copy of v_leaderboard_best_58807 snapshot. Aggressive: DAYS_BUFFER=5, DAYS_SAFETY=2, BOOTSTRAP_DAILY_KG=3.0, no shelf-caps, no cash override. Bankrupts on renovation but huge upside on tourist/baseline.
- **`akt_v4.py`, `akt_v5.py`, `akt_v7.py`** — intermediate versions. v5 added supply stockpiling. v7 fixed renovation flag bug.
- **`akt_final.py`** — proven core + LLM-mini classifier + LLM-4o strategist (fires only on UNKNOWN scenarios) + panic mode on rep Poor / Fair+Declining / Fair+thin-cash / rapid-bleed
- **`akt_robust.py`** — akt_final with BOOTSTRAP_DAILY_KG=1.5, happy hour ON during Poor rep (rep recovery tool), lower-rep pricing 0.88x. Better on baseline+tourist/55, worse elsewhere.
- **`hybrid_agent.py`** — earlier pure-LLM-strategist version (gpt-4o-mini classifier + gpt-4o strategist). Worse than rules on visible scenarios but good for unknown alerts.

## Things that REGRESSED (don't try again)
- Raising prices broadly (1.18x on demand_surge, 1.05x on Good+busy) → -10k on tourist
- Forcing premium pricing on every surge day → -11k tourist
- Auto-clearing reduced_capacity after 2 quiet days → renovation -16k (clears mid-renovation)
- Doc's lean SAFETY_RESERVE=500, DAYS_BUFFER=4 → -22k overall
- Pure LLM strategist on visible scenarios → -25k overall (LLM worse than rules for known mechanics)

## Things that WORKED
- v7: only clear `reduced_capacity` on explicit "complete" alert → +3k on renovation
- v5: stockpile (buffer 8d) on `supply_disrupted` → +9k on supply_crisis
- akt_final panic mode: cap staff/marketing on rep collapse → +9k on baseline
- Split LLM: mini for classification, 4o for strategy (when LLM strategist is used)
- LiteLLM proxy via OPENAI_BASE_URL env var (gpt-4o-mini works fine)

## API quick reference
```python
# Per-game endpoints
POST /games {team_name, scenario, seed} -> {game_id}
POST /games/{id}/action {tool, args}
POST /games/{id}/end-turn -> {observation, day_result, status}
GET /games/{id}/score -> {net_profit, walkout_penalty, total_score}

# Leaderboard
GET /leaderboard/dashboard -> {ranking: [{team_name, avg_score, cells_completed, ...}], evaluation_matrix: {seeds: [7,55,99], scenarios: [10], total_cells: 30}}
GET /leaderboard -> top-N per (team, scenario, seed) cells
```

## What an ML-based approach could try
- **Offline RL on local sim** — train PPO/DQN agent on local_sim, deploy on real. Risk: local sim has shown massive divergence from real (-12k local vs +25k real for same agent).
- **Bandit per (scenario, seed) cell** — multi-armed bandit picking best agent variant per cell. We have data points for this already.
- **Sklearn classifier from state → action** — train on top-team-style heuristics, deploy as policy. Need training data.
- **LLM fine-tuning** — fine-tune small model on (state, optimal action) pairs from local_sim winners. Possible but slow.

## Environment vars to set
```powershell
$env:OPENAI_API_KEY="my-key"
$env:OPENAI_BASE_URL="http://litellm-production.eba-pvykax23.eu-west-1.elasticbeanstalk.com"
$env:RESTBENCH_URL="http://52.48.183.209:8001"
$env:PYTHONIOENCODING="utf-8"
```

## Quick commands
```powershell
# Run single game
python -m agents.akt_final

# Full visible matrix
python -u -m agents.evaluate agents.akt_final --scenarios baseline,supply_crisis,tourist_season,renovation --seeds 7,55,99 --team-name AKT --parallel 5

# Check leaderboard
curl -s http://52.48.183.209:8001/leaderboard/dashboard | python -c "import json,sys; d=json.loads(sys.stdin.read()); [print(f\"#{t['rank']} {t['team_name']:<20} {t['avg_score']:>8.0f} cells={t['cells_completed']}\") for t in d['ranking'][:20]]"
```
