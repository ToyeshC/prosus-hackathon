# Hackathon Handoff — RestBench AKT Agent

## Current state (final)
- Team: **AKT** on http://52.48.183.209:8001
- **Avg: 33,038**, 12 cells/30 completed, 0 bankruptcies
- Peak team: **050 at 33,612**
- Top 5 leaderboard: Et-al-Agents 56,320, caps17 53,694, a_bunch_of_idiots 51,470

## 🏆 Winning agent: `toyesh_v2.py` + `gpt-4.1`

Hybrid architecture (forked from ToyeshC/prosus-hackathon `toyesh` branch, then improved):
- Rule-based safety + ordering + staffing (deterministic)
- LLM judgment only on triggers (new alerts, rep Poor/Fair, low cash, stockouts)
- LLM **cannot place orders** — only sets price/marketing/HH/special/staff
- Sophisticated `delivery_day` math per supplier schedule
- Multi-gate reorder logic with shelf-life caps

### Our 4 improvements over Toyesh's base
1. **Bankruptcy guard cash<3000 → 2500** (more ordering room)
2. **Stable-trend pricing 1.06 → 1.03** (avoid aggressive-pricing trap)
3. **LLM-mini scenario classifier** for hidden/novel alerts
4. **Force `gpt-4.1`** as the model (proven +3-5k over gpt-4o, mini, and gpt-5)

## Run command (final)

```powershell
$env:OPENAI_API_KEY="<litellm-key>"
$env:OPENAI_BASE_URL="http://litellm-production.eba-pvykax23.eu-west-1.elasticbeanstalk.com"
$env:RESTBENCH_URL="http://52.48.183.209:8001"
$env:AGENT_MODEL="gpt-4.1"
$env:PYTHONIOENCODING="utf-8"

python -u -m agents.evaluate agents.toyesh_v2 \
  --scenarios baseline,supply_crisis,tourist_season,renovation \
  --seeds 7,55,99 --team-name AKT --parallel 1
```

**Important: `--parallel 1`** — higher parallelism causes 429 rate-limit errors that kill the avg (one errored cell = -100k drag).

## Files to ship

| File | Purpose |
|---|---|
| `agents/toyesh_v2.py` | The winning agent |
| `agents/toyesh_v2_prompt.txt` | LLM judgment prompt |
| `agents/runner.py` | HTTP client (from starter kit) |
| `agents/evaluate.py` | Multi-scenario eval harness |
| `requirements.txt` | `openai>=1.0` |

## Critical mechanics learned
- **Matrix seeds: [7, 55, 99]** — NOT 42 (lost hours on seed=42)
- **Dashboard tracks LATEST per cell** — submission overwrites
- **Per-team rate limit**: 60 games/hour
- **Multi-team trick**: fresh quota per `--team-name`
- **gpt-4.1 wins** over gpt-4o, gpt-4o-mini, gpt-5, gpt-5-mini, o3-mini on this task

## Score progression (today)
```
18,826  rule-based adaptive_agent (starting point, rank #16)
22,819  AKT-fresh-diag akt_smart + gpt-4o + bankruptcy guard
25,335  toyesh_agent + gpt-4o on team 060
29,540  toyesh_v2 (4 improvements) + gpt-4o on team 020
30,907  v2 retry variance
32,387  v2 + gpt-4o-mini on team 080
33,612  v2 + gpt-4.1 on team 050 (peak)
33,038  AKT locked at v2+gpt-4.1 ← FINAL OFFICIAL
```

## Things that REGRESSED (don't try)
- Aggressive pricing >1.15x (drove customers away)
- Auto-clearing reduced_capacity on quiet days (clears mid-renovation)
- Doc's lean SAFETY_RESERVE=500, DAYS_BUFFER=4 (cash bleeds during disruption)
- Pure LLM strategist on visible scenarios
- Multi-feature stacking (akt_pro: -14k)
- gpt-4o-mini for forecasting (variance)
- o3-mini, gpt-5 (don't support response_format properly → fell to heuristic mode)

## Things that WORKED
- gpt-4.1 model (+3-5k)
- Bankruptcy guard cash<2500 (+29k from preventing -100k cells)
- v7 renovation fix: only clear flag on explicit "complete" alert (+3k)
- Supply stockpiling buffer 8d on `supply_disrupted` (+9k)
- Panic mode on rep collapse (+9k baseline)
- LLM judgment ONLY on triggers (Toyesh's insight — most days use heuristics)
- LLM forbidden from `place_order` — orders are 100% deterministic rules

## API quick reference
```python
GET /scenarios -> [{name, display_name, description, difficulty}]
GET /leaderboard/dashboard -> {ranking, evaluation_matrix}
POST /games {team_name, scenario, seed}
POST /games/{id}/action {tool, args}
POST /games/{id}/end-turn -> {observation, day_result, status}
GET /games/{id}/score -> {net_profit, ..., total_score}
```

## Hidden scenarios (unlock at TBD)
- black_swan, feast_or_famine, health_scare, inflation, premium_pivot, silent_drift
- toyesh_v2 has LLM classifier for novel alerts → should generalize
- 18 cells (6 × 3 seeds) become available when unlocked
