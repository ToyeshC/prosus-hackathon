# Relay — Design Document

Prosus × AISO AI Agent Hackathon, 18 May 2026.
Team **Relay** — autonomous restaurant operator on RestBench.

This document explains *why* the agent looks the way it does. It maps each design choice back to one of the four judging criteria from the Prosus cheat sheet: **Agent Autonomy**, **Impact**, **Technical Quality**, **Creativity**.

---

## 1. The problem

> *"Your challenge is to build an AI agent capable of independently managing and optimizing a simulated restaurant environment."*
> — Cheat sheet, p. 2

RestBench simulates 30 days of an Italian restaurant. Each turn the agent receives an `observation` (cash, inventory, suppliers, weather, customer signals, alerts) and emits zero or more `tool calls` (order, set price, set staff, marketing, happy hour, daily special, save notes). The simulator runs a full service day, deducts costs, and returns the next observation.

Score = `net_profit − penalties`.
Bankruptcy = −100,000.
Final evaluation = 10 scenarios × 3 *random* seeds (6 scenarios are hidden until 16:00).

The cheat sheet states the deciding axis bluntly:

> *"An agent that scores +40k on baseline but bankrupts on a hidden scenario will lose to a steady +5k everywhere."*

So the goal isn't peak performance on any one scenario — it's **consistent positive score across a matrix of conditions we partly don't know in advance**.

---

## 2. Architecture in one diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│  TURN LOOP (best_agent.strategy)                                    │
│                                                                     │
│   observation, notes(JSON)                                          │
│        │                                                            │
│        ▼                                                            │
│   ┌──────────────────────┐                                          │
│   │  Safety Rules        │  inventory days-of-stock, bankruptcy     │
│   │  (pure heuristic)    │  guard, stockout flags, staff floor,     │
│   │                      │  pending-order dedup, supplier math      │
│   └──────────┬───────────┘                                          │
│              │ safety_actions, safety_meta                          │
│              ▼                                                      │
│   ┌──────────────────────┐                                          │
│   │  LLM Decision Layer  │  fires ONLY on: day 1, new alerts,       │
│   │  (gated, optional)   │  reputation Poor/Fair, low cash,         │
│   │                      │  critical stockout, declining + alert    │
│   └──────────┬───────────┘                                          │
│              │ llm_actions                                          │
│              ▼                                                      │
│   ┌──────────────────────┐                                          │
│   │  Filter & Merge      │  strip LLM actions that contradict       │
│   │                      │  safety rules; clip prices to [0.8x,1.2x]│
│   └──────────┬───────────┘                                          │
│              ▼                                                      │
│   ┌──────────────────────┐                                          │
│   │  save_notes()        │  persist updated JSON state              │
│   └──────────┬───────────┘                                          │
│              ▼                                                      │
│         submit_all → end_turn                                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

The architecture is deliberately **safety-rules-first, LLM-second**. The cheat sheet's #1 rule is "Survive every game". Safety rules can't be overridden by an LLM hallucination — they execute first, the LLM gets the post-safety state, and the filter drops anything that contradicts.

**Files implementing the layers:**

| Layer | File | Notes |
|---|---|---|
| Safety rules | `agents/best_agent.py::safety_rules()` | inventory, bankruptcy, stockout flags, staff floor, supplier math |
| LLM gate + call | `agents/best_agent.py::llm_decide()` | gated; loads prompt from `best_agent_prompt.txt` |
| LLM filter | `agents/best_agent.py::filter_llm_actions()` | strip set_menu/place_order from LLM; clamp prices |
| Notes memory | `agents/best_agent.py::update_notes_state()`, `compact_notes()` | structured JSON < 4000 chars |
| Game runner | `agents/runner.py` | unchanged starter-kit HTTP driver |
| Eval harness | `agents/evaluate.py` | parallel scenario × seed matrix |
| Evolution loop | `agents/evolve.py` | AutoResearch-style prompt mutation |
| DevConsole | `agents/best_agent.py::DevConsole` | JSONL + HTML trace per game, written to `artifacts/` |

---

## 3. Safety rules — what they protect against

Every safety rule maps to a specific failure mode called out in the cheat sheet or AGENT_CONTRACT.

| Rule | Code reference | Failure it prevents |
|---|---|---|
| Inventory days-of-stock estimate + reorder when below threshold | `safety_rules()` block iterating active ingredients | "Most failures come from teams missing `dishes_unavailable_at` and `pending_orders`." (cheat sheet p. 5) |
| Pending-order dedup (don't re-order what's already in transit) | `dedup_pending_order()` | Cash drain from double-ordering |
| Supplier delivery-day math | `next_valid_delivery()` | Ordering from a Wed-only supplier on Thursday → 6-day gap |
| Stockout flagging from `dishes_unavailable_at` | `collect_stockout_flags()` | Reputation spirals from repeated stockouts |
| Bankruptcy guard at cash < 3000 | inside `safety_rules()` | The bankruptcy = −100,000 cliff |
| Staff floor based on reputation band | `staff_floor()` | Walkouts → ghost reviews → reputation collapse |
| Order cap when cash < cost + 1000 | inside the `for ingredient` loop | Avoid drawing cash below the buffer |
| Banned-supplier list updated from alerts | `update_notes_state()` | Repeated orders from a halted supplier wasting money |

These are all pure Python — no LLM, no randomness. They execute every turn, take ~10ms, and never fail because of an API timeout.

**Judging hook:** *Technical Quality.* Robust under unseen scenarios because the safety rules don't depend on scenario-specific knowledge — they react to observable state (`inventory`, `pending_orders`, `cash`).

---

## 4. The LLM judgment layer

The LLM is the *override* layer, not the primary controller. By default the heuristic decides everything. The LLM is invited only when something looks off:

```python
should_call_llm = any([
    int(observation.get("day", 1)) == 1,           # initial framing
    bool(current_alerts - previous_alerts),         # scenario change announced
    observation.get("reputation_band") in {"Poor", "Fair"},  # recovery mode
    safe_float(observation.get("cash")) < 5000,    # cash crunch
    bool(service_summary.get("dishes_unavailable_at")),       # stockouts happening
    observation.get("customer_trend") == "Declining" and current_alerts,
])
```

When the gate fires:
- The system prompt is loaded from `agents/best_agent_prompt.txt` (mutated by the evolve loop).
- The user message is the full observation + the structured notes JSON.
- Temperature = 0.2 — low randomness, we want consistent overrides.
- 20-second timeout. On exception, the agent falls back silently to the heuristic.

The LLM is restricted to a **subset of tools**:
- `set_price`, `set_marketing_spend`, `run_happy_hour`, `offer_daily_special`, `set_staff_level`.
- It cannot `place_order` (inventory is heuristic-only).
- It cannot `set_menu` (heuristic owns the menu plan).
- It cannot `save_notes` (the agent owns the schema).

`filter_llm_actions()` enforces this — anything outside the allowed set is silently dropped, and any LLM action that conflicts with the safety layer (e.g. cuts staff below the floor, prices outside the 0.8x–1.2x band) is dropped.

**Judging hook:** *Agent Autonomy & Creativity.* The LLM isn't a thin wrapper around `if/else` — it sees the full state and reasons over it within tight guardrails. But we don't pretend it makes every decision: the LLM has been observed to *hurt* aggregate score when it touches stable states, so we narrow its scope.

---

## 5. Memory — three layers

The cheat sheet flags: *"Use save_notes as memory. LLMs have no memory between turns otherwise."* But it understates the case — we use three memory layers, not one:

1. **In-game (per turn):** `notes` field in observation, serialised JSON ≤ 4000 chars. Schema is structured (banned_suppliers, stockout_history, ingredient_usage EMA, daily_revenue, scenario_guess, price_adjustments, rules). Compacted via `compact_notes()` — oldest list entries dropped first.
2. **Cross-iteration (between evolve runs):** `agents/learnings.md` — free-form append-only log written by the evolve loop ("KEEP @iter4 avg=34344: clarify renovation marketing 0-50 EUR"). The mutator reads the last 4000 chars on every iteration, so prior decisions inform the next mutation.
3. **Per-game trace (for the pitch + post-hoc analysis):** `agents/best_agent.py::DevConsole` writes a JSONL per game to `artifacts/{team}_{scenario}_{seed}_{stamp}.jsonl` and an HTML viewer to the same path with `.html`. Each entry records the day's reasoning summary, the safety actions taken, the LLM actions, the final actions, and the snapshot of notes. This is what we'll demo on stage if we make the top 5.

**Judging hook:** *Technical Quality.* Memory is engineered, not bolted on.

---

## 6. AutoResearch — the evolution loop

The single most differentiating piece of the project, and inspired by:
- karpathy/autoresearch — the original pattern of LLM-mutated experiments.
- FlorisFok/AutoResearchYC — applied to YC-Bench, beat the previous leaderboard #1 by $500k.

`agents/evolve.py` runs the loop:

```
while iterations remaining:
    1. Eval current best_agent_prompt.txt across (4 scenarios × 3 seeds) = 12 games in parallel.
    2. Parse final avg score, per-scenario breakdown, worst games.
    3. Append (timestamp, score, description, prompt_hash) to results.tsv.
    4. If score improved vs best-so-far: KEEP prompt, write to learnings.md, git-commit.
       Else: REVERT to best, write a REGRESSION line to learnings.md.
    5. Snapshot the (possibly reverted) prompt to agents/prompt_history/vNNN_score_X.txt.
    6. Call mutator LLM with:
         - the current prompt
         - the score history (last 15 rows from results.tsv)
         - the per-scenario eval breakdown
         - the worst-game lines
         - the accumulated learnings.md
       Instruct: "Propose ONE targeted change. First line = description. Then the new prompt."
    7. Write the mutated prompt to best_agent_prompt.txt.
    8. If best_score ≥ target (60k by default): stop early.
```

**What this gives us:**
- **Autonomy.** The loop runs unattended overnight, generates ~20 prompt variants per night with no human in the loop after launch.
- **Reproducibility.** Every prompt version is in `agents/prompt_history/` with the score it produced. Roll back at any time.
- **Honesty.** When a mutation regresses, the loop reverts and writes that fact to `learnings.md`. The mutator sees its own failures and tries a different direction.

**Recent trajectory (iter 1–4):**

| Iter | Score | Mutation |
|---|---|---|
| 1 | 30,217 | initial prompt |
| 2 | 32,847 | "minimise marketing 0–50 EUR during renovation" → kept |
| 3 | 21,815 | "reduce staff to minimum during renovation if cash < 2500" → reverted |
| 4 | 34,344 | "maintain only 5–6 active dishes during renovation" → kept |

Per-iteration cost: ~12 games × ~6 LLM calls × $0.003 ≈ **$0.20/iter** through the Prosus litellm proxy.

**Judging hook:** *Creativity.* This is an unusual application of the Karpathy AutoResearch pattern to an agentic benchmark. We didn't just hand-tune a prompt — we let the LLM tune its own prompt with bounded iteration.

---

## 7. Parallel exploration — Phase A

Beyond the sequential evolve loop, we run **5 parallel sub-agents in worktrees**, each owning a single hypothesis:

| Variant | Hypothesis | Code locus |
|---|---|---|
| A1 — narrow gate | LLM is over-touching stable states; restrict gate to (day 1 ∨ new alert ∨ reputation Poor ∨ critical stockout ∨ cash < 4k) | `llm_decide()` |
| A2 — renovation hardcoded | Heuristic doesn't claim the post-renovation satisfaction bonus; force two-phase staff/price/marketing schedule | `heuristic_decide()` |
| A3 — supply-crisis defensive | Avoid suppliers with lead_time > 1 during supply_crisis; raise inventory target_days to 12; cap orders when cash < 5k | `safety_rules()` + `supplier_options_for_ingredient()` |
| A4 — tourist peak | Staff to 10 + prices 1.18× during surge; cut back hard on collapse | `heuristic_decide()` |
| A5 — conservative prompt | Brand-new prompt telling LLM to default to `[]` and only override on narrow triggers | `best_agent_prompt.txt` |

Each sub-agent runs in an isolated git worktree, evaluates on **seeds {7, 55, 99}** (the dev-leaderboard seeds), reports a single score back to the main thread. The main thread merges the winning variant(s) into the trunk branch.

Why isolation? The five hypotheses are independent and partially conflicting (A2 and A4 both edit `heuristic_decide`). Worktrees let us test them in parallel without merge conflicts, then cherry-pick the deltas of the winners.

**Judging hook:** *Technical Quality + Creativity.* Treating prompt/heuristic engineering as a parallel search rather than a sequential edit is a non-obvious tooling choice.

---

## 8. Scenario adaptation

The 4 known scenarios are: `baseline`, `supply_crisis`, `tourist_season`, `renovation`. 6 hidden scenarios unlock at 16:00. Both the heuristic and the LLM key off the same `detect_scenario(observation, notes)` function:

```python
alerts = " ".join(observation.get("alerts", [])).lower()
if "tourist" in alerts or "surge" in alerts:     return "tourist_season"
if "renov" in alerts or "reduced seating" in alerts:  return "renovation"
if "supplier" in alerts or "outage" in alerts
   or "halted" in alerts or "disruption" in alerts:  return "supply_crisis"
return notes.get("scenario_guess", "baseline")
```

For each scenario the heuristic applies different parameter values (target_days inventory, price multiplier, marketing level, staff floor). The LLM prompt has explicit scenario-conditional sections covering the 4 known scenarios *plus* hypothesised hidden ones (`health_scare`, `viral_moment`, `economic_downturn`, `staff_walkout`, `competitor_opens`, `premium_shift`, `inflation`).

The hypothesis: even when a hidden scenario isn't named, its **alert text** will contain enough keywords for the prompt's general scenario-detection block to map it to a known archetype.

**Judging hook:** *Impact.* The agent isn't memorising — it's reacting to text in `alerts` and observable drift in the data, exactly as the cheat sheet asks.

---

## 9. Robustness

The final eval uses 3 **random** seeds, different from the dev seeds (cheat sheet p. 7). Defending against seed-specific overfit:

- Dev eval default: `--seeds 42,88,123` (3 seeds × 4 scenarios = 12 games per iteration, fits in ~12 min).
- Lock-in eval (before 16:00 freeze): `--seeds 42,88,123,7,55,99` (6 seeds × 4 scenarios = 24 games).
- Acceptance: avg ≥ 50k AND no single cell < 0 AND `min/mean` ratio > 0.4. Reject any prompt that improves avg while worsening min.
- During the 16:00–17:00 eval phase: re-run any cell whose score is more than 1σ below the cell-median. Best score per cell counts (cheat sheet rule).

Per cheat sheet p. 7: *"Don't keep iterating on prompts in this phase. Lock your agent before the matrix opens. Use the time to run."*

---

## 10. What we'd build next

(Cheat sheet p. 6 asks: *"Please provide a short summary of what you would envision as the next steps."*)

1. **Heuristic param tuning.** The evolve loop currently only mutates the prompt. The biggest score lever is probably the heuristic parameters (target_days, reorder_trigger_days, price_mult lookup, marketing levels). A follow-up evolve loop that mutates a small `params.json` instead of the prompt would unlock another tier.
2. **Cross-game priors.** Each game starts blind. Loading a per-scenario priors JSON into `save_notes` on day 1 — distilled from prior runs — would seed the LLM with what worked last time.
3. **Online seed sampling.** Instead of fixed dev seeds, sample uniformly from a larger pool each iteration. Reduces seed-specific overfit.
4. **Bandit-style scenario selection.** Spend more eval budget on the scenarios with highest variance (currently renovation and supply_crisis seed=42).
5. **Multi-model ensembling.** Cheap reasoning model for state classification, frontier model only when the gate identifies a high-stakes decision.

---

## 11. Pitch talking points

If we make the top 5 (announced 18:00), we have 5 minutes + 3 Q&A.

**The one thing we're proud of:**
> The AutoResearch evolution loop ran unattended overnight, evolving the strategy prompt across 20+ iterations. The mutator LLM reads the score history, sees its own past regressions in `learnings.md`, and proposes one targeted change per iteration. It's not just black-box optimisation — every iteration writes a one-sentence English description ("clarify marketing should be 0-50 EUR during renovation"). The result is a paper trail of *why* the prompt looks the way it does.

**One hard scenario we handled well:**
> `renovation` started at 18k avg — the worst scenario. After two evolve iterations the prompt learned that the second half of the renovation window is the *opportunity*, not just the cost. Prices nudge up after day 13 to harvest the satisfaction bonus the simulator pays for surviving the disruption. Renovation now contributes ~30k+.

**Architecture in one breath:**
> Safety rules in pure Python protect against bankruptcy and stockouts; an LLM judges price, staffing, and promotion overrides through a narrow gate; an AutoResearch loop tunes the LLM's own prompt overnight. Everything else is starter-kit.

---

## Appendix — file map

```
agents/
  best_agent.py            # production agent — safety + LLM + filter + notes
  best_agent_prompt.txt    # LLM system prompt, mutated by evolve.py
  evolve.py                # AutoResearch loop
  evaluate.py              # multi-scenario × multi-seed harness (starter kit)
  runner.py                # HTTP game driver (starter kit, untouched)
  learnings.md             # cross-iteration freeform notes for the mutator
  prompt_history/v*.txt    # snapshot per iter (lineage)

archive/                   # moved here for cleanliness — not on the production path
  baselines/do_nothing.py     naive_rule.py    llm_template.py
            starter_template.py compare.py
  notes/auto.txt
  prompt_history/             # early lineage snapshots

artifacts/                 # per-game DevConsole traces (jsonl + html)

results.tsv                # evolve log (timestamp, score, description, hash)
run.log                    # last subprocess stdout/stderr
.env                       # OPENAI_BASE_URL + key (gitignored)

DESIGN.md                  # this file
README.md  AGENT_CONTRACT.md  STRATEGY_GUIDE.md   # starter-kit docs
Prosus_AISO AI Agent Hackathon Cheat Sheet .pdf
```

---

*Relay — Prosus × AISO Hackathon, 18 May 2026.*
