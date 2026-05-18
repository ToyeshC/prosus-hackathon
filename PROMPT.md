# RELAY — Prosus Hackathon Agent Context

## What You Are Building
An autonomous restaurant management agent that maximizes profit
over a multi-day simulation. Decisions every turn: pricing, staffing,
inventory orders, supplier selection, promotional strategy.
The simulation is dynamic — scenarios change, supply shocks happen,
demand fluctuates. You must survive AND optimize.

## The Only Rule That Cannot Break
Never go bankrupt. Bankruptcy = -100,000 score, automatic loss.
All other decisions are subordinate to solvency.
When in doubt, be conservative with cash.

---

## Two-Layer Architecture

### Layer 1 — Safety Rules (execute BEFORE any LLM call, every turn)
Hardcoded Python. The LLM cannot override these.

    SAFETY_RULES = {
        "min_cash_buffer": 500,
        "reorder_threshold": 2,        # reorder any item with ≤2 days stock
        "max_staff_overage": 1.2,      # never staff >120% of expected covers
        "bankrupt_prevention": True,   # if cash < buffer, cancel non-essential orders
    }

Execute in this order every turn:
1. Read full state: inventory, cash, pending_orders,
   dishes_unavailable_at, alerts, staffing
2. Read save_notes
3. Apply safety rules → produce constrained action set
4. Pass constrained state to LLM for strategic optimization
5. Execute LLM decisions
6. Write observations to save_notes

### Layer 2 — LLM Strategic Layer
Model: GPT-4o for strategic decisions
Model: GPT-4o-mini for cheap turn-by-turn classification if needed
Budget: $100 provided — more than enough for 30+ games

Handles: dynamic pricing, supplier selection, staffing optimization,
scenario-specific strategy, promotion decisions, recovery planning.

---

## Non-Negotiables to Read Every Turn

Most failure modes come from ignoring these:

- dishes_unavailable_at — NEVER order or price a dish that is
  currently unavailable. Doing so wastes money silently.
- pending_orders — current demand load. Use to calibrate staffing.
- alerts — FREE scenario intelligence. Every scenario change
  announces itself here. Read this first every turn.
- cash — track trajectory, not just current value.
  Is it trending up or down day over day?

---

## Memory Architecture (save_notes)

The LLM has no memory between API calls. save_notes IS your memory.
Write every turn. Max 500 tokens. Summarize aggressively.
Token bloat in notes kills performance.

Structure to maintain:

    === GAME MEMORY ===
    Day: {n}
    Scenario: {baseline|supply_crisis|tourist_season|renovation|unknown}
    Cash: {current} | Trend: {+/-X per day}
    Last 3 days profit: {values}

    === SCENARIO SIGNALS ===
    Alerts received: {list}
    Unavailable items: {list}
    Supplier status: {who is up/down}

    === LEARNED PATTERNS ===
    Peak hours: {observed}
    Price experiments: {item, delta, demand_response}
    Optimal staff level: {observed at what demand}
    Reorder cadence: {what works}

    === ACTIVE STRATEGY ===
    Mode: {conservative|optimize|grow}
    Pricing stance: {hold|raise|lower} because {reason}
    Focus item: {highest margin available item}

    === RULES LEARNED ===
    {add specific things that caused loss or gain as discovered}

---

## Scenario Detection (from alerts field)

- "supplier" + "unavailable" → supply_crisis
- "tourist" OR demand spike >30% day-over-day → tourist_season
- "renovation" OR capacity_reduced → renovation
- No alerts + stable demand → baseline
- Sudden demand DROP, no supply issue → competitor or health_scare
- Sudden demand SPIKE, no season flag → viral_moment
- Staff availability drops mid-game → staff_walkout

---

## Strategy by Scenario

### baseline
- Test pricing incrementally: +5%, watch demand response
- Staff to 90% of expected covers
- Maintain 3-day safety stock
- Focus on highest-margin menu items

### supply_crisis
- Check dishes_unavailable_at immediately
- Remove unavailable items from active menu
- Raise prices on available scarce items (cost pass-through)
- Switch to alternative suppliers immediately
- Reduce perishable orders aggressively
- Protect cash — this scenario gets expensive fast

### tourist_season
- Staff UP proactively before demand hits, not reactively
- Raise prices 15-25% — tourists are price-inelastic
- Maximize throughput on high-margin items
- Order more inventory in advance
- Highest-profit scenario — do not be conservative here

### renovation
- Reduce staff proportionally to reduced capacity
- Cut inventory orders proportionally
- Focus on cash preservation
- Survival scenario, not growth

### competitor_opens (likely hidden)
- Demand drops 20-30% without supply issue
- Do NOT race to bottom on price
- Differentiate on customer satisfaction
- Reduce staffing to match lower demand
- Hold quality, reduce waste aggressively

### health_scare (likely hidden)
- Demand collapses fast
- Pull flagged items immediately
- Reduce orders drastically
- Skeleton staff only
- Cash preservation until recovery signal in alerts

### viral_moment (likely hidden)
- Demand spikes without warning
- Staff up immediately
- Do not raise prices aggressively (reputation risk)
- Order more inventory immediately — stockout risk is high

### economic_downturn (likely hidden)
- Price sensitivity increases
- Shift toward lower-price, higher-volume items
- Reduce staffing slightly
- Maintain margins through cost reduction not price increase

### staff_walkout (likely hidden)
- Capacity drops suddenly
- Reduce accepted orders to match available staff
- Do not promise what you cannot deliver
- Prioritize highest-margin orders

---

## Restaurant Domain Knowledge

### Pricing
- Lunch demand is inelastic (people must eat) — hold or raise
- Dinner demand is elastic (people choosing) — test carefully
- Tourists: fully inelastic — raise confidently
- Downturn: price sensitive — hold, play volume

### Inventory
- Waste = silent killer. Perishables expire = pure loss.
- Safety stock: avg_daily_demand × lead_time + 1 day buffer
- Never stockout on your top 3 revenue items
- Supply crisis: prioritize high-margin items in limited stock

### Staffing
- Understaffing → satisfaction drops → future demand drops (compounding)
- Overstaffing off-peak → direct wage hemorrhage
- Staff to 80% of expected peak, +1 buffer for high-demand periods

### Cash Flow
- Cash ≠ profit. Watch cash flow separately from P&L.
- Always maintain minimum buffer. Never let cash approach zero.
- Track daily trajectory. Two consecutive down days = switch to conservative.

---

## Evaluation Structure

### Stage 1 — Leaderboard (determines top 5)
- 10 scenarios × 3 seeds = 30 games total
- 4 known scenarios available all day for development
- 6 hidden scenarios unlock at approximately 16:00
- Best score per (scenario, seed) cell counts — retries allowed
- Max 5 concurrent games per team
- Partial matrix ranks BELOW complete matrix at any score
- Complete all 30 games before 17:00. This is the #1 priority.

Score landmarks:
- do_nothing: −100,000 (bankrupt by day ~14)
- naive_rule: ~−15,000 (survives, never optimizes)
- 0 or positive: beat naïve baseline
- +15,000 to +35,000: competitive for top 5

Key rule: Consistency beats variance.
An agent scoring +5,000 on all 30 games beats one scoring
+40,000 on some and going bankrupt on others.

### Stage 2 — Pitch (top 5 only, 18:30–19:15)
- 5 minutes pitch + 3 minutes Q&A
- Judged on: autonomy, impact, technical quality, creativity
- Prepare: architecture diagram, one thing you're proud of,
  one unexpected scenario you handled well
- Dev console showing agent reasoning per turn = devastating pitch asset

---

## Pre-Submission Dreaming (Before 16:00)

Run the agent on all 4 known scenarios multiple times with different
seeds. After each run, distill into system prompt:
- What decisions produced the highest profit
- What caused unexpected loss
- What thresholds triggered problems
- What patterns emerged in demand/supply

Update hardcoded safety rules based on observed failure points.
This is the dreaming layer applied to agent development.
The agent arrives at evaluation with baked-in experience.

Example learned rules to add:
- On day 1, always be conservative — demand pattern unknown
- Never order more than 3x daily average for perishables
- If cash drops 2 consecutive days, switch to conservative mode
- tourist_season: raise prices on day 1, do not wait for confirmation
- supply_crisis: switch supplier on the SAME turn the alert fires

---

## What a Good Agent Does (Master List)

1. Read dishes_unavailable_at EVERY turn — never serve unavailable items
2. Read pending_orders EVERY turn — use to calibrate staffing
3. Read alerts EVERY turn — free scenario intelligence
4. Safety rules run FIRST before any LLM call
5. Use save_notes for memory — no memory exists otherwise
6. Detect scenario from alerts, switch strategy immediately
7. Dynamic pricing based on observed demand elasticity
8. Staff proactively before surges, not reactively after
9. Never let cash approach the bankruptcy threshold
10. Reason from current state not hardcoded day numbers
11. Complete all 30 games — partial matrix is worse than low scores
12. Day 1 of any game: observe first, optimize second

---

## Tech Stack
- Language: Python
- LLM: GPT-4o (strategic layer) via provided OpenAI API key
- Optional: GPT-4o-mini for fast intent classification
- Memory: save_notes in-simulation + enriched system prompt pre-evaluation
- Observability: skip for submission run (latency risk during 30-game matrix)

---

## Time Plan

09:00–09:30  Read repo, understand API schema, run do_nothing
09:30–11:00  Build safety rule layer, get non-bankrupt agent on baseline
11:00–12:00  Add save_notes memory, add LLM strategic layer
12:00–13:00  Lunch + test on all 4 known scenarios
13:00–15:00  Add scenario detection, iterate prompts, run simulations
15:00–15:30  Run dreaming cycle: distill learnings into system prompt
15:30–16:00  Lock agent, no more changes, prepare for matrix
16:00–17:00  Run all 30 games, max 5 concurrent, monitor leaderboard