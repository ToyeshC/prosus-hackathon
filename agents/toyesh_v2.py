from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from agents.runner import DEFAULT_URL, run_game

MODEL = os.getenv("AGENT_MODEL", "gpt-4o")
PROMPT_FILE = Path(__file__).with_name("toyesh_v2_prompt.txt")
NOTE_LIMIT = 4000

_client = None
def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),
                         base_url=os.environ.get("OPENAI_BASE_URL") or None)
    return _client
DAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
DAY_TO_INDEX = {name: idx for idx, name in enumerate(DAY_NAMES)}
REPUTATION_ORDER = {
    "Poor": 0,
    "Fair": 1,
    "Good": 2,
    "Very Good": 3,
    "Excellent": 4,
}
WEATHER_BAD = {"rainy", "stormy"}


@dataclass
class OrderCandidate:
    ingredient: str
    supplier: str
    quantity_kg: float
    unit_price: float
    delivery_day: int

    @property
    def cost(self) -> float:
        return self.quantity_kg * self.unit_price


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_notes(raw_notes: Any) -> dict[str, Any]:
    if not raw_notes:
        return {}
    if isinstance(raw_notes, dict):
        return raw_notes
    if not isinstance(raw_notes, str):
        return {}
    try:
        parsed = json.loads(raw_notes)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def load_prompt() -> str:
    try:
        return PROMPT_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "Return only a JSON array of actions."


def inventory_map(observation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        item["ingredient"]: item
        for item in observation.get("inventory", [])
        if item.get("ingredient")
    }


def menu_book_map(observation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        dish["name"]: dish
        for dish in observation.get("menu_book", [])
        if dish.get("name")
    }


def active_dishes(observation: dict[str, Any]) -> list[dict[str, Any]]:
    book = menu_book_map(observation)
    names = observation.get("active_menu") or [
        dish["name"] for dish in observation.get("menu_book", []) if dish.get("is_active")
    ]
    return [book[name] for name in names if name in book]


def ingredient_usage_yesterday(observation: dict[str, Any]) -> dict[str, float]:
    usage: dict[str, float] = {}
    service_summary = observation.get("service_summary") or {}
    sold = service_summary.get("dishes_sold", {}) or {}
    dishes = menu_book_map(observation)
    for dish_name, count in sold.items():
        recipe = dishes.get(dish_name, {})
        for item in recipe.get("ingredients", []):
            ingredient = item.get("ingredient")
            qty = safe_float(item.get("quantity_kg"))
            if ingredient and count:
                usage[ingredient] = usage.get(ingredient, 0.0) + qty * safe_float(count)
    return usage


def pending_by_ingredient(observation: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for order in observation.get("pending_orders", []):
        ingredient = order.get("ingredient")
        if ingredient:
            result.setdefault(ingredient, []).append(order)
    return result


def next_valid_delivery(today_day: int, supplier: dict[str, Any]) -> int | None:
    delivery_days = supplier.get("delivery_days", [])
    if not delivery_days:
        return None
    lead = int(supplier.get("lead_time_days", 0))
    earliest = today_day + lead
    allowed = {DAY_TO_INDEX[day] for day in delivery_days if day in DAY_TO_INDEX}
    if not allowed:
        return None
    for day in range(earliest, earliest + 14):
        weekday = (day - 1) % 7
        if weekday in allowed:
            return day
    return None


def dedup_pending_order(
    observation: dict[str, Any],
    ingredient: str,
    today_day: int,
    max_arrival_gap: int = 2,
) -> bool:
    for order in observation.get("pending_orders", []):
        if order.get("ingredient") != ingredient:
            continue
        delivery_day = int(order.get("delivery_day", 10**9))
        if delivery_day <= today_day + max_arrival_gap:
            return True
    return False


def supplier_options_for_ingredient(
    observation: dict[str, Any],
    notes: dict[str, Any],
    ingredient: str,
    today_day: int,
) -> list[OrderCandidate]:
    banned = set(notes.get("banned_suppliers", []) or [])
    options: list[OrderCandidate] = []
    for supplier in observation.get("supplier_catalog", []):
        name = supplier.get("name")
        price = safe_float((supplier.get("ingredients") or {}).get(ingredient), default=-1)
        if not name or price <= 0 or name in banned:
            continue
        delivery_day = next_valid_delivery(today_day, supplier)
        if delivery_day is None:
            continue
        options.append(OrderCandidate(
            ingredient=ingredient,
            supplier=name,
            quantity_kg=0.0,
            unit_price=price,
            delivery_day=delivery_day,
        ))
    options.sort(key=lambda item: (item.unit_price, item.delivery_day, item.supplier))
    return options


def choose_supplier(
    observation: dict[str, Any],
    notes: dict[str, Any],
    ingredient: str,
    today_day: int,
) -> OrderCandidate | None:
    options = supplier_options_for_ingredient(observation, notes, ingredient, today_day)
    if not options:
        return None
    soon = [option for option in options if option.delivery_day <= today_day + 3]
    return soon[0] if soon else options[0]


def staff_floor(observation: dict[str, Any]) -> int:
    rep = observation.get("reputation_band", "Good")
    floor = 7 if REPUTATION_ORDER.get(rep, 2) <= REPUTATION_ORDER["Fair"] else 3
    return floor


def current_menu_price_map(observation: dict[str, Any]) -> dict[str, float]:
    result = {}
    for dish in observation.get("menu_book", []):
        if dish.get("name") is not None:
            result[dish["name"]] = safe_float(dish.get("current_price"))
    return result


def cheapest_supplier_costs(observation: dict[str, Any], notes: dict[str, Any], today_day: int) -> dict[str, float]:
    costs: dict[str, float] = {}
    for supplier in observation.get("supplier_catalog", []):
        name = supplier.get("name")
        if name in set(notes.get("banned_suppliers", []) or []):
            continue
        delivery_day = next_valid_delivery(today_day, supplier)
        if delivery_day is None:
            continue
        for ingredient, price in (supplier.get("ingredients") or {}).items():
            cost = safe_float(price, default=0)
            if cost <= 0:
                continue
            if ingredient not in costs or cost < costs[ingredient]:
                costs[ingredient] = cost
    return costs


def heuristic_decide(observation: dict[str, Any], notes: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    today_day = int(observation.get("day", 1))
    day_of_week = observation.get("day_of_week", "")
    cash = safe_float(observation.get("cash"))
    rep = observation.get("reputation_band", "Good")
    trend = observation.get("customer_trend", "Stable")
    weather = str(observation.get("weather_today", "")).lower()
    active_names = list(observation.get("active_menu", []))
    active_set = set(active_names)
    dishes = menu_book_map(observation)
    inventory = inventory_map(observation)
    pending = pending_by_ingredient(observation)
    cost_map = cheapest_supplier_costs(observation, notes, today_day)
    current_prices = current_menu_price_map(observation)
    scenario = detect_scenario(observation, notes)
    service_summary = observation.get("service_summary") or {}

    target_staff = int(observation.get("staff_level", 8))
    if rep in {"Poor", "Fair"}:
        target_staff = max(target_staff, 7)
    elif int(observation.get("days_remaining", 0)) <= 2 and rep in {"Good", "Very Good", "Excellent"}:
        target_staff = max(5, target_staff - 2)
    elif int(observation.get("days_remaining", 0)) <= 3 and rep in {"Good", "Very Good", "Excellent"}:
        target_staff = max(5, target_staff - 1)
    elif observation.get("day_of_week") in {"Friday", "Saturday", "Sunday"}:
        target_staff = 7
    elif scenario == "renovation":
        target_staff = 6
    else:
        target_staff = 5 if cash < 9000 else 6
    if target_staff != int(observation.get("staff_level", 8)):
        actions.append({"tool": "set_staff_level", "args": {"level": target_staff}})

    def dish_coverage(dish_name: str) -> float:
        dish = dishes.get(dish_name, {})
        coverage = 999.0
        for item in dish.get("ingredients", []):
            ingredient = item.get("ingredient")
            qty = safe_float(item.get("quantity_kg"))
            if not ingredient or qty <= 0:
                continue
            stock = safe_float(inventory.get(ingredient, {}).get("total_kg")) + sum(
                safe_float(order.get("quantity_kg")) for order in pending.get(ingredient, [])
            )
            coverage = min(coverage, stock / qty if qty > 0 else 999.0)
        return coverage

    menu_plan = [
        ("Pizza Margherita", 1.0),
        ("Spaghetti Carbonara", 1.0),
        ("Chicken Caesar Salad", 1.0),
        ("Mushroom Risotto", 1.0),
        ("Grilled Salmon", 2.0),
        ("Pizza Pepperoni", 2.0),
        ("Mushroom Tagliatelle", 2.0),
        ("Chicken Parmesan", 2.0),
    ]
    desired_menu = [
        dish for dish, min_cov in menu_plan
        if dish in dishes and dish_coverage(dish) >= min_cov
    ][:7]
    if len(desired_menu) < 5:
        ranked = sorted(
            ((dish_coverage(name), name) for name in dishes),
            reverse=True,
        )
        for _, name in ranked:
            if name not in desired_menu:
                desired_menu.append(name)
            if len(desired_menu) >= 5:
                break
    if len(desired_menu) >= 5 and set(desired_menu) != active_set:
        actions.append({"tool": "set_menu", "args": {"dishes": desired_menu}})
        active_names = desired_menu
        active_set = set(desired_menu)

    price_mult = 1.0
    if rep in {"Poor", "Fair"}:
        price_mult = 1.0
    elif scenario == "tourist_season" or trend == "Growing":
        price_mult = 1.15
    elif observation.get("day_of_week") in {"Friday", "Saturday", "Sunday"}:
        price_mult = 1.12
    elif trend == "Stable":
        price_mult = 1.03  # v2: lowered from 1.06 — aggressive pricing on stable hurts demand
    elif weather in WEATHER_BAD:
        price_mult = 0.97

    for dish_name in active_names:
        dish = dishes.get(dish_name)
        if not dish:
            continue
        base_price = safe_float(dish.get("base_price"))
        target_price = round(min(base_price * 1.2, max(base_price * 0.8, base_price * price_mult)), 2)
        if abs(target_price - current_prices.get(dish_name, base_price)) >= 0.05:
            actions.append({"tool": "set_price", "args": {"dish": dish_name, "price": target_price}})

    marketing = 0
    if cash < 4000:
        marketing = 0
    elif int(observation.get("days_remaining", 0)) <= 4:
        marketing = 0
    elif scenario == "tourist_season":
        marketing = 260
    elif weather in WEATHER_BAD or observation.get("day_of_week") in {"Monday", "Tuesday"}:
        marketing = 80
    else:
        marketing = 120
    actions.append({"tool": "set_marketing_spend", "args": {"amount": marketing}})

    happy_streak = int(notes.get("happy_hour_streak", 0) or 0)
    if (
        marketing <= 120
        and weather in WEATHER_BAD | {"rainy", "stormy"}
        and rep != "Excellent"
        and happy_streak < 3
    ) or (
        observation.get("day_of_week") in {"Monday", "Tuesday"}
        and rep != "Excellent"
        and happy_streak < 3
        and cash >= 4000
    ):
        actions.append({"tool": "run_happy_hour", "args": {}})

    best_special = None
    best_margin = -10**9
    for dish_name in active_names:
        dish = dishes.get(dish_name)
        if not dish:
            continue
        ingredient_cost = 0.0
        reliable = True
        for item in dish.get("ingredients", []):
            ingredient = item.get("ingredient")
            qty = safe_float(item.get("quantity_kg"))
            if not ingredient:
                continue
            ingredient_cost += cost_map.get(ingredient, 0.0) * qty
            stock = safe_float(inventory.get(ingredient, {}).get("total_kg")) + sum(
                safe_float(order.get("quantity_kg")) for order in pending.get(ingredient, [])
            )
            if stock < qty * 4:
                reliable = False
        margin = safe_float(dish.get("current_price") or dish.get("base_price")) - ingredient_cost
        if reliable and margin > best_margin:
            best_margin = margin
            best_special = dish_name
    if best_special:
        actions.append({"tool": "offer_daily_special", "args": {"dish": best_special}})

    return actions


def normalize_action(action: Any) -> dict[str, Any] | None:
    if not isinstance(action, dict):
        return None
    tool = action.get("tool")
    args = action.get("args", {})
    if not isinstance(tool, str) or not isinstance(args, dict):
        return None
    return {"tool": tool, "args": args}


def action_key(action: dict[str, Any]) -> tuple[str, str]:
    return action["tool"], json.dumps(action.get("args", {}), sort_keys=True)


def collect_stockout_flags(
    observation: dict[str, Any],
    notes: dict[str, Any],
) -> set[str]:
    flags = set(notes.get("critical_ingredients", []) or [])
    service_summary = observation.get("service_summary") or {}
    unavailable = service_summary.get("dishes_unavailable_at", {}) or {}
    dishes = menu_book_map(observation)
    history = notes.setdefault("stockout_history", {})
    for dish_name, hour in unavailable.items():
        if not isinstance(hour, (int, float)):
            continue
        history.setdefault(dish_name, []).append(observation.get("day"))
        history[dish_name] = history[dish_name][-5:]
        if hour >= 17:
            continue
        recipe = dishes.get(dish_name, {})
        ingredients = recipe.get("ingredients", [])
        if not ingredients:
            continue
        max_qty = max(safe_float(item.get("quantity_kg")) for item in ingredients)
        for item in ingredients:
            if safe_float(item.get("quantity_kg")) == max_qty and item.get("ingredient"):
                flags.add(item["ingredient"])
    return flags


def safety_rules(
    observation: dict[str, Any],
    notes: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    meta: dict[str, Any] = {
        "set_staff": False,
        "set_marketing": False,
        "happy_hour_blocked": False,
        "order_spend": 0.0,
        "order_ingredients": set(),
        "price_set": set(),
        "menu_set": False,
    }

    today_day = int(observation.get("day", 1))
    day_of_week = observation.get("day_of_week", "")
    cash = safe_float(observation.get("cash"))
    usage = ingredient_usage_yesterday(observation)
    inventory = inventory_map(observation)
    pending = pending_by_ingredient(observation)
    critical_ingredients = collect_stockout_flags(observation, notes)
    notes["critical_ingredients"] = sorted(critical_ingredients)

    service_summary = observation.get("service_summary") or {}
    walkouts = service_summary.get("walkout_band", "None")
    current_staff = int(observation.get("staff_level", 8))
    min_staff = staff_floor(observation)
    target_staff = current_staff
    if walkouts == "Many" and current_staff < 10:
        target_staff = max(target_staff, current_staff + 2)
    target_staff = max(target_staff, min_staff)

    if cash < 2500:
        target_staff = min(target_staff, max(3, current_staff - 2))
        target_staff = max(target_staff, min_staff)
        actions.append({"tool": "set_marketing_spend", "args": {"amount": 0}})
        meta["set_marketing"] = True
        meta["happy_hour_blocked"] = True

    if target_staff != current_staff:
        actions.append({"tool": "set_staff_level", "args": {"level": int(target_staff)}})
        meta["set_staff"] = True

    active_ingredients: set[str] = set()
    recipe_defaults: dict[str, float] = {}
    active_dish_list = active_dishes(observation)
    for dish in active_dishes(observation):
        for item in dish.get("ingredients", []):
            ingredient = item.get("ingredient")
            qty = safe_float(item.get("quantity_kg"))
            if ingredient:
                active_ingredients.add(ingredient)
                recipe_defaults[ingredient] = max(recipe_defaults.get(ingredient, 0.0), qty)
    active_ingredients.update(critical_ingredients)
    historical_usage = notes.get("ingredient_usage", {}) or {}
    baseline_covers = safe_float(
        notes.get("baseline_covers")
        or service_summary.get("total_covers")
        or 90
    )
    per_dish_sales_guess = max(8.0, baseline_covers / max(3.0, len(active_dish_list) * 2.5))

    max_order_spend = 500.0 if cash < 2500 else float("inf")
    spent = 0.0

    for ingredient in sorted(active_ingredients):
        if dedup_pending_order(observation, ingredient, today_day):
            continue

        options = supplier_options_for_ingredient(observation, notes, ingredient, today_day)
        if not options:
            continue
        next_gap = min(option.delivery_day - today_day for option in options)

        daily_usage = usage.get(ingredient, 0.0)
        daily_usage = max(daily_usage, safe_float(historical_usage.get(ingredient)))
        if daily_usage <= 0 and ingredient in critical_ingredients:
            daily_usage = max(recipe_defaults.get(ingredient, 0.0) * 8.0, 0.5)
        if daily_usage <= 0 and ingredient in recipe_defaults:
            daily_usage = max(recipe_defaults.get(ingredient, 0.0) * per_dish_sales_guess, 0.8)
        if daily_usage <= 0:
            continue

        inv_kg = safe_float(inventory.get(ingredient, {}).get("total_kg"))
        pending_kg = sum(safe_float(order.get("quantity_kg")) for order in pending.get(ingredient, []))
        days_of_stock_left = inv_kg / daily_usage if daily_usage > 0 else 99.0
        pending_coverage = pending_kg / daily_usage if daily_usage > 0 else 0.0

        reorder_trigger_days = 6.0 if ingredient in critical_ingredients else 5.0
        if day_of_week in {"Friday", "Saturday"}:
            reorder_trigger_days += 2.0
        reorder_trigger_days += max(0.0, next_gap - 2.0)
        if days_of_stock_left + pending_coverage >= reorder_trigger_days:
            continue

        option = options[0]
        soon = [candidate for candidate in options if candidate.delivery_day <= today_day + 3]
        if soon:
            option = soon[0]

        supplier = next(
            (sup for sup in observation.get("supplier_catalog", []) if sup.get("name") == option.supplier),
            None,
        )
        min_qty = safe_float((supplier or {}).get("min_order_kg"), 1.0)
        if "supply_crisis" == notes.get("scenario_guess"):
            target_days = 10.0
        elif cash < 6000:
            target_days = 6.0
        else:
            target_days = 8.0
        if day_of_week in {"Friday", "Saturday"}:
            target_days += 2.0
        target_days += max(0.0, next_gap - 2.0)
        shelf_life = safe_float(inventory.get(ingredient, {}).get("shelf_life_days"), default=7.0)
        target_days = max(4.0, min(target_days, shelf_life - 1.0))
        needed_qty = max(daily_usage * target_days - inv_kg - pending_kg, 0.0)
        order_qty = max(min_qty, round(max(needed_qty, daily_usage * 2.5), 1))
        order = OrderCandidate(
            ingredient=ingredient,
            supplier=option.supplier,
            quantity_kg=order_qty,
            unit_price=option.unit_price,
            delivery_day=option.delivery_day,
        )

        if order.delivery_day > today_day + 3 and len(options) > 1:
            continue

        if cash < order.cost + 1000:
            continue
        if spent + order.cost > max_order_spend:
            continue

        actions.append({
            "tool": "place_order",
            "args": {
                "supplier": order.supplier,
                "ingredient": order.ingredient,
                "quantity_kg": order.quantity_kg,
            },
        })
        spent += order.cost
        meta["order_ingredients"].add(order.ingredient)

    meta["order_spend"] = spent
    return actions, meta


def llm_decide(
    observation: dict[str, Any],
    notes: dict[str, Any],
) -> list[dict[str, Any]]:
    api_key_present = any(
        os.getenv(name)
        for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AZURE_API_KEY")
    )
    if not api_key_present:
        return heuristic_decide(observation, notes)

    service_summary = observation.get("service_summary") or {}
    previous_alerts = set(notes.get("known_alerts", []) or [])
    current_alerts = set(observation.get("alerts", []) or [])
    should_call_llm = any([
        int(observation.get("day", 1)) == 1,
        bool(current_alerts - previous_alerts),
        observation.get("reputation_band") in {"Poor", "Fair"},
        safe_float(observation.get("cash")) < 5000,
        bool(service_summary.get("dishes_unavailable_at")),
        observation.get("customer_trend") == "Declining" and current_alerts,
    ])
    if not should_call_llm:
        return heuristic_decide(observation, notes)

    prompt = load_prompt()
    user_payload = {
        "day": observation.get("day"),
        "days_remaining": observation.get("days_remaining"),
        "observation": observation,
        "notes": notes,
    }
    try:
        response = _get_client().chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
            ],
            temperature=0.2,
            max_tokens=1200,
        )
        content = (response.choices[0].message.content or "").strip()
        if content.startswith("```"):
            parts = content.split("\n", 1)
            content = parts[1] if len(parts) > 1 else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        parsed = json.loads(content)
        if not isinstance(parsed, list):
            return []
        return [action for item in parsed if (action := normalize_action(item))]
    except Exception as exc:
        print(f"  LLM error day {observation.get('day')}: {exc}")
        return []


def filter_llm_actions(
    observation: dict[str, Any],
    notes: dict[str, Any],
    safety_actions: list[dict[str, Any]],
    llm_actions: list[dict[str, Any]],
    safety_meta: dict[str, Any],
) -> list[dict[str, Any]]:
    allowed_tools = {
        "set_price",
        "set_marketing_spend",
        "run_happy_hour",
        "offer_daily_special",
        "set_staff_level",
    }
    book = menu_book_map(observation)
    active_names = set(observation.get("active_menu", []))
    cash = safe_float(observation.get("cash"))
    today_day = int(observation.get("day", 1))
    kept: list[dict[str, Any]] = []
    seen = {action_key(action) for action in safety_actions}
    future_menu = None
    llm_order_spend = 0.0

    for action in llm_actions:
        tool = action["tool"]
        args = action["args"]
        if tool not in allowed_tools:
            continue
        key = action_key(action)
        if key in seen:
            continue

        if tool == "set_staff_level":
            if safety_meta["set_staff"]:
                continue
            level = args.get("level")
            if not isinstance(level, int):
                continue
            if level < staff_floor(observation) or not 3 <= level <= 15:
                continue

        elif tool == "set_marketing_spend":
            amount = safe_float(args.get("amount"), default=-1)
            if safety_meta["set_marketing"] or not 0 <= amount <= 500:
                continue
            if cash < 2500 and amount > 0:
                continue

        elif tool == "run_happy_hour":
            if safety_meta["happy_hour_blocked"]:
                continue

        elif tool == "set_price":
            dish = args.get("dish")
            price = safe_float(args.get("price"), default=-1)
            dish_info = book.get(dish)
            if not dish_info:
                continue
            base_price = safe_float(dish_info.get("base_price"))
            if price < base_price * 0.8 or price > base_price * 1.2:
                continue

        elif tool == "offer_daily_special":
            dish = args.get("dish")
            allowed_menu = future_menu or active_names
            if dish not in allowed_menu:
                continue

        kept.append(action)
        seen.add(action_key(action))

    return kept


_SCENARIO_CACHE = {}

def _llm_classify_scenario(alerts: list[str]) -> str | None:
    """LLM-mini fallback for novel/hidden scenario alerts."""
    if not alerts:
        return None
    key = tuple(sorted(alerts))
    if key in _SCENARIO_CACHE:
        return _SCENARIO_CACHE[key]
    try:
        client = _get_client()
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "Classify these restaurant alerts into ONE category. "
                    "Output JSON only: {\"scenario\": \"tourist_season|renovation|"
                    "supply_crisis|inflation|health_scare|black_swan|"
                    "premium_pivot|silent_drift|feast_or_famine|baseline\"}")},
                {"role": "user", "content": json.dumps(alerts)},
            ],
            temperature=0.0, max_tokens=50,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(r.choices[0].message.content or "{}")
        scenario = parsed.get("scenario", "baseline")
        _SCENARIO_CACHE[key] = scenario
        return scenario
    except Exception:
        return None


def detect_scenario(observation: dict[str, Any], notes: dict[str, Any]) -> str:
    alerts_list = observation.get("alerts", []) or []
    alerts = " ".join(alerts_list).lower()
    if "tourist" in alerts or "surge" in alerts:
        return "tourist_season"
    if "renov" in alerts or "reduced seating" in alerts:
        return "renovation"
    if "supplier" in alerts or "outage" in alerts or "halted" in alerts or "disruption" in alerts:
        return "supply_crisis"
    # v2: when keywords miss, try LLM classifier
    if alerts_list:
        scenario = _llm_classify_scenario(alerts_list)
        if scenario and scenario != "baseline":
            return scenario
    return notes.get("scenario_guess", "baseline")


def update_notes_state(
    observation: dict[str, Any],
    notes: dict[str, Any],
    actions: list[dict[str, Any]],
) -> dict[str, Any]:
    next_notes = dict(notes)
    next_notes["day"] = int(observation.get("day", 1))
    next_notes["scenario_guess"] = detect_scenario(observation, notes)
    next_notes["baseline_covers"] = int(
        next_notes.get("baseline_covers")
        or (observation.get("service_summary") or {}).get("total_covers")
        or 0
    )

    alert_list = list(dict.fromkeys((next_notes.get("known_alerts", []) or []) + list(observation.get("alerts", []))))
    next_notes["known_alerts"] = alert_list[-10:]

    banned = set(next_notes.get("banned_suppliers", []) or [])
    for alert in observation.get("alerts", []):
        lowered = alert.lower()
        if any(token in lowered for token in ("halted", "outage", "unavailable", "disruption", "delayed")):
            for supplier in observation.get("supplier_catalog", []):
                name = supplier.get("name", "")
                if name and name.lower() in lowered:
                    banned.add(name)
    next_notes["banned_suppliers"] = sorted(banned)[:10]

    revenues = list(next_notes.get("daily_revenue", []) or [])
    yesterday = safe_float(observation.get("yesterday_revenue"))
    if yesterday > 0:
        revenues.append(round(yesterday, 2))
    next_notes["daily_revenue"] = revenues[-15:]

    previous_usage = dict(next_notes.get("ingredient_usage", {}) or {})
    latest_usage = ingredient_usage_yesterday(observation)
    for ingredient, qty in latest_usage.items():
        prior = safe_float(previous_usage.get(ingredient))
        previous_usage[ingredient] = round(qty if prior <= 0 else (0.6 * prior + 0.4 * qty), 2)
    next_notes["ingredient_usage"] = {
        key: previous_usage[key]
        for key in sorted(previous_usage)[:20]
    }

    staff_level = observation.get("staff_level")
    for action in actions:
        if action["tool"] == "set_staff_level":
            staff_level = action["args"].get("level", staff_level)
    next_notes["staff_level"] = int(staff_level or 0)

    price_adjustments = dict(next_notes.get("price_adjustments", {}) or {})
    for action in actions:
        if action["tool"] == "set_price":
            price_adjustments[action["args"]["dish"]] = action["args"]["price"]
    next_notes["price_adjustments"] = price_adjustments
    next_notes["happy_hour_streak"] = (
        int(next_notes.get("happy_hour_streak", 0) or 0) + 1
        if any(action["tool"] == "run_happy_hour" for action in actions)
        else 0
    )

    rules = list(dict.fromkeys((next_notes.get("rules", []) or []) + [
        "never drop below 7 staff" if staff_floor(observation) >= 7 else "staff can flex to 3",
    ]))
    for ingredient in next_notes.get("critical_ingredients", []) or []:
        rules.append(f"watch {ingredient} stock")
    next_notes["rules"] = list(dict.fromkeys(rules))[-12:]
    return next_notes


def compact_notes(notes: dict[str, Any]) -> str:
    working = dict(notes)
    trim_order = [
        "rules",
        "known_alerts",
        "daily_revenue",
        "stockout_history",
        "price_adjustments",
        "banned_suppliers",
        "critical_ingredients",
    ]
    for _ in range(20):
        text = json.dumps(working, ensure_ascii=True, separators=(",", ":"))
        if len(text) <= NOTE_LIMIT:
            return text
        for key in trim_order:
            value = working.get(key)
            if isinstance(value, list) and value:
                working[key] = value[-max(1, len(value) - 1):]
                break
            if isinstance(value, dict) and value:
                first_key = next(iter(value))
                value.pop(first_key, None)
                break
        else:
            break
    text = json.dumps(working, ensure_ascii=True, separators=(",", ":"))
    return text[:NOTE_LIMIT]


def save_notes_action(
    observation: dict[str, Any],
    notes: dict[str, Any],
    actions: list[dict[str, Any]],
) -> dict[str, Any]:
    updated = update_notes_state(observation, notes, actions)
    return {
        "tool": "save_notes",
        "args": {"text": compact_notes(updated)},
    }


def strategy(observation: dict[str, Any], day: int) -> list[dict[str, Any]]:
    notes = parse_notes(observation.get("notes"))
    safety_actions, safety_meta = safety_rules(observation, notes)
    llm_actions = llm_decide(observation, notes)
    llm_actions = filter_llm_actions(observation, notes, safety_actions, llm_actions, safety_meta)
    actions = safety_actions + llm_actions
    actions.append(save_notes_action(observation, notes, actions))
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hybrid best agent")
    parser.add_argument("--scenario", default="baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--team-name", default="Relay")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print(f"Using model: {MODEL}")
    run_game(
        strategy,
        base_url=args.url,
        team_name=args.team_name,
        scenario=args.scenario,
        seed=args.seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
