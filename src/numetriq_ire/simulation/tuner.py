from __future__ import annotations

from dataclasses import replace
from typing import Iterable, List, Dict, Optional

import torch

from numetriq_ire.simulation.monte_carlo import DemandModel, LeadTimeModel, Policy, SimParams, run_inventory_mc


def tune_order_up_to(
    *,
    params: SimParams,
    base_policy: Policy,
    demand: DemandModel,
    lead_time: LeadTimeModel,
    initial_on_hand: float,
    unit_cost: float,
    holding_cost_per_unit_day: float,
    stockout_penalty_per_unit: float,
    order_up_to_grid: Iterable[float],
    objective: str = "expected_total_cost",
    constraint_service_level_min: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> List[Dict]:
    """
    Sweep order_up_to over a grid and return a list of result dicts (one per candidate),
    sorted by objective ascending.

    objective: one of:
      - expected_total_cost
      - expected_stockout_units
      - stockout_probability
      - expected_stockout_cost
      - expected_holding_cost
      - expected_cash_spent_on_orders
    """
    results: List[Dict] = []

    for ou in order_up_to_grid:
        policy = replace(base_policy, order_up_to=float(ou))
        r = run_inventory_mc(
            params=params,
            policy=policy,
            demand=demand,
            lead_time=lead_time,
            initial_on_hand=initial_on_hand,
            unit_cost=unit_cost,
            holding_cost_per_unit_day=holding_cost_per_unit_day,
            stockout_penalty_per_unit=stockout_penalty_per_unit,
            device=device,
        )
        r["candidate_order_up_to"] = float(ou)
        results.append(r)

    # Apply constraint if present
    if constraint_service_level_min is not None:
        filtered = [
            r for r in results
            if r.get("service_level_fill_rate", 0.0) >= float(constraint_service_level_min)
        ]
        # If everything gets filtered out, keep original list so user sees something
        results = filtered if filtered else results

    if objective not in results[0]:
        raise ValueError(f"Objective '{objective}' not in results. Available keys: {list(results[0].keys())}")

    results.sort(key=lambda d: d[objective])
    return results