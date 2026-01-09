from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class SimParams:
    """Parameters for a simple inventory Monte Carlo simulation."""
    n_paths: int = 50_000
    horizon_days: int = 30
    seed: int = 42


@dataclass(frozen=True)
class Policy:
    """Simple periodic review order-up-to policy."""
    review_every_days: int = 7
    order_up_to: float = 250.0  # units


@dataclass(frozen=True)
class DemandModel:
    """IID daily demand with Normal noise, truncated at 0."""
    mean_daily: float = 8.0
    std_daily: float = 3.0


@dataclass(frozen=True)
class LeadTimeModel:
    """Lead time in days (discrete)."""
    mean_days: float = 5.0
    std_days: float = 1.0
    min_days: int = 1
    max_days: int = 21


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _trunc_normal_nonneg(
    shape: Tuple[int, ...],
    mean: float,
    std: float,
    device: torch.device,
) -> torch.Tensor:
    x = torch.randn(shape, device=device) * std + mean
    return torch.clamp(x, min=0.0)


def _sample_lead_time_days(
    n_paths: int,
    mean_days: float,
    std_days: float,
    min_days: int,
    max_days: int,
    device: torch.device,
) -> torch.Tensor:
    # Sample continuous normal then round to integer days, clamp
    lt = torch.randn((n_paths,), device=device) * std_days + mean_days
    lt = torch.round(lt).to(torch.int64)
    return torch.clamp(lt, min=min_days, max=max_days)


@torch.inference_mode()
def run_inventory_mc(
    *,
    params: SimParams,
    policy: Policy,
    demand: DemandModel,
    lead_time: LeadTimeModel,
    initial_on_hand: float = 200.0,
    unit_cost: float = 10.0,
    holding_cost_per_unit_day: float = 0.02,
    stockout_penalty_per_unit: float = 5.0,
    device: Optional[torch.device] = None,
) -> dict:
    """
    GPU-friendly Monte Carlo simulation for a single SKU.
    Returns aggregate risk metrics for the horizon.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _seed_all(params.seed)

    n = params.n_paths
    T = params.horizon_days

    # State vectors per path
    on_hand = torch.full((n,), float(initial_on_hand), device=device)
    on_order_qty = torch.zeros((n,), device=device)  # only one outstanding order in this simple model
    on_order_eta = torch.full((n,), -1, device=device, dtype=torch.int64)  # day index when it arrives (-1 = none)

    # Trackers
    total_demand = torch.zeros((n,), device=device)
    total_stockout_units = torch.zeros((n,), device=device)
    total_holding_cost = torch.zeros((n,), device=device)
    total_stockout_cost = torch.zeros((n,), device=device)
    total_order_cost = torch.zeros((n,), device=device)

    # Pre-sample demands [n, T]
    daily_demand = _trunc_normal_nonneg((n, T), demand.mean_daily, demand.std_daily, device=device)

    for day in range(T):
        day_i = torch.tensor(day, device=device, dtype=torch.int64)

        # Receive orders arriving today
        arriving = (on_order_eta == day_i)
        if arriving.any():
            on_hand = on_hand + on_order_qty * arriving.to(on_hand.dtype)
            on_order_qty = on_order_qty * (~arriving).to(on_order_qty.dtype)
            on_order_eta = torch.where(arriving, torch.full_like(on_order_eta, -1), on_order_eta)

        # Demand happens
        d = daily_demand[:, day]
        total_demand += d

        sold = torch.minimum(on_hand, d)
        stockout = d - sold

        on_hand = on_hand - sold
        total_stockout_units += stockout

        # Costs (simple)
        total_holding_cost += on_hand * holding_cost_per_unit_day
        total_stockout_cost += stockout * stockout_penalty_per_unit

        # Review & place order if it's a review day and no outstanding order
        if policy.review_every_days > 0 and (day % policy.review_every_days == 0):
            no_outstanding = (on_order_eta < 0)
            if no_outstanding.any():
                target = torch.full((n,), float(policy.order_up_to), device=device)
                needed = torch.clamp(target - on_hand, min=0.0)
                # Only order for paths with no outstanding
                order_qty = needed * no_outstanding.to(needed.dtype)

                # Sample lead time for those orders
                lt = _sample_lead_time_days(
                    n_paths=n,
                    mean_days=lead_time.mean_days,
                    std_days=lead_time.std_days,
                    min_days=lead_time.min_days,
                    max_days=lead_time.max_days,
                    device=device,
                )

                # Set outstanding orders
                on_order_qty = torch.where(no_outstanding, order_qty, on_order_qty)
                on_order_eta = torch.where(
                    no_outstanding,
                    torch.clamp(day_i + lt, max=torch.tensor(T - 1, device=device, dtype=torch.int64)),
                    on_order_eta,
                )

                total_order_cost += order_qty * unit_cost

    # Aggregate metrics
    stockout_prob = (total_stockout_units > 0).float().mean().item()
    expected_stockout_units = total_stockout_units.mean().item()
    expected_demand = total_demand.mean().item()

    cash_tied_up = (total_order_cost).mean().item()
    holding_cost = total_holding_cost.mean().item()
    stockout_cost = total_stockout_cost.mean().item()

    service_level = 1.0 - (total_stockout_units.sum().item() / max(total_demand.sum().item(), 1e-9))

    return {
        "device": str(device),
        "n_paths": params.n_paths,
        "horizon_days": params.horizon_days,
        "review_every_days": policy.review_every_days,
        "order_up_to": policy.order_up_to,
        "demand_mean_daily": demand.mean_daily,
        "demand_std_daily": demand.std_daily,
        "lead_time_mean_days": lead_time.mean_days,
        "lead_time_std_days": lead_time.std_days,
        "stockout_probability": stockout_prob,
        "expected_stockout_units": expected_stockout_units,
        "expected_total_demand": expected_demand,
        "service_level_fill_rate": float(service_level),
        "expected_cash_spent_on_orders": cash_tied_up,
        "expected_holding_cost": holding_cost,
        "expected_stockout_cost": stockout_cost,
        "expected_total_cost": cash_tied_up + holding_cost + stockout_cost,
    }