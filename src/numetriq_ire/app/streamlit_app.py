from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from numetriq_ire.utils.gpu import get_device, gpu_summary
from numetriq_ire.simulation.monte_carlo import DemandModel, LeadTimeModel, Policy, SimParams
from numetriq_ire.simulation.tuner import tune_order_up_to


st.set_page_config(page_title="NuMetriq Inventory Risk Engine", layout="wide")


def decision_summary(best: dict) -> str:
    ou = best["candidate_order_up_to"]
    fill = best["service_level_fill_rate"]
    pso = best["stockout_probability"]
    cost = best["expected_total_cost"]
    return (
        f"**Recommendation:** set *order-up-to* to **{ou:.0f} units**.\n\n"
        f"- Expected fill rate (service): **{fill:.3f}**\n"
        f"- Probability of any stockout in horizon: **{pso:.3f}**\n"
        f"- Expected total cost (orders + holding + stockout penalty): **{cost:,.0f}**\n\n"
        f"Interpretation: this sits near the **efficient frontier**—moving to higher service levels "
        f"costs disproportionately more, while moving lower saves a little but increases stockout risk."
    )


def plot_cost_vs_ou(df: pd.DataFrame, objective: str):
    d = df.sort_values("candidate_order_up_to")
    fig = plt.figure()
    plt.plot(d["candidate_order_up_to"], d[objective], marker="o")
    plt.xlabel("Order-up-to (units)")
    plt.ylabel(objective)
    plt.title(f"Policy sweep: {objective} vs order-up-to")
    plt.tight_layout()
    return fig


def plot_stockout_vs_ou(df: pd.DataFrame):
    d = df.sort_values("candidate_order_up_to")
    fig = plt.figure()
    plt.plot(d["candidate_order_up_to"], d["stockout_probability"], marker="o")
    plt.xlabel("Order-up-to (units)")
    plt.ylabel("stockout_probability")
    plt.title("Policy sweep: stockout probability vs order-up-to")
    plt.tight_layout()
    return fig


def plot_pareto(df: pd.DataFrame, objective: str):
    fig = plt.figure()
    plt.scatter(df["service_level_fill_rate"], df[objective])
    # label a few best points so it's readable
    df2 = df.sort_values(objective).head(8)
    for _, row in df2.iterrows():
        plt.annotate(
            f"OU={int(row['candidate_order_up_to'])}",
            (row["service_level_fill_rate"], row[objective]),
            fontsize=8,
        )
    plt.xlabel("service_level_fill_rate")
    plt.ylabel(objective)
    plt.title("Tradeoff: cost vs service level (efficient frontier view)")
    plt.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def run_tuning_cached(
    *,
    n_paths: int,
    horizon_days: int,
    seed: int,
    review_every_days: int,
    initial_on_hand: float,
    unit_cost: float,
    holding_cost_per_unit_day: float,
    stockout_penalty_per_unit: float,
    demand_mean_daily: float,
    demand_std_daily: float,
    lead_mean_days: float,
    lead_std_days: float,
    lead_min_days: int,
    lead_max_days: int,
    grid_min: int,
    grid_max: int,
    coarse_step: int,
    fine_center: int,
    fine_half_width: int,
    fine_step: int,
    objective: str,
    service_level_min: float | None,
):
    device = get_device()

    params = SimParams(n_paths=n_paths, horizon_days=horizon_days, seed=seed)
    base_policy = Policy(review_every_days=review_every_days, order_up_to=250.0)

    demand = DemandModel(mean_daily=demand_mean_daily, std_daily=demand_std_daily)
    lead_time = LeadTimeModel(
        mean_days=lead_mean_days,
        std_days=lead_std_days,
        min_days=lead_min_days,
        max_days=lead_max_days,
    )

    coarse = np.arange(grid_min, grid_max + 1, coarse_step)
    fine = np.arange(
        max(grid_min, fine_center - fine_half_width),
        min(grid_max, fine_center + fine_half_width) + 1,
        fine_step,
    )
    grid = np.unique(np.concatenate([coarse, fine]))

    results = tune_order_up_to(
        params=params,
        base_policy=base_policy,
        demand=demand,
        lead_time=lead_time,
        initial_on_hand=initial_on_hand,
        unit_cost=unit_cost,
        holding_cost_per_unit_day=holding_cost_per_unit_day,
        stockout_penalty_per_unit=stockout_penalty_per_unit,
        order_up_to_grid=grid,
        objective=objective,
        constraint_service_level_min=service_level_min,
        device=device,
    )
    df = pd.DataFrame(results)
    best = results[0]
    return df, best, str(device), grid


def main():
    st.title("NuMetriq Inventory Risk Engine (GPU)")
    st.caption("GPU-accelerated Monte Carlo policy tuning for inventory decisions.")

    with st.expander("GPU / environment", expanded=False):
        s = gpu_summary()
        st.json(s)

    st.subheader("Inputs")

    colA, colB, colC = st.columns(3)

    with colA:
        objective = st.selectbox(
            "Objective (what are we optimizing?)",
            options=[
                "expected_total_cost",
                "stockout_probability",
                "expected_stockout_units",
                "expected_stockout_cost",
                "expected_holding_cost",
                "expected_cash_spent_on_orders",
            ],
            index=0,
        )
        service_level_min_toggle = st.checkbox("Enforce minimum service level?", value=False)
        service_level_min = None
        if service_level_min_toggle:
            service_level_min = st.slider("Minimum fill rate (service level)", 0.50, 0.99, 0.95, 0.01)

        review_every_days = st.slider("Review cadence (days)", 1, 14, 7, 1)
        horizon_days = st.slider("Horizon (days)", 14, 120, 60, 1)

    with colB:
        demand_mean_daily = st.number_input("Demand mean (units/day)", value=10.0, step=0.5)
        demand_std_daily = st.number_input("Demand std (units/day)", value=7.0, step=0.5)
        initial_on_hand = st.number_input("Initial on-hand inventory", value=80.0, step=5.0)

        lead_mean_days = st.number_input("Lead time mean (days)", value=9.0, step=0.5)
        lead_std_days = st.number_input("Lead time std (days)", value=4.0, step=0.5)

    with colC:
        unit_cost = st.number_input("Unit cost ($)", value=10.0, step=0.5)
        holding_cost_per_unit_day = st.number_input("Holding cost per unit/day ($)", value=0.05, step=0.01, format="%.3f")
        stockout_penalty_per_unit = st.number_input("Stockout penalty per unit ($)", value=100.0, step=5.0)

        n_paths = st.slider("Monte Carlo paths", 10_000, 200_000, 80_000, 10_000)
        seed = st.number_input("Random seed", value=42, step=1)

    st.subheader("Search grid")
    g1, g2, g3 = st.columns(3)
    with g1:
        grid_min = st.number_input("Grid min (OU)", value=50, step=10)
        grid_max = st.number_input("Grid max (OU)", value=650, step=10)
    with g2:
        coarse_step = st.number_input("Coarse step", value=50, step=10)
        fine_step = st.number_input("Fine step", value=10, step=1)
    with g3:
        fine_center = st.number_input("Fine center", value=310, step=10)
        fine_half_width = st.number_input("Fine half-width", value=150, step=10)

    run = st.button("Run tuning", type="primary")

    if run:
        with st.spinner("Running GPU Monte Carlo sweeps..."):
            df, best, device_str, grid = run_tuning_cached(
                n_paths=n_paths,
                horizon_days=horizon_days,
                seed=seed,
                review_every_days=review_every_days,
                initial_on_hand=initial_on_hand,
                unit_cost=unit_cost,
                holding_cost_per_unit_day=holding_cost_per_unit_day,
                stockout_penalty_per_unit=stockout_penalty_per_unit,
                demand_mean_daily=demand_mean_daily,
                demand_std_daily=demand_std_daily,
                lead_mean_days=lead_mean_days,
                lead_std_days=lead_std_days,
                lead_min_days=1,
                lead_max_days=30,
                grid_min=int(grid_min),
                grid_max=int(grid_max),
                coarse_step=int(coarse_step),
                fine_center=int(fine_center),
                fine_half_width=int(fine_half_width),
                fine_step=int(fine_step),
                objective=objective,
                service_level_min=service_level_min,
            )

        st.success(f"Done. Device used: **{device_str}**. Candidates evaluated: **{len(grid)}**")

        left, right = st.columns([1, 1])

        with left:
            st.subheader("Recommended policy")
            st.markdown(decision_summary(best))

        with right:
            st.subheader("Key metrics (best)")
            st.dataframe(pd.DataFrame([best]).T.rename(columns={0: "value"}), use_container_width=True)

        st.subheader("Plots")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.pyplot(plot_cost_vs_ou(df, objective), clear_figure=True)
        with c2:
            st.pyplot(plot_stockout_vs_ou(df), clear_figure=True)
        with c3:
            st.pyplot(plot_pareto(df, objective), clear_figure=True)

        st.subheader("All candidates")
        show_cols = [
            "candidate_order_up_to",
            "expected_total_cost",
            "service_level_fill_rate",
            "stockout_probability",
            "expected_stockout_units",
            "expected_cash_spent_on_orders",
            "expected_holding_cost",
            "expected_stockout_cost",
        ]
        cols = [c for c in show_cols if c in df.columns]
        st.dataframe(df[cols].sort_values(objective), use_container_width=True)


if __name__ == "__main__":
    main()