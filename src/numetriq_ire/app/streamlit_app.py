from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from numetriq_ire.utils.gpu import get_device, gpu_summary
from numetriq_ire.utils.run_io import make_run_dir, save_json
from numetriq_ire.simulation.monte_carlo import DemandModel, LeadTimeModel, Policy, SimParams
from numetriq_ire.simulation.tuner import tune_order_up_to

st.set_page_config(page_title="NuMetriq Inventory Risk Engine", layout="wide")


def decision_summary(best: dict) -> str:
    ou = float(best.get("candidate_order_up_to", float("nan")))
    fill = float(best.get("service_level_fill_rate", float("nan")))
    pso = float(best.get("stockout_probability", float("nan")))
    cost = float(best.get("expected_total_cost", float("nan")))

    return (
        f"**Recommendation:** set *order-up-to* to **{ou:.0f} units**.\n\n"
        f"- Expected fill rate (service): **{fill:.3f}**\n"
        f"- Probability of any stockout over the horizon: **{pso:.3f}**\n"
        f"- Expected total cost (orders + holding + stockout penalty): **{cost:,.0f}**\n\n"
        "Interpretation: this recommendation sits near the **efficient frontier**—"
        "pushing service higher costs disproportionately more, while pushing lower "
        "saves a little but increases stockout risk."
    )


def build_grid(
    grid_min: int,
    grid_max: int,
    coarse_step: int,
    fine_center: int,
    fine_half_width: int,
    fine_step: int,
) -> np.ndarray:
    coarse = np.arange(grid_min, grid_max + 1, coarse_step)
    fine_lo = max(grid_min, fine_center - fine_half_width)
    fine_hi = min(grid_max, fine_center + fine_half_width)
    fine = np.arange(fine_lo, fine_hi + 1, fine_step)
    grid = np.unique(np.concatenate([coarse, fine]))
    return grid


def fig_cost_vs_ou(df: pd.DataFrame, objective: str) -> plt.Figure:
    d = df.sort_values("candidate_order_up_to")
    fig = plt.figure()
    plt.plot(d["candidate_order_up_to"], d[objective], marker="o")
    plt.xlabel("Order-up-to (units)")
    plt.ylabel(objective)
    plt.title(f"Policy sweep: {objective} vs order-up-to")
    plt.tight_layout()
    return fig


def fig_stockout_vs_ou(df: pd.DataFrame) -> plt.Figure:
    d = df.sort_values("candidate_order_up_to")
    fig = plt.figure()
    plt.plot(d["candidate_order_up_to"], d["stockout_probability"], marker="o")
    plt.xlabel("Order-up-to (units)")
    plt.ylabel("stockout_probability")
    plt.title("Policy sweep: stockout probability vs order-up-to")
    plt.tight_layout()
    return fig


def fig_pareto(df: pd.DataFrame, objective: str) -> plt.Figure:
    fig = plt.figure()
    plt.scatter(df["service_level_fill_rate"], df[objective])

    # Label a few best candidates so it stays readable
    df2 = df.sort_values(objective).head(8)
    for _, row in df2.iterrows():
        ou = int(row["candidate_order_up_to"])
        x = float(row["service_level_fill_rate"])
        y = float(row[objective])
        plt.annotate(f"OU={ou}", (x, y), fontsize=8)

    plt.xlabel("service_level_fill_rate")
    plt.ylabel(objective)
    plt.title("Tradeoff: cost vs service level (efficient frontier view)")
    plt.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def run_tuning(
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
    objective: str,
    service_level_min: float | None,
    grid: np.ndarray,
) -> tuple[pd.DataFrame, dict, str]:
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
    return df, best, str(device)


def main() -> None:
    st.title("NuMetriq Inventory Risk Engine (GPU)")
    st.caption("GPU-accelerated Monte Carlo policy tuning for inventory decisions.")

    with st.expander("GPU / environment", expanded=False):
        st.json(gpu_summary())

    st.subheader("Inputs")

    col1, col2, col3 = st.columns(3)

    with col1:
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

        enforce_min_service = st.checkbox("Enforce minimum service level?", value=False)
        service_level_min = None
        if enforce_min_service:
            service_level_min = st.slider("Minimum fill rate (service level)", 0.50, 0.99, 0.95, 0.01)

        review_every_days = st.slider("Review cadence (days)", 1, 14, 7, 1)
        horizon_days = st.slider("Horizon (days)", 14, 120, 60, 1)

    with col2:
        demand_mean_daily = st.number_input("Demand mean (units/day)", value=10.0, step=0.5)
        demand_std_daily = st.number_input("Demand std (units/day)", value=7.0, step=0.5)
        initial_on_hand = st.number_input("Initial on-hand inventory", value=80.0, step=5.0)

        lead_mean_days = st.number_input("Lead time mean (days)", value=9.0, step=0.5)
        lead_std_days = st.number_input("Lead time std (days)", value=4.0, step=0.5)

    with col3:
        unit_cost = st.number_input("Unit cost ($)", value=10.0, step=0.5)
        holding_cost_per_unit_day = st.number_input(
            "Holding cost per unit/day ($)",
            value=0.05,
            step=0.01,
            format="%.3f",
        )
        stockout_penalty_per_unit = st.number_input("Stockout penalty per unit ($)", value=100.0, step=5.0)

        n_paths = st.slider("Monte Carlo paths", 10_000, 200_000, 80_000, 10_000)
        seed = st.number_input("Random seed", value=42, step=1)

    st.subheader("Search grid")
    g1, g2, g3 = st.columns(3)
    with g1:
        grid_min = int(st.number_input("Grid min (OU)", value=50, step=10))
        grid_max = int(st.number_input("Grid max (OU)", value=650, step=10))
    with g2:
        coarse_step = int(st.number_input("Coarse step", value=50, step=10))
        fine_step = int(st.number_input("Fine step", value=10, step=1))
    with g3:
        fine_center = int(st.number_input("Fine center", value=310, step=10))
        fine_half_width = int(st.number_input("Fine half-width", value=150, step=10))

    st.subheader("Outputs")
    o1, o2 = st.columns(2)
    with o1:
        save_run = st.checkbox("Save run outputs to artifacts/", value=True)
    with o2:
        run_name = st.text_input("Run name (optional)", value="streamlit_tuning")

    run_clicked = st.button("Run tuning", type="primary")

    if run_clicked:
        grid = build_grid(
            grid_min=grid_min,
            grid_max=grid_max,
            coarse_step=coarse_step,
            fine_center=fine_center,
            fine_half_width=fine_half_width,
            fine_step=fine_step,
        )

        with st.spinner("Running GPU Monte Carlo sweeps..."):
            df, best, device_str = run_tuning(
                n_paths=int(n_paths),
                horizon_days=int(horizon_days),
                seed=int(seed),
                review_every_days=int(review_every_days),
                initial_on_hand=float(initial_on_hand),
                unit_cost=float(unit_cost),
                holding_cost_per_unit_day=float(holding_cost_per_unit_day),
                stockout_penalty_per_unit=float(stockout_penalty_per_unit),
                demand_mean_daily=float(demand_mean_daily),
                demand_std_daily=float(demand_std_daily),
                lead_mean_days=float(lead_mean_days),
                lead_std_days=float(lead_std_days),
                lead_min_days=1,
                lead_max_days=30,
                objective=str(objective),
                service_level_min=service_level_min,
                grid=grid,
            )

        st.success(f"Done. Device used: **{device_str}**. Candidates evaluated: **{len(grid)}**")

        # Build figures once (so we can both show and optionally save)
        fig_cost = fig_cost_vs_ou(df, objective)
        fig_stock = fig_stockout_vs_ou(df)
        fig_frontier = fig_pareto(df, objective)

        run_dir = None
        if save_run:
            base_name = run_name.strip() if run_name and run_name.strip() else None
            run_dir = make_run_dir(run_name=base_name)

            df.to_csv(run_dir / "policy_grid.csv", index=False)

            save_json(
                run_dir / "run_meta.json",
                {
                    "device": device_str,
                    "objective": objective,
                    "service_level_min": service_level_min,
                    "n_paths": int(n_paths),
                    "horizon_days": int(horizon_days),
                    "seed": int(seed),
                    "review_every_days": int(review_every_days),
                    "initial_on_hand": float(initial_on_hand),
                    "unit_cost": float(unit_cost),
                    "holding_cost_per_unit_day": float(holding_cost_per_unit_day),
                    "stockout_penalty_per_unit": float(stockout_penalty_per_unit),
                    "demand_mean_daily": float(demand_mean_daily),
                    "demand_std_daily": float(demand_std_daily),
                    "lead_mean_days": float(lead_mean_days),
                    "lead_std_days": float(lead_std_days),
                    "grid_min": grid_min,
                    "grid_max": grid_max,
                    "coarse_step": coarse_step,
                    "fine_center": fine_center,
                    "fine_half_width": fine_half_width,
                    "fine_step": fine_step,
                    "best_candidate": best,
                },
            )

            fig_cost.savefig(run_dir / "cost_vs_order_up_to.png", dpi=160)
            fig_stock.savefig(run_dir / "stockout_prob_vs_order_up_to.png", dpi=160)
            fig_frontier.savefig(run_dir / "pareto_cost_vs_service.png", dpi=160)

            st.info(f"Saved run to: {run_dir}")

        left, right = st.columns([1, 1])

        with left:
            st.subheader("Recommended policy")
            st.markdown(decision_summary(best))

        with right:
            st.subheader("Best candidate (details)")
            st.dataframe(pd.DataFrame([best]).T.rename(columns={0: "value"}), use_container_width=True)

        st.subheader("Plots")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.pyplot(fig_cost, clear_figure=True)
        with c2:
            st.pyplot(fig_stock, clear_figure=True)
        with c3:
            st.pyplot(fig_frontier, clear_figure=True)

        st.subheader("All candidates")
        cols = [
            "candidate_order_up_to",
            "expected_total_cost",
            "service_level_fill_rate",
            "stockout_probability",
            "expected_stockout_units",
            "expected_cash_spent_on_orders",
            "expected_holding_cost",
            "expected_stockout_cost",
        ]
        cols = [c for c in cols if c in df.columns]
        st.dataframe(df[cols].sort_values(objective), use_container_width=True)


if __name__ == "__main__":
    main()