import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numetriq_ire.utils.gpu import get_device, print_gpu_summary
from numetriq_ire.utils.run_io import make_run_dir, save_json
from numetriq_ire.simulation.monte_carlo import DemandModel, LeadTimeModel, Policy, SimParams
from numetriq_ire.simulation.tuner import tune_order_up_to


def main():
    print_gpu_summary()
    device = get_device()

    # --- Config for this run ---
    params = SimParams(n_paths=30_000, horizon_days=30, seed=42)
    base_policy = Policy(review_every_days=7, order_up_to=250)

    demand = DemandModel(mean_daily=8.0, std_daily=3.0)
    lead_time = LeadTimeModel(mean_days=5.0, std_days=1.0, min_days=1, max_days=21)

    grid = np.arange(150, 451, 25)  # 150..450
    objective = "expected_total_cost"
    service_level_min = 0.95

    # --- Run ---
    results = tune_order_up_to(
        params=params,
        base_policy=base_policy,
        demand=demand,
        lead_time=lead_time,
        initial_on_hand=200,
        unit_cost=10.0,
        holding_cost_per_unit_day=0.02,
        stockout_penalty_per_unit=5.0,
        order_up_to_grid=grid,
        objective=objective,
        constraint_service_level_min=service_level_min,
        device=device,
    )

    best = results[0]
    print("\n--- Best policy (by expected_total_cost) ---")
    print("order_up_to:", best["candidate_order_up_to"])
    print("expected_total_cost:", best["expected_total_cost"])
    print("service_level_fill_rate:", best["service_level_fill_rate"])
    print("stockout_probability:", best["stockout_probability"])

    print("\n--- Top 5 candidates ---")
    for r in results[:5]:
        print(
            f"OU={r['candidate_order_up_to']:.0f} | "
            f"Cost={r['expected_total_cost']:.2f} | "
            f"Fill={r['service_level_fill_rate']:.4f} | "
            f"P(stockout)={r['stockout_probability']:.3f}"
        )

    # --- Save artifacts ---
    run_dir = make_run_dir(run_name="tune_order_up_to")
    df = pd.DataFrame(results)

    csv_path = run_dir / "policy_grid.csv"
    df.to_csv(csv_path, index=False)

    meta_path = run_dir / "run_meta.json"
    save_json(
        meta_path,
        {
            "objective": objective,
            "service_level_min": service_level_min,
            "grid_min": float(grid.min()),
            "grid_max": float(grid.max()),
            "grid_step": float(grid[1] - grid[0]) if len(grid) > 1 else None,
            "best_candidate": best,
        },
    )

    # Plot: cost vs order_up_to
    fig_path = run_dir / "cost_vs_order_up_to.png"
    df_sorted = df.sort_values("candidate_order_up_to")

    plt.figure()
    plt.plot(df_sorted["candidate_order_up_to"], df_sorted[objective], marker="o")
    plt.xlabel("Order-up-to (units)")
    plt.ylabel(objective)
    plt.title("Policy sweep: cost vs order-up-to")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)

    # Plot: stockout probability vs order_up_to
    fig2_path = run_dir / "stockout_prob_vs_order_up_to.png"
    plt.figure()
    plt.plot(df_sorted["candidate_order_up_to"], df_sorted["stockout_probability"], marker="o")
    plt.xlabel("Order-up-to (units)")
    plt.ylabel("stockout_probability")
    plt.title("Policy sweep: stockout probability vs order-up-to")
    plt.tight_layout()
    plt.savefig(fig2_path, dpi=160)

    print("\nSaved:")
    print(" -", csv_path)
    print(" -", meta_path)
    print(" -", fig_path)
    print(" -", fig2_path)


if __name__ == "__main__":
    main()