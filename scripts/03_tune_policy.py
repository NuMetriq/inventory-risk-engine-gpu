import numpy as np

from numetriq_ire.utils.gpu import get_device, print_gpu_summary
from numetriq_ire.simulation.monte_carlo import DemandModel, LeadTimeModel, Policy, SimParams
from numetriq_ire.simulation.tuner import tune_order_up_to


def main():
    print_gpu_summary()
    device = get_device()

    params = SimParams(n_paths=30_000, horizon_days=30, seed=42)
    base_policy = Policy(review_every_days=7, order_up_to=250)

    demand = DemandModel(mean_daily=8.0, std_daily=3.0)
    lead_time = LeadTimeModel(mean_days=5.0, std_days=1.0, min_days=1, max_days=21)

    grid = np.arange(150, 451, 25)  # 150..450

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
        objective="expected_total_cost",
        constraint_service_level_min=0.95,  # optional; set None to disable
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


if __name__ == "__main__":
    main()