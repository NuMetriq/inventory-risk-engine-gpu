from numetriq_ire.utils.gpu import get_device, print_gpu_summary
from numetriq_ire.simulation.monte_carlo import DemandModel, LeadTimeModel, Policy, SimParams, run_inventory_mc


def main():
    print_gpu_summary()
    device = get_device()

    params = SimParams(n_paths=50_000, horizon_days=30, seed=42)
    policy = Policy(review_every_days=7, order_up_to=250)
    demand = DemandModel(mean_daily=8.0, std_daily=3.0)
    lead_time = LeadTimeModel(mean_days=5.0, std_days=1.0, min_days=1, max_days=21)

    results = run_inventory_mc(
        params=params,
        policy=policy,
        demand=demand,
        lead_time=lead_time,
        initial_on_hand=200,
        unit_cost=10.0,
        holding_cost_per_unit_day=0.02,
        stockout_penalty_per_unit=5.0,
        device=device,
    )

    print("\n--- Simulation Results ---")
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()