from config import (
    N,
    T,
    alpha,
    beta,
    collaboration_cost,
    discount_factor,
    figures_dir,
    gamma,
    initial_corruption_rate,
    memory_endowment,
    num_simulations,
    num_trajectories,
    results_dir,
    save_csv,
    save_trajectories,
    seed,
    simulation_results_csv,
    task_probability,
    trajectory_results_csv,
)
from simulation import many_simulations, one_simulation


if __name__ == "__main__":

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Run one simulation.
    results = one_simulation(
        N=N,
        T=T,
        initial_corruption_rate=initial_corruption_rate,
        task_probability=task_probability,
        beta=beta,
        gamma=gamma,
        alpha=alpha,
        memory_endowment=memory_endowment,
        collaboration_cost=collaboration_cost,
        discount_factor=discount_factor,
        seed=seed,
    )

    print("Single simulation results:")
    print(f"Initial number of corrupted agents: {results['initial_num_corrupted']}")
    print(f"Initial corruption rate: {results['initial_corruption_rate_actual']:.4f}")
    print(f"Final number of honest agents: {results['final_num_honest']}")
    print(f"Final number of corrupted agents: {results['final_num_corrupted']}")
    print(f"Final corruption rate: {results['final_corruption_rate']:.4f}")
    print(f"Mean collaboration rate: {results['collaboration_rate'].mean():.4f}")
    print(f"Mean belief cutoff: {results['belief_cutoff'].mean():.4f}")
    print(f"Mean clean pool accuracy: {results['mean_clean_pool_accuracy']:.4f}")
    print(f"Mean raw recommendation accuracy: {results['mean_raw_recommendation_accuracy']:.4f}")
    print(f"Mean submitted vote accuracy: {results['mean_submitted_vote_accuracy']:.4f}")
    print(f"Mean collective decision accuracy: {results['mean_collective_decision_accuracy']:.4f}")
    print(f"Mean reversal rate: {results['mean_reversal_rate']:.4f}")
    print(f"Total conversions: {results['total_conversions']}")

    # Run repeated simulations.
    summary = many_simulations(
        num_simulations=num_simulations,
        N=N,
        T=T,
        initial_corruption_rate=initial_corruption_rate,
        task_probability=task_probability,
        beta=beta,
        gamma=gamma,
        alpha=alpha,
        memory_endowment=memory_endowment,
        collaboration_cost=collaboration_cost,
        discount_factor=discount_factor,
        save_csv=save_csv,
        csv_filename=simulation_results_csv,
        save_trajectories=save_trajectories,
        num_trajectories=num_trajectories,
        trajectory_csv_filename=trajectory_results_csv,
    )

    print("\nSummary of many simulations:")
    print(f"Mean final corruption rate: {summary['mean_final_corruption_rate']:.4f}")
    print(f"Standard deviation of final corruption rate: {summary['std_final_corruption_rate']:.4f}")
    print(f"Minimum final corruption rate: {summary['min_final_corruption_rate']:.4f}")
    print(f"Maximum final corruption rate: {summary['max_final_corruption_rate']:.4f}")
    print(f"Mean spread amount: {summary['mean_spread_amount']:.4f}")
    print(f"Mean spread rate change: {summary['mean_spread_rate_change']:.4f}")
    print(f"Probability of full corruption: {summary['prob_full_corruption']:.4f}")
    print(f"Probability of no spread: {summary['prob_no_spread']:.4f}")
    print(f"Mean collaboration rate: {summary['mean_collaboration_rate']:.4f}")
    print(f"Mean clean pool accuracy: {summary['mean_clean_pool_accuracy']:.4f}")
    print(f"Mean raw recommendation accuracy: {summary['mean_raw_recommendation_accuracy']:.4f}")
    print(f"Mean submitted vote accuracy: {summary['mean_submitted_vote_accuracy']:.4f}")
    print(f"Mean collective decision accuracy: {summary['mean_collective_decision_accuracy']:.4f}")
    print(f"Mean reversal rate: {summary['mean_reversal_rate']:.4f}")
    print(f"Mean total conversions: {summary['mean_total_conversions']:.4f}")