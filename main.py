import os

from simulation import one_simulation, many_simulations

from config import (
    N,
    T,
    num_simulations,
    initial_corruption_rate,
    beta,
    gamma,
    alpha,
    memory_endowment,
    collaboration_cost,
    discount_factor,
    seed,
    save_csv,
    save_trajectories,
    num_trajectories,
    simulation_results_csv,
    trajectory_results_csv,
    results_dir,
    figures_dir

)

if __name__ == "__main__":

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)


    # Run a single simulation.
    results = one_simulation(
        N=N,
        T=T,
        initial_corruption_rate=initial_corruption_rate,
        beta=beta,
        gamma=gamma,
        alpha=alpha,
        memory_endowment=memory_endowment,
        collaboration_cost=collaboration_cost,
        discount_factor=discount_factor,
        seed=seed)

    print("Single simulation results:")
    print(f"Initial number of corrupted agents: {results['initial_num_corrupted']}")
    print(f"Initial corrupted rate: {results['initial_corruption_rate_actual']:.4f}")
    print(f"Final number of honest agents: {results['final_num_honest']}")
    print(f"Final number of corrupted agents: {results['final_num_corrupted']}")
    print(f"Final corrupted rate: {results['final_corruption_rate']:.4f}")
    print(f"Mean collaboration rate: {results['collaboration_rate'].mean():.4f}")
    print(f"Mean belief cutoff: {results['belief_cutoff'].mean():.4f}")

    # Run many simulations.
    summary = many_simulations(
        num_simulations=num_simulations,
        N=N,
        T=T,
        initial_corruption_rate=initial_corruption_rate,
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
        trajectory_csv_filename=trajectory_results_csv
    )

    print("\nSummary of many simulations:")
    print(f"Mean final corrupted rate: {summary['mean_final_corruption_rate']:.4f}")
    print(f"Standard deviation of final corrupted rate: {summary['std_final_corruption_rate']:.4f}")
    print(f"Minimum final corrupted rate: {summary['min_final_corruption_rate']:.4f}")
    print(f"Maximum final corrupted rate: {summary['max_final_corruption_rate']:.4f}")
    print(f"Mean spread amount: {summary['mean_spread_amount']:.4f}")
    print(f"Mean spread rate change: {summary['mean_spread_rate_change']:.4f}")
    print(f"Probability of full corruption: {summary['prob_full_corruption']:.4f}")
    print(f"Probability of no spread: {summary['prob_no_spread']:.4f}")