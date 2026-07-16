import numpy as np
import pandas as pd

from network import ring_network
from dynamics import initialize_states, initialize_beliefs, apply_contagion, update_beliefs
from proposal_rules import make_proposals, collaboration_pairs

def one_simulation(N, T, initial_corruption_rate, belief_threshold, beta, seed=None):

    rng = np.random.default_rng(seed)

    # Create a ring network with N nodes.
    neighbors = ring_network(N)

    # Initialize the states and beliefs of the nodes in the network.
    corrupted = initialize_states(N, initial_corruption_rate, rng)
    belief = initialize_beliefs(N, initial_corruption_rate)

    # Record the initial number of corrupted agents and the actual corruption rate.
    initial_num_corrupted = int(np.sum(corrupted))
    initial_corruption_rate_actual = float(np.mean(corrupted))
    
    # Store statistics over time.
    num_corrupted_history = []
    num_honest_history = []
    corruption_rate_history = []
    collaboration_rate_history = []

    for t in range(T):

        # Record the current statistics at each time step.
        num_corrupted = int(np.sum(corrupted))
        num_honest = N - num_corrupted
        corruption_rate = num_corrupted/N


        num_corrupted_history.append(num_corrupted)
        num_honest_history.append(num_honest)
        corruption_rate_history.append(corruption_rate)

        # Each agent makes a proposal.
        proposals = make_proposals(
            N=N,
            neighbors=neighbors,
            belief=belief,
            belief_threshold=belief_threshold,
            rng=rng)

        # Mutual proposal from collaboration pairs.
        partners, pairs = collaboration_pairs(N=N, proposals=proposals)

        # Record the collaboration rate.
        collaboration_rate = np.mean(partners != -1)
        collaboration_rate_history.append(collaboration_rate)

        # Apply contagion.
        next_corrupted = apply_contagion(corrupted=corrupted, pairs=pairs, beta=beta, rng=rng)

        # Update beliefs based on the current and next corrupted states.
        belief = update_beliefs(belief=belief,
                                corrupted=corrupted,
                                next_corrupted=next_corrupted,
                                pairs=pairs,
                                beta=beta)

        # Update the corrupted states for the next iteration.
        corrupted = next_corrupted

    # Store the statistics for the current time step.
    results = {
            "initial_num_corrupted": initial_num_corrupted,
            "initial_corruption_rate_actual": initial_corruption_rate_actual,
            "num_corrupted": np.array(num_corrupted_history),
            "num_honest": np.array(num_honest_history),
            "corruption_rate": np.array(corruption_rate_history),
            "collaboration_rate": np.array(collaboration_rate_history),
            "final_num_corrupted": int(np.sum(corrupted)),
            "final_num_honest": int(N - np.sum(corrupted)),
            "final_corruption_rate": float(np.mean(corrupted)),
            "final_states": corrupted,
            "final_beliefs": belief
    }

    return results
    

def many_simulations(
        num_simulations,
        N,
        T,
        initial_corruption_rate,
        belief_threshold,
        beta,
        save_csv,
        csv_filename,
        save_trajectories,
        num_trajectories,
        trajectory_csv_filename
        ):

    run_records = []
    trajectory_records = []
    

    for simulation in range(num_simulations):
        results = one_simulation(N=N, T=T, initial_corruption_rate=initial_corruption_rate, belief_threshold=belief_threshold, beta=beta, seed=simulation)

        run_record = {
            "run_id": simulation,
            "N": N,
            "T": T,
            "initial_corruption_rate_parameter": initial_corruption_rate,
            "beta": beta,
            "belief_threshold": belief_threshold,
            "initial_num_corrupted": results["initial_num_corrupted"],
            "initial_corruption_rate_actual": results["initial_corruption_rate_actual"],
            "final_num_corrupted": results["final_num_corrupted"],
            "final_num_honest": results["final_num_honest"],
            "final_corruption_rate": results["final_corruption_rate"],
            "spread_amount": results["final_num_corrupted"] - results["initial_num_corrupted"],
            "spread_rate_change": results["final_corruption_rate"] - results["initial_corruption_rate_actual"],
            "full_corruption": results["final_num_corrupted"] == N,
            "no_spread": results["final_num_corrupted"] <= results["initial_num_corrupted"]

        }

        run_records.append(run_record)

        if save_trajectories and simulation < num_trajectories:
            for t in range(len(results["corruption_rate"])):
                trajectory_record = {
                    "run_id": simulation,
                    "time_step": t,
                    "N": N,
                    "T": T,
                    "initial_corruption_rate_parameter": initial_corruption_rate,
                    "beta": beta,
                    "belief_threshold": belief_threshold,
                    "num_corrupted": results["num_corrupted"][t],
                    "num_honest": results["num_honest"][t],
                    "corruption_rate": results["corruption_rate"][t],
                    "collaboration_rate": results["collaboration_rate"][t]
                }

                trajectory_records.append(trajectory_record)

    
    results_df = pd.DataFrame(run_records)

    if save_csv:
        results_df.to_csv(csv_filename, index=False)
        print(f"Simulation results saved to {csv_filename}")

    if save_trajectories:
        trajectory_df = pd.DataFrame(trajectory_records)
        trajectory_df.to_csv(trajectory_csv_filename, index=False)
        print(f"Trajectory results saved to {trajectory_csv_filename}")

    summary = {
        "mean_final_corruption_rate": results_df["final_corruption_rate"].mean(),
        "std_final_corruption_rate": results_df["final_corruption_rate"].std(),
        "min_final_corruption_rate": results_df["final_corruption_rate"].min(),
        "max_final_corruption_rate": results_df["final_corruption_rate"].max(),
        "mean_spread_amount": results_df["spread_amount"].mean(),
        "mean_spread_rate_change": results_df["spread_rate_change"].mean(),
        "prob_full_corruption": results_df["full_corruption"].mean(),
        "prob_no_spread": results_df["no_spread"].mean(),
        "all_final_corruption_rates": results_df["final_corruption_rate"].to_numpy(),
        "all_final_num_corrupted": results_df["final_num_corrupted"].to_numpy(),
        "all_initial_num_corrupted": results_df["initial_num_corrupted"].to_numpy(),
        "run_results": results_df
    }

    return summary