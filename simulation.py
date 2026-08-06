from pathlib import Path

import numpy as np
import pandas as pd

from cutoff import accuracy, calculate_belief_cutoff
from dynamics import initialize_beliefs, initialize_states, update_beliefs
from network import ring_network
from proposal_rules import collaboration_pairs, make_proposals
from voting import build_decision_pools, generate_pool_outcomes, majority_decision


def one_simulation(
    N, T, initial_corruption_rate, task_probability, beta, gamma, alpha,
    memory_endowment, collaboration_cost, discount_factor, seed=None
):

    if N < 3:
        raise ValueError("N must be at least 3 for a ring network.")

    if T <= 0:
        raise ValueError("T must be positive.")

    if not 0 <= initial_corruption_rate <= 1:
        raise ValueError("initial_corruption_rate must lie in [0, 1].")

    if not 0 <= task_probability <= 1:
        raise ValueError("task_probability must lie in [0, 1].")

    if not 0 <= beta <= 1:
        raise ValueError("beta must lie in [0, 1].")

    rng = np.random.default_rng(seed)

    # Initialise the model.
    neighbors = ring_network(N)
    corrupted = initialize_states(N, initial_corruption_rate, rng)
    belief = initialize_beliefs(N, initial_corruption_rate)

    initial_num_corrupted = int(np.sum(corrupted))
    initial_corruption_rate_actual = float(np.mean(corrupted))

    solo_accuracy = float(accuracy(memory=memory_endowment, gamma=gamma, alpha=alpha))
    pair_accuracy = float(accuracy(memory=2 * memory_endowment, gamma=gamma, alpha=alpha))

    num_corrupted_history = []
    num_honest_history = []
    corruption_rate_history = []
    collaboration_rate_history = []
    belief_cutoff_history = []

    task_state_history = []
    collective_decision_history = []

    raw_recommendation_accuracy_history = []
    submitted_vote_accuracy_history = []
    collective_decision_accuracy_history = []
    clean_pool_accuracy_history = []

    reversal_rate_history = []

    num_pools_history = []
    num_pair_pools_history = []
    num_singleton_pools_history = []
    num_mixed_pools_history = []

    num_hh_pairs_history = []
    num_hc_pairs_history = []
    num_cc_pairs_history = []

    num_reversed_pools_history = []
    num_conversions_history = []

    for t in range(T):

        belief_cutoff = calculate_belief_cutoff(
            t=t,
            T=T,
            beta=beta,
            gamma=gamma,
            alpha=alpha,
            memory_endowment=memory_endowment,
            collaboration_cost=collaboration_cost,
            discount_factor=discount_factor,
        )
        belief_cutoff_history.append(belief_cutoff)

        # Form collaboration pairs.
        proposals = make_proposals(
            N=N,
            neighbors=neighbors,
            belief=belief,
            belief_cutoff=belief_cutoff,
            corrupted=corrupted,
            rng=rng,
        )

        partners, pairs = collaboration_pairs(N=N, proposals=proposals)
        collaboration_rate = float(np.mean(partners != -1))
        collaboration_rate_history.append(collaboration_rate)

        # Generate recommendations, votes and conversions.
        pools = build_decision_pools(N=N, pairs=pairs)
        theta = int(rng.random() < task_probability)

        next_corrupted, votes, vote_correct, period_statistics = generate_pool_outcomes(
            pools=pools,
            corrupted=corrupted,
            theta=theta,
            beta=beta,
            gamma=gamma,
            alpha=alpha,
            memory_endowment=memory_endowment,
            rng=rng,
        )

        collective_decision = majority_decision(votes=votes, rng=rng)
        submitted_vote_accuracy = float(np.mean(vote_correct))
        collective_decision_correct = int(collective_decision == theta)

        # Update beliefs.
        belief = update_beliefs(
            belief=belief,
            neighbors=neighbors,
            partners=partners,
            pairs=pairs,
            corrupted=corrupted,
            next_corrupted=next_corrupted,
            vote_correct=vote_correct,
            solo_accuracy=solo_accuracy,
            pair_accuracy=pair_accuracy,
            beta=beta,
        )

        corrupted = next_corrupted

        # Record period results.
        num_corrupted = int(np.sum(corrupted))
        num_honest = N - num_corrupted

        num_corrupted_history.append(num_corrupted)
        num_honest_history.append(num_honest)
        corruption_rate_history.append(num_corrupted / N)

        task_state_history.append(theta)
        collective_decision_history.append(collective_decision)

        raw_recommendation_accuracy_history.append(period_statistics["raw_recommendation_accuracy"])
        submitted_vote_accuracy_history.append(submitted_vote_accuracy)
        collective_decision_accuracy_history.append(collective_decision_correct)
        clean_pool_accuracy_history.append(period_statistics["clean_pool_accuracy"])

        reversal_rate_history.append(period_statistics["reversal_rate"])

        num_pools_history.append(period_statistics["num_pools"])
        num_pair_pools_history.append(period_statistics["num_pair_pools"])
        num_singleton_pools_history.append(period_statistics["num_singleton_pools"])
        num_mixed_pools_history.append(period_statistics["num_mixed_pools"])

        num_hh_pairs_history.append(period_statistics["num_hh_pairs"])
        num_hc_pairs_history.append(period_statistics["num_hc_pairs"])
        num_cc_pairs_history.append(period_statistics["num_cc_pairs"])

        num_reversed_pools_history.append(period_statistics["num_reversed_pools"])
        num_conversions_history.append(period_statistics["num_conversions"])

    results = {
        "initial_num_corrupted": initial_num_corrupted,
        "initial_corruption_rate_actual": initial_corruption_rate_actual,
        "num_corrupted": np.asarray(num_corrupted_history),
        "num_honest": np.asarray(num_honest_history),
        "corruption_rate": np.asarray(corruption_rate_history),
        "collaboration_rate": np.asarray(collaboration_rate_history),
        "belief_cutoff": np.asarray(belief_cutoff_history),
        "task_state": np.asarray(task_state_history),
        "collective_decision": np.asarray(collective_decision_history),
        "raw_recommendation_accuracy": np.asarray(raw_recommendation_accuracy_history),
        "submitted_vote_accuracy": np.asarray(submitted_vote_accuracy_history),
        "collective_decision_accuracy": np.asarray(collective_decision_accuracy_history),
        "clean_pool_accuracy": np.asarray(clean_pool_accuracy_history),
        "reversal_rate": np.asarray(reversal_rate_history),
        "num_pools": np.asarray(num_pools_history),
        "num_pair_pools": np.asarray(num_pair_pools_history),
        "num_singleton_pools": np.asarray(num_singleton_pools_history),
        "num_mixed_pools": np.asarray(num_mixed_pools_history),
        "num_hh_pairs": np.asarray(num_hh_pairs_history),
        "num_hc_pairs": np.asarray(num_hc_pairs_history),
        "num_cc_pairs": np.asarray(num_cc_pairs_history),
        "num_reversed_pools": np.asarray(num_reversed_pools_history),
        "num_conversions": np.asarray(num_conversions_history),
        "mean_raw_recommendation_accuracy": float(np.mean(raw_recommendation_accuracy_history)),
        "mean_submitted_vote_accuracy": float(np.mean(submitted_vote_accuracy_history)),
        "mean_collective_decision_accuracy": float(np.mean(collective_decision_accuracy_history)),
        "mean_clean_pool_accuracy": float(np.mean(clean_pool_accuracy_history)),
        "mean_reversal_rate": float(np.mean(reversal_rate_history)),
        "mean_num_hh_pairs": float(np.mean(num_hh_pairs_history)),
        "mean_num_hc_pairs": float(np.mean(num_hc_pairs_history)),
        "mean_num_cc_pairs": float(np.mean(num_cc_pairs_history)),
        "total_conversions": int(np.sum(num_conversions_history)),
        "final_num_corrupted": int(np.sum(corrupted)),
        "final_num_honest": int(N - np.sum(corrupted)),
        "final_corruption_rate": float(np.mean(corrupted)),
        "final_states": corrupted.copy(),
        "final_beliefs": belief.copy(),
    }

    return results


def many_simulations(
    num_simulations, N, T, initial_corruption_rate, task_probability, beta, gamma, alpha,
    memory_endowment, collaboration_cost, discount_factor, save_csv, csv_filename,
    save_trajectories, num_trajectories, trajectory_csv_filename
):

    if num_simulations <= 0:
        raise ValueError("num_simulations must be positive.")

    run_records = []
    trajectory_records = []

    print(f"Running {num_simulations} simulations with N={N} and T={T}...")

    for simulation in range(num_simulations):

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
            seed=simulation,
        )

        initial_num_honest = N - results["initial_num_corrupted"]
        spread_amount = results["final_num_corrupted"] - results["initial_num_corrupted"]
        spread_proportion_honest = spread_amount / initial_num_honest if initial_num_honest > 0 else 0.0

        run_record = {
            "run_id": simulation,
            "seed": simulation,
            "N": N,
            "T": T,
            "initial_corruption_rate_parameter": initial_corruption_rate,
            "task_probability": task_probability,
            "beta": beta,
            "gamma": gamma,
            "alpha": alpha,
            "memory_endowment": memory_endowment,
            "collaboration_cost": collaboration_cost,
            "discount_factor": discount_factor,
            "initial_num_corrupted": results["initial_num_corrupted"],
            "initial_num_honest": initial_num_honest,
            "initial_corruption_rate_actual": results["initial_corruption_rate_actual"],
            "final_num_corrupted": results["final_num_corrupted"],
            "final_num_honest": results["final_num_honest"],
            "final_corruption_rate": results["final_corruption_rate"],
            "spread_amount": spread_amount,
            "spread_rate_change": results["final_corruption_rate"] - results["initial_corruption_rate_actual"],
            "spread_proportion_honest": spread_proportion_honest,
            "full_corruption": results["final_num_corrupted"] == N,
            "no_spread": spread_amount == 0,
            "mean_collaboration_rate": float(np.mean(results["collaboration_rate"])),
            "belief_cutoff": float(np.mean(results["belief_cutoff"])),
            "mean_raw_recommendation_accuracy": results["mean_raw_recommendation_accuracy"],
            "mean_submitted_vote_accuracy": results["mean_submitted_vote_accuracy"],
            "mean_collective_decision_accuracy": results["mean_collective_decision_accuracy"],
            "mean_clean_pool_accuracy": results["mean_clean_pool_accuracy"],
            "mean_reversal_rate": results["mean_reversal_rate"],
            "mean_num_hh_pairs": results["mean_num_hh_pairs"],
            "mean_num_hc_pairs": results["mean_num_hc_pairs"],
            "mean_num_cc_pairs": results["mean_num_cc_pairs"],
            "total_conversions": results["total_conversions"],
        }
        run_records.append(run_record)

        if save_trajectories and simulation < num_trajectories:
            for t in range(T):
                num_pair_pools = int(results["num_pair_pools"][t])

                if num_pair_pools > 0:
                    hh_pair_rate = results["num_hh_pairs"][t] / num_pair_pools
                    hc_pair_rate = results["num_hc_pairs"][t] / num_pair_pools
                    cc_pair_rate = results["num_cc_pairs"][t] / num_pair_pools
                else:
                    hh_pair_rate = 0.0
                    hc_pair_rate = 0.0
                    cc_pair_rate = 0.0

                trajectory_record = {
                    "run_id": simulation,
                    "seed": simulation,
                    "time_step": t,
                    "period": t + 1,
                    "N": N,
                    "T": T,
                    "initial_corruption_rate_parameter": initial_corruption_rate,
                    "task_probability": task_probability,
                    "beta": beta,
                    "gamma": gamma,
                    "alpha": alpha,
                    "memory_endowment": memory_endowment,
                    "collaboration_cost": collaboration_cost,
                    "discount_factor": discount_factor,
                    "num_corrupted": results["num_corrupted"][t],
                    "num_honest": results["num_honest"][t],
                    "corruption_rate": results["corruption_rate"][t],
                    "collaboration_rate": results["collaboration_rate"][t],
                    "belief_cutoff": results["belief_cutoff"][t],
                    "task_state": results["task_state"][t],
                    "collective_decision": results["collective_decision"][t],
                    "raw_recommendation_accuracy": results["raw_recommendation_accuracy"][t],
                    "submitted_vote_accuracy": results["submitted_vote_accuracy"][t],
                    "collective_decision_accuracy": results["collective_decision_accuracy"][t],
                    "clean_pool_accuracy": results["clean_pool_accuracy"][t],
                    "reversal_rate": results["reversal_rate"][t],
                    "num_pools": results["num_pools"][t],
                    "num_pair_pools": num_pair_pools,
                    "num_singleton_pools": results["num_singleton_pools"][t],
                    "num_mixed_pools": results["num_mixed_pools"][t],
                    "num_hh_pairs": results["num_hh_pairs"][t],
                    "num_hc_pairs": results["num_hc_pairs"][t],
                    "num_cc_pairs": results["num_cc_pairs"][t],
                    "hh_pair_rate": hh_pair_rate,
                    "hc_pair_rate": hc_pair_rate,
                    "cc_pair_rate": cc_pair_rate,
                    "num_reversed_pools": results["num_reversed_pools"][t],
                    "num_conversions": results["num_conversions"][t],
                }
                trajectory_records.append(trajectory_record)

        completed = simulation + 1

        if completed % 100 == 0 or completed == num_simulations:
            print(f"Completed {completed}/{num_simulations} simulations.", flush=True)

    results_df = pd.DataFrame(run_records)

    if save_csv:
        csv_path = Path(csv_filename)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(csv_path, index=False)
        print(f"Simulation results saved to {csv_path}")

    if save_trajectories:
        trajectory_path = Path(trajectory_csv_filename)
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(trajectory_records).to_csv(trajectory_path, index=False)
        print(f"Trajectory results saved to {trajectory_path}")

    summary = {
        "mean_final_corruption_rate": results_df["final_corruption_rate"].mean(),
        "std_final_corruption_rate": results_df["final_corruption_rate"].std(),
        "min_final_corruption_rate": results_df["final_corruption_rate"].min(),
        "max_final_corruption_rate": results_df["final_corruption_rate"].max(),
        "mean_spread_amount": results_df["spread_amount"].mean(),
        "mean_spread_rate_change": results_df["spread_rate_change"].mean(),
        "mean_spread_proportion_honest": results_df["spread_proportion_honest"].mean(),
        "prob_full_corruption": results_df["full_corruption"].mean(),
        "prob_no_spread": results_df["no_spread"].mean(),
        "mean_collaboration_rate": results_df["mean_collaboration_rate"].mean(),
        "mean_raw_recommendation_accuracy": results_df["mean_raw_recommendation_accuracy"].mean(),
        "mean_submitted_vote_accuracy": results_df["mean_submitted_vote_accuracy"].mean(),
        "mean_collective_decision_accuracy": results_df["mean_collective_decision_accuracy"].mean(),
        "mean_clean_pool_accuracy": results_df["mean_clean_pool_accuracy"].mean(),
        "mean_reversal_rate": results_df["mean_reversal_rate"].mean(),
        "mean_num_hh_pairs": results_df["mean_num_hh_pairs"].mean(),
        "mean_num_hc_pairs": results_df["mean_num_hc_pairs"].mean(),
        "mean_num_cc_pairs": results_df["mean_num_cc_pairs"].mean(),
        "mean_total_conversions": results_df["total_conversions"].mean(),
        "all_final_corruption_rates": results_df["final_corruption_rate"].to_numpy(),
        "all_final_num_corrupted": results_df["final_num_corrupted"].to_numpy(),
        "all_initial_num_corrupted": results_df["initial_num_corrupted"].to_numpy(),
        "run_results": results_df,
    }

    return summary