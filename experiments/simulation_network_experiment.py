from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from cutoff import accuracy, calculate_belief_cutoff
from dynamics import initialize_beliefs, initialize_states, update_beliefs
from proposal_rules import collaboration_pairs, make_proposals
from voting import build_decision_pools, generate_pool_outcomes, majority_decision


def one_network_simulation(
    neighbors,
    N,
    T,
    initial_corruption_rate,
    task_probability,
    beta,
    gamma,
    alpha,
    memory_endowment,
    collaboration_cost,
    discount_factor,
    seed=None,
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

