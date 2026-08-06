import numpy as np

from cutoff import accuracy


def build_decision_pools(N, pairs):

    matched = np.zeros(N, dtype=bool)
    pools = []

    # Add collaboration pairs.
    for i, j in pairs:
        i, j = int(i), int(j)

        if i == j:
            raise ValueError("An agent cannot collaborate with itself.")

        if matched[i] or matched[j]:
            raise ValueError("The collaboration pairs do not form a valid matching.")

        pools.append((i, j))
        matched[i] = True
        matched[j] = True

    # Add unmatched agents as singleton pools.
    for i in np.flatnonzero(~matched):
        pools.append((int(i),))

    return pools


def generate_pool_outcomes(
    pools, corrupted, theta, beta, gamma, alpha, memory_endowment, rng
):

    corrupted = np.asarray(corrupted, dtype=bool)
    N = len(corrupted)

    if theta not in (0, 1):
        raise ValueError("theta must be either 0 or 1.")

    if not 0 <= beta <= 1:
        raise ValueError("beta must lie in [0, 1].")

    next_corrupted = corrupted.copy()
    votes = np.empty(N, dtype=np.int8)
    vote_correct = np.empty(N, dtype=bool)
    assigned = np.zeros(N, dtype=bool)

    num_pair_pools = 0
    num_singleton_pools = 0

    num_hh_pairs = 0
    num_hc_pairs = 0
    num_cc_pairs = 0

    num_reversed_pools = 0
    num_conversions = 0

    raw_correct_count = 0
    clean_accuracy_sum = 0.0

    for pool in pools:
        members = np.asarray(pool, dtype=int)

        if len(members) not in (1, 2):
            raise ValueError("Each decision pool must contain one or two agents.")

        if np.any(assigned[members]):
            raise ValueError("An agent was assigned to more than one decision pool.")

        assigned[members] = True

        if len(members) == 1:
            num_singleton_pools += 1
        else:
            num_pair_pools += 1

        pool_memory = len(members) * memory_endowment
        pool_accuracy = float(accuracy(memory=pool_memory, gamma=gamma, alpha=alpha))
        clean_accuracy_sum += pool_accuracy

        # Generate one raw recommendation for the pool.
        raw_correct = bool(rng.random() < pool_accuracy)
        raw_recommendation = theta if raw_correct else 1 - theta
        raw_correct_count += int(raw_correct)

        num_corrupted_in_pool = int(np.sum(corrupted[members]))
        reversed_recommendation = False

        # Record pair type before any new conversion.
        if len(members) == 2:
            if num_corrupted_in_pool == 0:
                num_hh_pairs += 1
            elif num_corrupted_in_pool == 1:
                num_hc_pairs += 1
            else:
                num_cc_pairs += 1

        # A fully corrupted pool always reverses its recommendation.
        if num_corrupted_in_pool == len(members):
            reversed_recommendation = True

        # An H-C pair reverses and converts with probability beta.
        elif 0 < num_corrupted_in_pool < len(members):
            if len(members) != 2:
                raise RuntimeError("A mixed pool must contain two agents.")

            mixed_event = bool(rng.random() < beta)

            if mixed_event:
                reversed_recommendation = True
                honest_members = members[~corrupted[members]]
                next_corrupted[honest_members] = True
                num_conversions += int(len(honest_members))

        num_reversed_pools += int(reversed_recommendation)

        submitted_vote = raw_recommendation ^ int(reversed_recommendation)
        votes[members] = submitted_vote
        vote_correct[members] = submitted_vote == theta

    if not np.all(assigned):
        raise RuntimeError("Not every agent was assigned to a decision pool.")

    num_pools = len(pools)

    period_statistics = {
        "num_pools": num_pools,
        "num_pair_pools": num_pair_pools,
        "num_singleton_pools": num_singleton_pools,
        "num_mixed_pools": num_hc_pairs,
        "num_hh_pairs": num_hh_pairs,
        "num_hc_pairs": num_hc_pairs,
        "num_cc_pairs": num_cc_pairs,
        "num_reversed_pools": num_reversed_pools,
        "num_conversions": num_conversions,
        "raw_recommendation_accuracy": raw_correct_count / num_pools,
        "clean_pool_accuracy": clean_accuracy_sum / num_pools,
        "reversal_rate": num_reversed_pools / num_pools,
    }

    return next_corrupted, votes, vote_correct, period_statistics


def majority_decision(votes, rng):

    votes = np.asarray(votes, dtype=int)

    if votes.ndim != 1:
        raise ValueError("votes must be a one-dimensional array.")

    if not np.all(np.isin(votes, [0, 1])):
        raise ValueError("Every submitted vote must be either 0 or 1.")

    num_vote_one = int(np.sum(votes))
    N = len(votes)

    if num_vote_one > N / 2:
        return 1

    if num_vote_one < N / 2:
        return 0

    return int(rng.random() < 0.5)