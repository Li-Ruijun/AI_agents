import numpy as np

from cutoff import accuracy
from voting import majority_decision


def get_pool_votes(pools, votes):
    pool_votes = []

    for pool in pools:
        pool_vote = votes[pool[0]]

        for member in pool:
            if votes[member] != pool_vote:
                raise ValueError("Agents in the same pool must submit the same vote.")

        pool_votes.append(pool_vote)

    return np.asarray(pool_votes, dtype=int)


def agent_majority_decision(votes, rng):
    return majority_decision(votes=votes, rng=rng)


def pool_majority_decision(pools, votes, rng):
    pool_votes = get_pool_votes(pools, votes)
    return majority_decision(votes=pool_votes, rng=rng)


def accuracy_weighted_pool_decision(pools, votes, gamma, alpha, memory_endowment, rng):
    pool_votes = get_pool_votes(pools, votes)

    weight_zero = 0.0
    weight_one = 0.0

    for pool, vote in zip(pools, pool_votes):
        pool_memory = len(pool) * memory_endowment
        pool_weight = float(accuracy(memory=pool_memory, gamma=gamma, alpha=alpha))

        if vote == 1:
            weight_one += pool_weight
        else:
            weight_zero += pool_weight

    if weight_one > weight_zero:
        return 1

    if weight_zero > weight_one:
        return 0

    return int(rng.random() < 0.5)