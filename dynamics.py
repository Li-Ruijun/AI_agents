import numpy as np


def initialize_states(N, initial_corruption_rate, rng):

    return rng.random(N) < initial_corruption_rate


def initialize_beliefs(N, initial_corruption_rate):

    return np.full((N, N), initial_corruption_rate, dtype=float)


def direct_nonconversion_update(prior_belief, beta):

    denominator = 1 - beta * prior_belief

    if denominator <= 1e-12:
        return 0.0

    posterior = prior_belief * (1 - beta) / denominator

    return float(np.clip(posterior, 0.0, 1.0))


def singleton_pool_belief_update(prior_j, observed_correct, solo_accuracy):

    prior_j = float(prior_j)
    A = float(solo_accuracy)

    if observed_correct:
        likelihood_honest = A
        likelihood_corrupt = 1 - A
    else:
        likelihood_honest = 1 - A
        likelihood_corrupt = A

    denominator = (1 - prior_j) * likelihood_honest + prior_j * likelihood_corrupt

    if denominator <= 0:
        raise ZeroDivisionError("The singleton observation has zero likelihood.")

    posterior = prior_j * likelihood_corrupt / denominator

    return float(np.clip(posterior, 0.0, 1.0))


def observed_pair_belief_update(prior_j, prior_k, observed_correct, pair_accuracy, beta):

    prior_j = float(prior_j)
    prior_k = float(prior_k)
    A = float(pair_accuracy)

    # Construct the joint prior.
    prior_HH = (1 - prior_j) * (1 - prior_k)
    prior_HC = (1 - prior_j) * prior_k
    prior_CH = prior_j * (1 - prior_k)
    prior_CC = prior_j * prior_k

    # Determine the vote likelihoods.
    if observed_correct:
        likelihood_no_reversal = A
        likelihood_reversal = 1 - A
    else:
        likelihood_no_reversal = 1 - A
        likelihood_reversal = A

    likelihood_mixed = (1 - beta) * likelihood_no_reversal + beta * likelihood_reversal

    denominator = (
        prior_HH * likelihood_no_reversal
        + prior_HC * likelihood_mixed
        + prior_CH * likelihood_mixed
        + prior_CC * likelihood_reversal
    )

    if denominator <= 0:
        raise ZeroDivisionError("The observed pair outcome has zero likelihood.")

    # Include agents already corrupted and agents converted in a mixed pair.
    posterior_j = (
        prior_CH * likelihood_mixed
        + prior_CC * likelihood_reversal
        + prior_HC * beta * likelihood_reversal
    ) / denominator

    posterior_k = (
        prior_HC * likelihood_mixed
        + prior_CC * likelihood_reversal
        + prior_CH * beta * likelihood_reversal
    ) / denominator

    return float(np.clip(posterior_j, 0.0, 1.0)), float(np.clip(posterior_k, 0.0, 1.0))


def update_beliefs(
    belief, neighbors, partners, pairs, corrupted, next_corrupted,
    vote_correct, solo_accuracy, pair_accuracy, beta
):

    prior_belief = belief.copy()
    new_belief = prior_belief.copy()

    # Update direct collaborators.
    for i, j in pairs:
        if not corrupted[i]:
            if next_corrupted[i]:
                new_belief[i, j] = 1.0
            else:
                new_belief[i, j] = direct_nonconversion_update(prior_belief[i, j], beta)

        if not corrupted[j]:
            if next_corrupted[j]:
                new_belief[j, i] = 1.0
            else:
                new_belief[j, i] = direct_nonconversion_update(prior_belief[j, i], beta)

    # Update non-direct neighbours.
    N = len(neighbors)

    for i in range(N):
        processed_pairs = set()

        for j in neighbors[i]:
            if partners[i] == j:
                continue

            observed_correct = bool(vote_correct[j])
            k = int(partners[j])

            if k == -1:
                new_belief[i, j] = singleton_pool_belief_update(
                    prior_j=prior_belief[i, j],
                    observed_correct=observed_correct,
                    solo_accuracy=solo_accuracy,
                )
                continue

            if k == i:
                continue

            observed_pair = tuple(sorted((int(j), k)))

            if observed_pair in processed_pairs:
                continue

            posterior_j, posterior_k = observed_pair_belief_update(
                prior_j=prior_belief[i, j],
                prior_k=prior_belief[i, k],
                observed_correct=observed_correct,
                pair_accuracy=pair_accuracy,
                beta=beta,
            )

            new_belief[i, j] = posterior_j
            new_belief[i, k] = posterior_k
            processed_pairs.add(observed_pair)

    return new_belief