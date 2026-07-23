import numpy as np

# For no-direct-collaboration case
def singleton_vote_correctness(corrupted, solo_accuracy, rng):
    
    # Generate vote correctness for each agent under the singleton approximation.
    N = len(corrupted)
    vote_correct = np.zeros(N, dtype=bool)

    for j in range(N):
        if not corrupted[j]:
            # If agent j is honest: P(vote is correct | j is honest) = a_s
            vote_correct[j] = rng.random() < solo_accuracy
        else:
            # If agent j is corrupted: P(vote is correct | j is corrupted) = 1 - a_s
            vote_correct[j] = rng.random() < (1 - solo_accuracy)

    return vote_correct


def singleton_belief_update(prior_belief, observed_correct, solo_accuracy):
# Bayesian belief update for a non-partner neighbour under singleton approximation.
# Agent i's belief about neighbour j being corrupted, after observing whether j's vote was correct

    b = prior_belief #belief[i, j]

    if observed_correct:

        # If j's vote is correct:
        # L^C(o = 1) = P(correct | corrupted) = 1 - a_s.
        likelihood_corrupt = 1 - solo_accuracy
        # L^H(o = 1) = P(correct | honest) = a_s.
        likelihood_honest = solo_accuracy
    else:
        
        # If j's vote is incorrect:
        # L^C(o = 0) = P(incorrect | corrupted) = a_s.
        likelihood_corrupt = solo_accuracy
        #  L^H(o = 0) = P(incorrect | honest) = 1 - a_s.
        likelihood_honest = 1 - solo_accuracy

    # Bayes' rule
    posterior = b * likelihood_corrupt/( b * likelihood_corrupt + (1 - b) * likelihood_honest)

    return posterior