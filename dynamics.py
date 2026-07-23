import numpy as np
from voting import singleton_belief_update

def initialize_states(N, initial_corruption_rate, rng):

    # Initialize the states of the nodes in the network.
    corrupted = rng.random(N) < initial_corruption_rate

    return corrupted


def initialize_beliefs(N, initial_corruption_rate):

    # Initialize the beliefs of the nodes in the network.
    belief = np.full((N,N), initial_corruption_rate, dtype=float)

    return belief


def apply_contagion(corrupted, pairs, beta, rng):

    # Apply the contagion process based on the collaboration pairs and the corruption probability.
    next_corrupted = corrupted.copy()

    for i, j in pairs:

        # If one of the nodes is corrupted and the other is not, there is a chance for the uncorrupted node to become corrupted.
        if corrupted[i] != corrupted[j]:
            if rng.random() < beta:
                next_corrupted[i] = True
                next_corrupted[j] = True

    return next_corrupted


def update_beliefs(belief, corrupted, next_corrupted, pairs, beta):
# Update beliefs for direct-collaboration case
    new_belief = belief.copy()

    for i, j in pairs:

        # Update the beliefs based on the current and next corrupted states of the nodes in the collaboration pairs.
        if not corrupted[i]:
            if next_corrupted[i]:
                # i became corrupted, so j must have been corrrupted.
                new_belief[i, j] = 1.0
            else:
                # i did not become corrupted, update by Bayes' rule.
                new_belief[i,j] = belief[i,j]*(1-beta)/(1-beta*belief[i,j])
        
        # Update the beliefs for the other node in the pair.
        if not corrupted[j]:
            if next_corrupted[j]:
                # j became corrupted, so i must have been corrrupted.
                new_belief[j, i] = 1.0
            else:
                # j did not become corrupted, update by Bayes' rule.
                new_belief[j,i] = belief[j,i]*(1-beta)/(1-beta*belief[j,i])
    
    return new_belief
    

def update_beliefs_nonpartners(belief, neighbors, pairs, vote_correct,  solo_accuracy):
# Update beliefs for no-direct_collaboration case
    new_belief = belief.copy()

    paired_edges = set()

    for i, j in pairs:
        paired_edges.add((i, j))
        paired_edges.add((j, i))

    N = len(neighbors)

    for i in range(N):
        for j in neighbors[i]:

            if (i, j) in paired_edges:
                continue

            new_belief[i, j] = singleton_belief_update(
                prior_belief=belief[i,j],
                observed_correct=vote_correct[j],
                solo_accuracy=solo_accuracy
            )
    
    return new_belief