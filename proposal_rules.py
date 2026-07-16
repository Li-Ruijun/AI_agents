import numpy as np

def make_proposals(N, neighbors, belief, belief_threshold, rng):

    # Make proposals based on the beliefs of the agents in the network.
    proposals = np.full(N, -1, dtype=int)

    for i in range(N):
        candidate_neighbors = []

        for j in neighbors[i]:
            if belief[i, j] <= belief_threshold:
                candidate_neighbors.append(j)

        if len(candidate_neighbors) > 0:
                # Choose the neighbor with the lowest belief
                min_belief = min(belief[i, j] for j in candidate_neighbors)
                
                best_neighbors = [j for j in candidate_neighbors if belief[i, j] == min_belief]
                proposals[i] = rng.choice(best_neighbors)
        
    return proposals


def collaboration_pairs(N, proposals):

    # Determine the collaboration pairs based on the proposals made by the nodes.
    partners = np.full(N, -1, dtype=int)
    pairs = []

    for i in range(N):
        j = proposals[i]

        # If the proposal is invalid, skip to the next agent.
        if j == -1:
            continue
        
        # Check if the proposal is mutual and both agents are unpaired.
        if proposals[j] == i:
            if partners[i] == -1 and partners[j] == -1:
                partners[i] = j
                partners[j] = i
                pairs.append((i, j))
    
    return partners, pairs
