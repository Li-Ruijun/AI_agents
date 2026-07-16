def ring_network(N):
    # Create a ring network with N agents, where each agent is connected to its two immediate neighbors.
    neighbors = []

    for i in range(N):
        left_neighbor = (i - 1) % N
        right_neighbor = (i + 1) % N
        neighbors.append([left_neighbor, right_neighbor])

    return neighbors