import networkx as nx


def ring_lattice_network(N, degree=4):
    graph = nx.watts_strogatz_graph(
        n=N,
        k=degree,
        p=0,
    )

    neighbors = [
        list(graph.neighbors(i))
        for i in range(N)
    ]

    return graph, neighbors


def small_world_network(N, degree=4, rewiring_probability=0.1, seed=None):
    graph = nx.connected_watts_strogatz_graph(
        n=N,
        k=degree,
        p=rewiring_probability,
        seed=seed,
    )

    neighbors = [
        list(graph.neighbors(i))
        for i in range(N)
    ]

    return graph, neighbors


def random_regular_network(N, degree=4, seed=None):
    graph = nx.random_regular_graph(
        d=degree,
        n=N,
        seed=seed,
    )

    neighbors = [
        list(graph.neighbors(i))
        for i in range(N)
    ]

    return graph, neighbors
