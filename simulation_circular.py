import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ring_network(N):

# Create a ring network with N nodes.
    neighbors = []
    for i in range(N):
        left_neighbor = (i - 1) % N
        right_neighbor = (i + 1) % N
        neighbors.append((left_neighbor, right_neighbor))
    
    return neighbors


def initialize_states(N, initial_corruption_rate, rng):

    # Initialize the states of the nodes in the network.
    corrupted = rng.random(N) < initial_corruption_rate

    return corrupted


def initialize_beliefs(N, initial_corruption_rate):

    # Initialize the beliefs of the nodes in the network.
    belief = np.full((N,N), initial_corruption_rate, dtype=float)

    return belief


def make_proposals(N, neighbors, belief, belief_threshold, rng):

    # Make proposals based on the beliefs of the nodes in the network.
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

        # If the proposal is invalid, skip to the next node.
        if j == -1:
            continue
        
        # Check if the proposal is mutual and both nodes are unpaired.
        if proposals[j] == i:
            if partners[i] == -1 and partners[j] == -1:
                partners[i] = j
                partners[j] = i
                pairs.append((i, j))
    
    return partners, pairs


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
    

def one_simulation(N=50, T=10000, initial_corruption_rate=0.1, belief_threshold=0.5, beta=0.3, seed=None):

    rng = np.random.default_rng(seed)

    # Create a ring network with N nodes.
    neighbors = ring_network(N)

    # Initialize the states and beliefs of the nodes in the network.
    corrupted = initialize_states(N, initial_corruption_rate, rng)
    belief = initialize_beliefs(N, initial_corruption_rate)

    # Record the initial number of corrupted agents and the actual corruption rate.
    initial_num_corrupted = int(np.sum(corrupted))
    initial_corruption_rate_actual = float(np.mean(corrupted))
    
    # Store statistics over time.
    num_corrupted_history = []
    num_honest_history = []
    corruption_rate_history = []
    collaboration_rate_history = []

    for t in range(T):

        # Record the current statistics at each time step.
        num_corrupted = int(np.sum(corrupted))
        num_honest = N - num_corrupted
        corruption_rate = num_corrupted/N


        num_corrupted_history.append(num_corrupted)
        num_honest_history.append(num_honest)
        corruption_rate_history.append(corruption_rate)

        # Each agent makes a proposal.
        proposals = make_proposals(N=N, neighbors=neighbors, belief=belief, belief_threshold=belief_threshold, rng=rng)

        # Mutual proposal from collaboration pairs.
        partners, pairs = collaboration_pairs(N=N, proposals=proposals)

        # Record the collaboration rate.
        collaboration_rate = np.mean(partners != -1)
        collaboration_rate_history.append(collaboration_rate)

        # Apply contagion.
        next_corrupted = apply_contagion(corrupted=corrupted, pairs=pairs, beta=beta, rng=rng)

        # Update beliefs based on the current and next corrupted states.
        belief = update_beliefs(belief=belief, corrupted=corrupted, next_corrupted=next_corrupted, pairs=pairs, beta=beta)

        # Update the corrupted states for the next iteration.
        corrupted = next_corrupted

    # Store the statistics for the current time step.
    results = {
            "initial_num_corrupted": initial_num_corrupted,
            "initial_corruption_rate_actual": initial_corruption_rate_actual,
            "num_corrupted": np.array(num_corrupted_history),
            "num_honest": np.array(num_honest_history),
            "corruption_rate": np.array(corruption_rate_history),
            "collaboration_rate": np.array(collaboration_rate_history),
            "final_num_corrupted": int(np.sum(corrupted)),
            "final_num_honest": int(N - np.sum(corrupted)),
            "final_corruption_rate": float(np.mean(corrupted)),
            "final_states": corrupted,
            "final_beliefs": belief
    }

    return results
    

def many_simulations(num_simulations=100, N=50, T=10000, initial_corruption_rate=0.1, belief_threshold=0.5, beta=0.3, save_csv=True, csv_filename="simulation_results.csv"):

    run_records = []
    

    for simulation in range(num_simulations):
        results = one_simulation(N=N, T=T, initial_corruption_rate=initial_corruption_rate, belief_threshold=belief_threshold, beta=beta, seed=simulation)

        run_record = {
            "run_id": simulation,
            "N": N,
            "T": T,
            "initial_corruption_rate_parameter": initial_corruption_rate,
            "beta": beta,
            "belief_threshold": belief_threshold,
            "initial_num_corrupted": results["initial_num_corrupted"],
            "initial_corruption_rate_actual": results["initial_corruption_rate_actual"],
            "final_num_corrupted": results["final_num_corrupted"],
            "final_num_honest": results["final_num_honest"],
            "final_corruption_rate": results["final_corruption_rate"],
            "full_corruption": results["final_num_corrupted"] == N,
            "no_spread": results["final_num_corrupted"] <= results["initial_num_corrupted"]

        }

        run_records.append(run_record)

    results_df = pd.DataFrame(run_records)

    if save_csv:
        results_df.to_csv(csv_filename, index=False)

    summary = {
        "mean_final_corruption_rate": results_df["final_corruption_rate"].mean(),
        "std_final_corruption_rate": results_df["final_corruption_rate"].std(),
        "min_final_corruption_rate": results_df["final_corruption_rate"].min(),
        "max_final_corruption_rate": results_df["final_corruption_rate"].max(),
        "prob_full_corruption": results_df["full_corruption"].mean(),
        "prob_no_spread": results_df["no_spread"].mean(),
        "all_final_corruption_rates": results_df["final_corruption_rate"].to_numpy(),
        "all_final_num_corrupted": results_df["final_num_corrupted"].to_numpy(),
        "all_initial_num_corrupted": results_df["initial_num_corrupted"].to_numpy()
    }

    return summary


def plot_one_simulation(results):

    T = len(results["corruption_rate"])

    plt.figure(figsize=(12, 6))

    plt.plot( range(T), results["corruption_rate"], label="Corrupted", color='red')

    plt.plot( range(T), results["collaboration_rate"], label="Collaboration Rate", color='blue')

    plt.xlabel("Time period")
    plt.ylabel("Rate")
    plt.title("Contagion Simulation on Ring Network")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_many_simulations(summary):

    final_rates = summary["all_final_corruption_rates"]

    plt.figure(figsize=(10, 6))

    plt.hist(final_rates, bins=20, color='purple')

    plt.xlabel("Final Corrupted Rate")
    plt.ylabel("Frequency")
    plt.title("Distribution of Final Corrupted Rates over Many Simulations")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":

    # Parameters
    N = 50
    T = 10000
    initial_corruption_rate = 0.1
    belief_threshold = 0.5
    beta = 0.3

    # Run a single simulation and plot the results
    results = one_simulation(N=N, T=T, initial_corruption_rate=initial_corruption_rate, belief_threshold=belief_threshold, beta=beta, seed=42)

    print("Single simulation results:")
    print(f"Final number of honest agents: {results['final_num_honest']}")
    print(f"Final number of corrupted agents: {results['final_num_corrupted']}")
    print(f"Final corrupted rate: {results['final_corruption_rate']:.4f}")


    # Run many simulations and summarize the results
    summary = many_simulations(num_simulations=100, N=N, T=T, initial_corruption_rate=initial_corruption_rate, belief_threshold=belief_threshold, beta=beta, save_csv=True, csv_filename="simulation_results.csv")

    print("\nSummary of many simulations:")
    print(f"Mean final corrupted rate: {summary['mean_final_corruption_rate']:.4f}")
    print(f"Standard deviation of final corrupted rate: {summary['std_final_corruption_rate']:.4f}")
    print(f"Minimum final corrupted rate: {summary['min_final_corruption_rate']:.4f}")
    print(f"Maximum final corrupted rate: {summary['max_final_corruption_rate']:.4f}")
    print(f"Probability of full corruption: {summary['prob_full_corruption']:.4f}")
    print(f"Probability of no spread: {summary['prob_no_spread']:.4f}")
    
    plot_one_simulation(results)
    plot_many_simulations(summary)
