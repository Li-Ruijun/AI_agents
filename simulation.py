import numpy as np
import pandas as pd

from network import ring_network
from dynamics import initialize_states, initialize_beliefs, apply_contagion, update_beliefs, update_beliefs_nonpartners
from proposal_rules import make_proposals, collaboration_pairs
from cutoff import calculate_belief_cutoff, accuracy
from voting import singleton_vote_correctness

def find_corruption_stabilization_period(initial_num_corrupted, num_corrupted_history):

    # Find the last period in which the number of corrupted agents increased.
    corruption_counts = np.concatenate(([initial_num_corrupted], np.asarray(num_corrupted_history, dtype=int)))

    increase_periods = np.flatnonzero(np.diff(corruption_counts) > 0)

    if len(increase_periods) == 0:
        return 0

    stabilization_period = int(increase_periods[-1] + 1)

    return stabilization_period

def count_collaboration_types(corrupted, pairs):
    # Count collaboration types according to agent states before contagion occurs.

    num_hh_pairs = 0
    num_hc_pairs = 0
    num_cc_pairs = 0

    for i, j in pairs:

        if not corrupted[i] and not corrupted[j]:
            num_hh_pairs += 1

        elif corrupted[i] and corrupted[j]:
            num_cc_pairs += 1

        else:
            num_hc_pairs +=1

    num_honest_collaborating = ( 2 * num_hh_pairs + num_hc_pairs)
    num_corrupted_collaborating = (2 * num_cc_pairs + num_hc_pairs)

    return {
        "num_hh_pairs": num_hh_pairs,
        "num_hc_pairs": num_hc_pairs,
        "num_cc_pairs": num_cc_pairs,
        "num_honest_collaborating":num_honest_collaborating,
        "num_corrupted_collaborating":num_corrupted_collaborating

    }
def one_simulation(N, T, initial_corruption_rate, beta, gamma, alpha, memory_endowment, collaboration_cost, discount_factor, seed=None):

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
    belief_cutoff_history = []

    # Store collaboration air composition over time.
    num_hh_pairs_history = []
    num_hc_pairs_history = []
    num_cc_pairs_history = []

    # Store the number of collaborating agents by state.
    num_honest_collaborating_history = []
    num_corrupted_collaborating_history = []

    for t in range(T):


        # Calculate endogenous belief cutoff
        belief_cutoff = calculate_belief_cutoff(
            t=t,
            T=T,
            beta=beta,
            gamma=gamma,
            alpha=alpha,
            memory_endowment=memory_endowment,
            collaboration_cost=collaboration_cost,
            discount_factor=discount_factor
        )

        belief_cutoff_history.append(belief_cutoff)

        # Each agent makes a proposal.
        proposals = make_proposals(
            N=N,
            neighbors=neighbors,
            belief=belief,
            belief_cutoff=belief_cutoff,
            corrupted=corrupted,
            rng=rng)

        # Mutual proposal from collaboration pairs.
        partners, pairs = collaboration_pairs(N=N, proposals=proposals)

        # Record the collaboration rate.
        collaboration_rate = np.mean(partners != -1)
        collaboration_rate_history.append(collaboration_rate)

        # Counr HH, HC and CC collaboration pairs.
        collaboration_types = count_collaboration_types(corrupted=corrupted, pairs=pairs)

        num_hh_pairs_history.append(collaboration_types["num_hh_pairs"])
        num_hc_pairs_history.append(collaboration_types["num_hc_pairs"])
        num_cc_pairs_history.append(collaboration_types["num_cc_pairs"])
        num_honest_collaborating_history.append(collaboration_types["num_honest_collaborating"])
        num_corrupted_collaborating_history.append(collaboration_types["num_corrupted_collaborating"])

        # Generate voting outcomes
        solo_accuracy = accuracy(
            memory=memory_endowment,
            gamma=gamma,
            alpha=alpha
        )

        vote_correct = singleton_vote_correctness(
            corrupted=corrupted,
            solo_accuracy=solo_accuracy,
            rng=rng
        )

        # Apply contagion.
        next_corrupted = apply_contagion(corrupted=corrupted, pairs=pairs, beta=beta, rng=rng)

        # Update beliefs based on the current and next corrupted states.
        belief = update_beliefs(belief=belief,
                                corrupted=corrupted,
                                next_corrupted=next_corrupted,
                                pairs=pairs,
                                beta=beta)

        belief = update_beliefs_nonpartners(
            belief=belief,
            neighbors=neighbors,
            pairs=pairs,
            vote_correct=vote_correct,
            solo_accuracy=solo_accuracy
        )

        # Update the corrupted states for the next iteration.
        corrupted = next_corrupted

        # Record the current statistics at each time step.
        num_corrupted = int(np.sum(corrupted))
        num_honest = N - num_corrupted
        corruption_rate = num_corrupted/N
        
        num_corrupted_history.append(num_corrupted)
        num_honest_history.append(num_honest)
        corruption_rate_history.append(corruption_rate)

        corruption_stabilization_period = (find_corruption_stabilization_period(initial_num_corrupted=initial_num_corrupted, num_corrupted_history=num_corrupted_history))
    # Store the statistics for the current time step.
        results = {
        "initial_num_corrupted":initial_num_corrupted,
        "initial_corruption_rate_actual":initial_corruption_rate_actual,
        "num_corrupted":np.array(num_corrupted_history),
        "num_honest":np.array(num_honest_history),
        "corruption_rate":np.array(corruption_rate_history),
        "collaboration_rate":np.array(collaboration_rate_history),
        "belief_cutoff":np.array(belief_cutoff_history),
        "num_hh_pairs":np.array(num_hh_pairs_history),
        "num_hc_pairs":np.array(num_hc_pairs_history),
        "num_cc_pairs":np.array(num_cc_pairs_history),
        "num_honest_collaborating":np.array(num_honest_collaborating_history),
        "num_corrupted_collaborating":np.array(num_corrupted_collaborating_history),
        "corruption_stabilization_period":corruption_stabilization_period,
        "final_num_corrupted":int(np.sum(corrupted)),
        "final_num_honest":int(N - np.sum(corrupted)),
        "final_corruption_rate":float(np.mean(corrupted)),
        "final_states":corrupted,
        "final_beliefs":belief
    }

    return results
    

def many_simulations(
        num_simulations,
        N,
        T,
        initial_corruption_rate,
        beta,
        gamma,
        alpha,
        memory_endowment,
        collaboration_cost,
        discount_factor,
        save_csv,
        csv_filename,
        save_trajectories,
        num_trajectories,
        trajectory_csv_filename
        ):

    run_records = []
    trajectory_records = []
    

    for simulation in range(num_simulations):
        results = one_simulation(
            N=N,
            T=T,
            initial_corruption_rate=initial_corruption_rate,
            beta=beta,
            gamma=gamma,
            alpha=alpha,
            memory_endowment=memory_endowment,
            collaboration_cost=collaboration_cost,
            discount_factor=discount_factor,
            seed=simulation)

        # Convert the collaboration-rate history into a NumPy array.
        collaboration_history = np.asarray(results["collaboration_rate"],dtype=float)
        
        if collaboration_history.size == 0:
            raise ValueError(
                "The collaboration-rate history is empty."
            )
        
        # Collaboration rate in the final simulation period.
        final_collaboration_rate = float(collaboration_history[-1])
        
        # Average collaboration rate across the entire simulation horizon.
        mean_collaboration_rate_over_horizon = float(collaboration_history.mean())
        
        if (simulation + 1) % 100 == 0:
            print(f"Completed {simulation + 1}/{num_simulations} simulations")

        run_record = {
            "run_id": simulation,
            "N": N,
            "T": T,
            "initial_corruption_rate_parameter": initial_corruption_rate,
            "beta": beta,
            "gamma":gamma,
            "alpha":alpha,
            "memory_endowment":memory_endowment,
            "collaboration_cost":collaboration_cost,
            "discount_factor":discount_factor,
            "initial_num_corrupted": results["initial_num_corrupted"],
            "initial_corruption_rate_actual": results["initial_corruption_rate_actual"],
            "final_num_corrupted": results["final_num_corrupted"],
            "final_num_honest": results["final_num_honest"],
            "final_corruption_rate": results["final_corruption_rate"],
            "spread_amount": results["final_num_corrupted"] - results["initial_num_corrupted"],
            "spread_rate_change": results["final_corruption_rate"] - results["initial_corruption_rate_actual"],
            "full_corruption": results["final_num_corrupted"] == N,
            "no_spread": results["final_num_corrupted"] <= results["initial_num_corrupted"],
            "belief_cutoff": float(results["belief_cutoff"].mean()),
            "final_collaboration_rate":final_collaboration_rate,
            "mean_collaboration_rate_over_horizon":mean_collaboration_rate_over_horizon,
            "corruption_stabilization_period":results["corruption_stabilization_period"],
            }

        run_records.append(run_record)

        if save_trajectories and simulation < num_trajectories:
            for t in range(len(results["corruption_rate"])):
                trajectory_record = {
                    "run_id": simulation,
                    "time_step": t,
                    "period": t + 1,
                    "N": N,
                    "T": T,
                    "initial_corruption_rate_parameter":initial_corruption_rate,
                    "beta": beta,
                    "gamma": gamma,
                    "alpha": alpha,
                    "memory_endowment":memory_endowment,
                    "collaboration_cost":collaboration_cost,
                    "discount_factor":discount_factor,
                    "num_corrupted":results["num_corrupted"][t],
                    "num_honest":results["num_honest"][t],
                    "corruption_rate":results["corruption_rate"][t],
                    "collaboration_rate":results["collaboration_rate"][t],
                    "belief_cutoff":results["belief_cutoff"][t],
                    "corruption_stabilization_period":results["corruption_stabilization_period"],
                    "num_hh_pairs":results["num_hh_pairs"][t],
                    "num_hc_pairs":results["num_hc_pairs"][t],
                    "num_cc_pairs":results["num_cc_pairs"][t],
                    "num_honest_collaborating":results["num_honest_collaborating"][t],
                    "num_corrupted_collaborating":results["num_corrupted_collaborating"][t]
                }

                trajectory_records.append(trajectory_record)

    
    results_df = pd.DataFrame(run_records)

    if save_csv:
        results_df.to_csv(csv_filename, index=False)
        print(f"Simulation results saved to {csv_filename}")

    if save_trajectories:
        trajectory_df = pd.DataFrame(trajectory_records)
        trajectory_df.to_csv(trajectory_csv_filename, index=False)
        print(f"Trajectory results saved to {trajectory_csv_filename}")

    summary = {"mean_final_corruption_rate": (results_df["final_corruption_rate"].mean()),
               "std_final_corruption_rate": (results_df["final_corruption_rate"].std(ddof=1)),
                "min_final_corruption_rate": (results_df["final_corruption_rate"].min()),
                "max_final_corruption_rate": (results_df["final_corruption_rate"].max()),
                "mean_spread_amount": (results_df["spread_amount"].mean()),
                "mean_spread_rate_change": (results_df["spread_rate_change"].mean()),
                "prob_full_corruption": (results_df["full_corruption"].mean()),
                "prob_no_spread": (results_df["no_spread"].mean()),
                "mean_final_collaboration_rate": (results_df["final_collaboration_rate"].mean()),
                "variance_final_collaboration_rate": (results_df["final_collaboration_rate"].var(ddof=1)),
                "std_final_collaboration_rate": (results_df["final_collaboration_rate"].std(ddof=1)),
                "mean_full_horizon_collaboration_rate": (results_df["mean_collaboration_rate_over_horizon"].mean()),
                "variance_full_horizon_collaboration_rate": (results_df["mean_collaboration_rate_over_horizon"].var(ddof=1)),
                "std_full_horizon_collaboration_rate": (results_df["mean_collaboration_rate_over_horizon"].std(ddof=1)),
                "all_final_corruption_rates": (results_df["final_corruption_rate"].to_numpy()),
                "all_final_num_corrupted": (results_df["final_num_corrupted"].to_numpy()),
                "all_initial_num_corrupted": (results_df["initial_num_corrupted"].to_numpy()),
                "all_final_collaboration_rates": (results_df["final_collaboration_rate"].to_numpy()),
                "all_mean_collaboration_rates_over_horizon": (results_df["mean_collaboration_rate_over_horizon"].to_numpy()),
                "run_results":results_df
                }

    return summary