from pathlib import Path
import sys

import numpy as np
import pandas as pd
import networkx as nx

# Allow this script to work either in the project root or inside an experiments/ folder.
PROJECT_ROOT = Path(__file__).resolve().parent
if not (PROJECT_ROOT / "simulation.py").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    T,
    alpha,
    discount_factor,
    gamma,
    memory_endowment,
    task_probability,
)

from network_variants import ring_lattice_network, small_world_network, random_regular_network
from simulation_network_experiment import one_network_simulation


# Experiment settings
NUM_SIMULATIONS = 10000
NUM_TRAJECTORIES = 7
RESUME_EXISTING = True


OUTPUT_DIR = (
    PROJECT_ROOT
    / "result_obsereable_pool"
    / "network_experiments"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Fixed parameters: only network structure changes in this experiment.
FIXED_PARAMETERS = {
    "N": 400,
    "initial_corruption_rate": 0.05,
    "beta": 0.50,
    "collaboration_cost": 0.02,
    "degree": 4,
}


NETWORK_TYPES = [
    "ring_lattice",
    "small_world",
    "random_regular",
]

REWIRING_PROBABILITY = 0.10


def build_settings():
    settings = []

    for setting_id, network_type in enumerate(NETWORK_TYPES, start=1):
        parameter_setting = (
            f"network={network_type}, "
            f"N={FIXED_PARAMETERS['N']}, "
            f"degree={FIXED_PARAMETERS['degree']}, "
            f"initial={FIXED_PARAMETERS['initial_corruption_rate']}, "
            f"beta={FIXED_PARAMETERS['beta']}, "
            f"cost={FIXED_PARAMETERS['collaboration_cost']}"
        )

        settings.append({
            "setting_id": setting_id,
            "parameter_setting": parameter_setting,
            "varied_parameter": "network_type",
            "varied_value": network_type,
            "network_type": network_type,
            "N": FIXED_PARAMETERS["N"],
            "initial_corruption_rate": FIXED_PARAMETERS["initial_corruption_rate"],
            "beta": FIXED_PARAMETERS["beta"],
            "collaboration_cost": FIXED_PARAMETERS["collaboration_cost"],
            "degree": FIXED_PARAMETERS["degree"],
        })

    return settings


def setting_file_label(setting):
    return setting["network_type"]


def generate_network(setting, network_seed):
    network_type = setting["network_type"]
    N = setting["N"]
    degree = setting["degree"]

    if network_type == "ring_lattice":
        graph, neighbors = ring_lattice_network(N=N, degree=degree)

    elif network_type == "small_world":
        graph, neighbors = small_world_network(
            N=N,
            degree=degree,
            rewiring_probability=REWIRING_PROBABILITY,
            seed=network_seed,
        )

    elif network_type == "random_regular":
        graph, neighbors = random_regular_network(N=N, degree=degree, seed=network_seed)

        attempt = 0
        while not nx.is_connected(graph):
            attempt += 1
            graph, neighbors = random_regular_network(N=N, degree=degree, seed=network_seed + attempt)

    else:
        raise ValueError(f"Unknown network type: {network_type}")

    return graph, neighbors


def existing_file_matches_setting(existing_results, setting):
    required_columns = {
        "parameter_setting",
        "network_type",
        "N",
        "initial_corruption_rate_parameter",
        "beta",
        "collaboration_cost",
        "target_degree",
    }

    if not required_columns.issubset(existing_results.columns):
        return False

    if len(existing_results) != NUM_SIMULATIONS:
        return False

    matches = (
        existing_results["network_type"].iloc[0] == setting["network_type"]
        and int(existing_results["N"].iloc[0]) == int(setting["N"])
        and np.isclose(existing_results["initial_corruption_rate_parameter"].iloc[0], setting["initial_corruption_rate"])
        and np.isclose(existing_results["beta"].iloc[0], setting["beta"])
        and np.isclose(existing_results["collaboration_cost"].iloc[0], setting["collaboration_cost"])
        and np.isclose(existing_results["target_degree"].iloc[0], setting["degree"])
    )

    if setting["network_type"] == "small_world":
        matches = matches and np.isclose(
            existing_results["rewiring_probability"].iloc[0],
            REWIRING_PROBABILITY,
        )

    return matches


def run_setting(setting):
    setting_id = setting["setting_id"]
    parameter_setting = setting["parameter_setting"]
    label = setting_file_label(setting)

    setting_results_path = OUTPUT_DIR / f"{label}_results.csv"
    setting_trajectories_path = OUTPUT_DIR / f"{label}_trajectories.csv"

    if RESUME_EXISTING and setting_results_path.exists():
        existing_results = pd.read_csv(setting_results_path)

        if existing_file_matches_setting(existing_results, setting):
            print(f"\nSetting {setting_id} already completed: {parameter_setting}")

            existing_trajectories = (
                pd.read_csv(setting_trajectories_path)
                if setting_trajectories_path.exists()
                else pd.DataFrame()
            )

            return existing_results, existing_trajectories

    run_records = []
    trajectory_records = []

    print(f"\nSetting {setting_id}: {parameter_setting}")

    for run_id in range(NUM_SIMULATIONS):
        simulation_seed = run_id
        network_seed = 100000 + run_id

        graph, neighbors = generate_network(setting, network_seed)

        degrees = np.array([degree for _, degree in graph.degree()])
        num_edges = graph.number_of_edges()
        mean_degree = float(np.mean(degrees))
        degree_std = float(np.std(degrees))
        clustering_coefficient = float(nx.average_clustering(graph))
        network_connected = nx.is_connected(graph)

        results = one_network_simulation(
            neighbors=neighbors,
            N=setting["N"],
            T=T,
            initial_corruption_rate=setting["initial_corruption_rate"],
            task_probability=task_probability,
            beta=setting["beta"],
            gamma=gamma,
            alpha=alpha,
            memory_endowment=memory_endowment,
            collaboration_cost=setting["collaboration_cost"],
            discount_factor=discount_factor,
            seed=simulation_seed,
        )

        initial_num_corrupted = results["initial_num_corrupted"]
        initial_num_honest = setting["N"] - initial_num_corrupted
        spread_amount = results["final_num_corrupted"] - initial_num_corrupted

        spread_proportion_honest = (
            spread_amount / initial_num_honest
            if initial_num_honest > 0
            else np.nan
        )

        spread_over_n = spread_amount / setting["N"]
        has_initial_corruption = initial_num_corrupted > 0

        no_spread_given_initial = (
            float(spread_amount == 0)
            if has_initial_corruption
            else np.nan
        )

        run_record = {
            "setting_id": setting_id,
            "parameter_setting": parameter_setting,
            "varied_parameter": setting["varied_parameter"],
            "varied_value": setting["varied_value"],
            "network_type": setting["network_type"],
            "run_id": run_id,
            "seed": simulation_seed,
            "network_seed": network_seed,
            "N": setting["N"],
            "T": T,
            "target_degree": setting["degree"],
            "num_edges": num_edges,
            "mean_degree": mean_degree,
            "degree_std": degree_std,
            "clustering_coefficient": clustering_coefficient,
            "network_connected": network_connected,
            "rewiring_probability": REWIRING_PROBABILITY if setting["network_type"] == "small_world" else np.nan,
            "initial_corruption_rate_parameter": setting["initial_corruption_rate"],
            "task_probability": task_probability,
            "beta": setting["beta"],
            "gamma": gamma,
            "alpha": alpha,
            "memory_endowment": memory_endowment,
            "collaboration_cost": setting["collaboration_cost"],
            "discount_factor": discount_factor,
            "initial_num_corrupted": initial_num_corrupted,
            "initial_num_honest": initial_num_honest,
            "initial_corruption_rate_actual": results["initial_corruption_rate_actual"],
            "final_num_corrupted": results["final_num_corrupted"],
            "final_num_honest": results["final_num_honest"],
            "final_corruption_rate": results["final_corruption_rate"],
            "spread_amount": spread_amount,
            "spread_rate_change": results["final_corruption_rate"] - results["initial_corruption_rate_actual"],
            "spread_over_n": spread_over_n,
            "spread_proportion_honest": spread_proportion_honest,
            "has_initial_corruption": has_initial_corruption,
            "full_corruption": results["final_num_corrupted"] == setting["N"],
            "no_spread": spread_amount == 0,
            "no_spread_given_initial": no_spread_given_initial,
            "mean_collaboration_rate": float(np.mean(results["collaboration_rate"])),
            "mean_belief_cutoff": float(np.mean(results["belief_cutoff"])),
            "mean_raw_recommendation_accuracy": results["mean_raw_recommendation_accuracy"],
            "mean_submitted_vote_accuracy": results["mean_submitted_vote_accuracy"],
            "mean_collective_decision_accuracy": results["mean_collective_decision_accuracy"],
            "mean_clean_pool_accuracy": results["mean_clean_pool_accuracy"],
            "mean_reversal_rate": results["mean_reversal_rate"],
            "mean_num_hh_pairs": results["mean_num_hh_pairs"],
            "mean_num_hc_pairs": results["mean_num_hc_pairs"],
            "mean_num_cc_pairs": results["mean_num_cc_pairs"],
            "total_conversions": results["total_conversions"],
        }

        run_records.append(run_record)

        if run_id < NUM_TRAJECTORIES:
            for t in range(T):
                num_pairs = int(results["num_pair_pools"][t])

                trajectory_record = {
                    "setting_id": setting_id,
                    "parameter_setting": parameter_setting,
                    "varied_parameter": setting["varied_parameter"],
                    "varied_value": setting["varied_value"],
                    "network_type": setting["network_type"],
                    "run_id": run_id,
                    "seed": simulation_seed,
                    "network_seed": network_seed,
                    "period": t + 1,
                    "N": setting["N"],
                    "T": T,
                    "target_degree": setting["degree"],
                    "num_edges": num_edges,
                    "mean_degree": mean_degree,
                    "degree_std": degree_std,
                    "clustering_coefficient": clustering_coefficient,
                    "initial_corruption_rate_parameter": setting["initial_corruption_rate"],
                    "task_probability": task_probability,
                    "beta": setting["beta"],
                    "gamma": gamma,
                    "alpha": alpha,
                    "memory_endowment": memory_endowment,
                    "collaboration_cost": setting["collaboration_cost"],
                    "discount_factor": discount_factor,
                    "num_corrupted": results["num_corrupted"][t],
                    "corruption_rate": results["corruption_rate"][t],
                    "collaboration_rate": results["collaboration_rate"][t],
                    "belief_cutoff": results["belief_cutoff"][t],
                    "num_pair_pools": num_pairs,
                    "num_hh_pairs": results["num_hh_pairs"][t],
                    "num_hc_pairs": results["num_hc_pairs"][t],
                    "num_cc_pairs": results["num_cc_pairs"][t],
                    "hh_pair_rate": results["num_hh_pairs"][t] / num_pairs if num_pairs > 0 else 0.0,
                    "hc_pair_rate": results["num_hc_pairs"][t] / num_pairs if num_pairs > 0 else 0.0,
                    "cc_pair_rate": results["num_cc_pairs"][t] / num_pairs if num_pairs > 0 else 0.0,
                    "num_conversions": results["num_conversions"][t],
                    "raw_recommendation_accuracy": results["raw_recommendation_accuracy"][t],
                    "submitted_vote_accuracy": results["submitted_vote_accuracy"][t],
                    "collective_decision_accuracy": results["collective_decision_accuracy"][t],
                }

                trajectory_records.append(trajectory_record)

        completed = run_id + 1

        if completed % 100 == 0 or completed == NUM_SIMULATIONS:
            print(f"Setting {setting_id} | {parameter_setting} | Completed {completed}/{NUM_SIMULATIONS} simulations.", flush=True)

    setting_results = pd.DataFrame(run_records)
    setting_trajectories = pd.DataFrame(trajectory_records)

    setting_results.to_csv(setting_results_path, index=False)
    setting_trajectories.to_csv(setting_trajectories_path, index=False)

    print(f"Setting {setting_id} results saved to {setting_results_path}")
    print(f"Setting {setting_id} trajectories saved to {setting_trajectories_path}")

    return setting_results, setting_trajectories


def build_summary(all_results):
    group_columns = [
        "setting_id",
        "parameter_setting",
        "varied_parameter",
        "varied_value",
        "network_type",
        "N",
        "target_degree",
        "initial_corruption_rate_parameter",
        "beta",
        "collaboration_cost",
    ]

    summary = (
        all_results
        .groupby(group_columns, dropna=False)
        .agg(
            num_simulations=("run_id", "count"),
            mean_num_edges=("num_edges", "mean"),
            mean_degree=("mean_degree", "mean"),
            mean_degree_std=("degree_std", "mean"),
            mean_clustering_coefficient=("clustering_coefficient", "mean"),
            mean_initial_corruption_rate=("initial_corruption_rate_actual", "mean"),
            probability_zero_initial_corruption=("has_initial_corruption", lambda x: 1.0 - x.mean()),
            mean_final_corruption_rate=("final_corruption_rate", "mean"),
            std_final_corruption_rate=("final_corruption_rate", "std"),
            min_final_corruption_rate=("final_corruption_rate", "min"),
            max_final_corruption_rate=("final_corruption_rate", "max"),
            mean_spread_amount=("spread_amount", "mean"),
            mean_spread_over_n=("spread_over_n", "mean"),
            mean_spread_proportion_honest=("spread_proportion_honest", "mean"),
            probability_no_spread=("no_spread", "mean"),
            probability_no_spread_given_initial=("no_spread_given_initial", "mean"),
            probability_full_corruption=("full_corruption", "mean"),
            mean_collaboration_rate=("mean_collaboration_rate", "mean"),
            mean_belief_cutoff=("mean_belief_cutoff", "mean"),
            mean_raw_recommendation_accuracy=("mean_raw_recommendation_accuracy", "mean"),
            mean_submitted_vote_accuracy=("mean_submitted_vote_accuracy", "mean"),
            mean_collective_decision_accuracy=("mean_collective_decision_accuracy", "mean"),
            mean_clean_pool_accuracy=("mean_clean_pool_accuracy", "mean"),
            mean_reversal_rate=("mean_reversal_rate", "mean"),
            mean_num_hh_pairs=("mean_num_hh_pairs", "mean"),
            mean_num_hc_pairs=("mean_num_hc_pairs", "mean"),
            mean_num_cc_pairs=("mean_num_cc_pairs", "mean"),
            mean_total_conversions=("total_conversions", "mean"),
        )
        .reset_index()
    )

    return summary


if __name__ == "__main__":
    settings = build_settings()
    settings_df = pd.DataFrame(settings)

    all_results = []
    all_trajectories = []

    settings_df.to_csv(OUTPUT_DIR / "parameter_settings.csv", index=False)

    print("Experiment: network structure analysis")
    print(f"Number of parameter settings: {len(settings)}")
    print(f"Simulations per setting: {NUM_SIMULATIONS}")

    print("\nParameter settings:")
    print(settings_df.to_string(index=False))

    for setting in settings:
        setting_results, setting_trajectories = run_setting(setting)
        all_results.append(setting_results)

        if not setting_trajectories.empty:
            all_trajectories.append(setting_trajectories)

    combined_results = pd.concat(all_results, ignore_index=True)
    combined_results.to_csv(OUTPUT_DIR / "all_simulation_results.csv", index=False)

    if all_trajectories:
        combined_trajectories = pd.concat(all_trajectories, ignore_index=True)
        combined_trajectories.to_csv(OUTPUT_DIR / "all_trajectory_results.csv", index=False)

    setting_summary = build_summary(combined_results)
    setting_summary.to_csv(OUTPUT_DIR / "setting_summary.csv", index=False)

    print("\nExperiment completed.")
    print(f"Results saved to {OUTPUT_DIR}")