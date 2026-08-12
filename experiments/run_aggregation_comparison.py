from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if not (PROJECT_ROOT / "simulation.py").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    T,
    alpha,
    beta,
    discount_factor,
    gamma,
    memory_endowment,
    task_probability,
)
from simulation_aggregation_experiment import one_aggregation_simulation


NUM_SIMULATIONS = 10000
NUM_TRAJECTORIES = 7
RESUME_EXISTING = True

OUTPUT_DIR = PROJECT_ROOT / "result_obsereable_pool" / "aggregation_experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIXED_PARAMETERS = {
    "N": 400,
    "collaboration_cost": 0.02,
}

INITIAL_CORRUPTION_RATES = [0.05, 0.10, 0.20, 0.30, 0.40]


def build_settings():
    settings = []

    for setting_id, initial_corruption_rate in enumerate(INITIAL_CORRUPTION_RATES):
        settings.append({
            "setting_id": setting_id,
            "parameter_setting": f"initial_corruption_{initial_corruption_rate:.2f}",
            "initial_corruption_rate": initial_corruption_rate,
        })

    return settings


def setting_file_label(initial_corruption_rate):
    return f"initial_corruption_{initial_corruption_rate:.2f}".replace(".", "_")


def existing_file_matches_setting(filename):
    if not filename.exists():
        return False

    try:
        existing_df = pd.read_csv(filename)
    except Exception:
        return False

    return len(existing_df) == NUM_SIMULATIONS


def run_setting(setting):
    initial_corruption_rate = setting["initial_corruption_rate"]
    label = setting_file_label(initial_corruption_rate)

    results_file = OUTPUT_DIR / f"{label}_results.csv"
    trajectory_file = OUTPUT_DIR / f"{label}_trajectories.csv"

    if RESUME_EXISTING and existing_file_matches_setting(results_file):
        print(f"Loading existing results for initial corruption = {initial_corruption_rate:.2f}")
        results_df = pd.read_csv(results_file)

        if trajectory_file.exists():
            trajectory_df = pd.read_csv(trajectory_file)
        else:
            trajectory_df = pd.DataFrame()

        return results_df, trajectory_df

    print(f"Running initial corruption = {initial_corruption_rate:.2f}")

    run_records = []
    trajectory_records = []

    for run_id in range(NUM_SIMULATIONS):
        results = one_aggregation_simulation(
            N=FIXED_PARAMETERS["N"],
            T=T,
            initial_corruption_rate=initial_corruption_rate,
            task_probability=task_probability,
            beta=beta,
            gamma=gamma,
            alpha=alpha,
            memory_endowment=memory_endowment,
            collaboration_cost=FIXED_PARAMETERS["collaboration_cost"],
            discount_factor=discount_factor,
            seed=run_id,
        )

        initial_num_honest = FIXED_PARAMETERS["N"] - results["initial_num_corrupted"]
        spread_amount = results["final_num_corrupted"] - results["initial_num_corrupted"]
        spread_proportion_honest = spread_amount / initial_num_honest if initial_num_honest > 0 else np.nan

        run_record = {
            "setting_id": setting["setting_id"],
            "parameter_setting": setting["parameter_setting"],
            "run_id": run_id,
            "seed": run_id,
            "N": FIXED_PARAMETERS["N"],
            "T": T,
            "initial_corruption_rate_parameter": initial_corruption_rate,
            "task_probability": task_probability,
            "beta": beta,
            "gamma": gamma,
            "alpha": alpha,
            "memory_endowment": memory_endowment,
            "collaboration_cost": FIXED_PARAMETERS["collaboration_cost"],
            "discount_factor": discount_factor,
            "initial_num_corrupted": results["initial_num_corrupted"],
            "initial_num_honest": initial_num_honest,
            "initial_corruption_rate_actual": results["initial_corruption_rate_actual"],
            "final_num_corrupted": results["final_num_corrupted"],
            "final_num_honest": results["final_num_honest"],
            "final_corruption_rate": results["final_corruption_rate"],
            "spread_amount": spread_amount,
            "spread_rate_change": results["final_corruption_rate"] - results["initial_corruption_rate_actual"],
            "spread_proportion_honest": spread_proportion_honest,
            "mean_collaboration_rate": float(np.mean(results["collaboration_rate"])),
            "mean_raw_recommendation_accuracy": results["mean_raw_recommendation_accuracy"],
            "mean_submitted_vote_accuracy": results["mean_submitted_vote_accuracy"],
            "mean_agent_majority_accuracy": results["mean_agent_majority_accuracy"],
            "mean_pool_majority_accuracy": results["mean_pool_majority_accuracy"],
            "mean_accuracy_weighted_accuracy": results["mean_accuracy_weighted_accuracy"],
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
                trajectory_record = {
                    "setting_id": setting["setting_id"],
                    "parameter_setting": setting["parameter_setting"],
                    "run_id": run_id,
                    "period": t,
                    "initial_corruption_rate_parameter": initial_corruption_rate,
                    "num_corrupted": results["num_corrupted"][t],
                    "corruption_rate": results["corruption_rate"][t],
                    "collaboration_rate": results["collaboration_rate"][t],
                    "agent_majority_accuracy": results["agent_majority_accuracy"][t],
                    "pool_majority_accuracy": results["pool_majority_accuracy"][t],
                    "accuracy_weighted_accuracy": results["accuracy_weighted_accuracy"][t],
                    "submitted_vote_accuracy": results["submitted_vote_accuracy"][t],
                }

                trajectory_records.append(trajectory_record)

    results_df = pd.DataFrame(run_records)
    trajectory_df = pd.DataFrame(trajectory_records)

    results_df.to_csv(results_file, index=False)
    trajectory_df.to_csv(trajectory_file, index=False)

    return results_df, trajectory_df


def build_summary(all_results):
    summary = all_results.groupby(
        ["setting_id", "parameter_setting", "initial_corruption_rate_parameter"],
        as_index=False,
    ).agg(
        num_simulations=("run_id", "count"),
        mean_initial_corruption_rate=("initial_corruption_rate_actual", "mean"),
        mean_final_corruption_rate=("final_corruption_rate", "mean"),
        mean_spread_amount=("spread_amount", "mean"),
        mean_spread_proportion_honest=("spread_proportion_honest", "mean"),
        mean_collaboration_rate=("mean_collaboration_rate", "mean"),
        mean_submitted_vote_accuracy=("mean_submitted_vote_accuracy", "mean"),
        mean_agent_majority_accuracy=("mean_agent_majority_accuracy", "mean"),
        mean_pool_majority_accuracy=("mean_pool_majority_accuracy", "mean"),
        mean_accuracy_weighted_accuracy=("mean_accuracy_weighted_accuracy", "mean"),
    )

    summary["pool_minus_agent"] = summary["mean_pool_majority_accuracy"] - summary["mean_agent_majority_accuracy"]
    summary["weighted_minus_agent"] = summary["mean_accuracy_weighted_accuracy"] - summary["mean_agent_majority_accuracy"]

    return summary


if __name__ == "__main__":
    settings = build_settings()

    all_results = []
    all_trajectories = []

    for setting in settings:
        results_df, trajectory_df = run_setting(setting)
        all_results.append(results_df)

        if not trajectory_df.empty:
            all_trajectories.append(trajectory_df)

    all_results_df = pd.concat(all_results, ignore_index=True)
    all_results_df.to_csv(OUTPUT_DIR / "all_simulation_results.csv", index=False)

    if all_trajectories:
        all_trajectories_df = pd.concat(all_trajectories, ignore_index=True)
        all_trajectories_df.to_csv(OUTPUT_DIR / "all_trajectory_results.csv", index=False)

    settings_df = pd.DataFrame(settings)
    settings_df.to_csv(OUTPUT_DIR / "parameter_settings.csv", index=False)

    summary_df = build_summary(all_results_df)
    summary_df.to_csv(OUTPUT_DIR / "setting_summary.csv", index=False)

    print()
    print(summary_df.to_string(index=False))
    print()
    print(f"Results saved to: {OUTPUT_DIR}")