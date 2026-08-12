from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import T, task_probability, gamma, alpha, memory_endowment
from dynamics import initialize_states
from voting import build_decision_pools, generate_pool_outcomes, majority_decision


# Experiment settings
NUM_SIMULATIONS = 10000
NUM_TRAJECTORIES = 7
RESUME_EXISTING = True


# Fixed parameters
N = 400
INITIAL_CORRUPTION_RATE = 0.05

# These parameters do not affect collaboration formation here because
# collaboration is disabled explicitly, but they are retained for
# consistency with the baseline setting.
BETA = 0.5
COLLABORATION_COST = 0.02


# Output settings
OUTPUT_DIR = PROJECT_ROOT / "result_obsereable_pool" / "no_collaboration_experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = OUTPUT_DIR / "no_collaboration_results.csv"
TRAJECTORIES_PATH = OUTPUT_DIR / "no_collaboration_trajectories.csv"
SUMMARY_PATH = OUTPUT_DIR / "setting_summary.csv"


def one_simulation_no_collaboration(
    N, T, initial_corruption_rate, task_probability, beta,
    gamma, alpha, memory_endowment, seed=None
):
    if N <= 0:
        raise ValueError("N must be positive.")

    if T <= 0:
        raise ValueError("T must be positive.")

    if not 0 <= initial_corruption_rate <= 1:
        raise ValueError("initial_corruption_rate must lie in [0, 1].")

    if not 0 <= task_probability <= 1:
        raise ValueError("task_probability must lie in [0, 1].")

    rng = np.random.default_rng(seed)

    # Initialise corruption states.
    corrupted = initialize_states(N, initial_corruption_rate, rng)

    initial_num_corrupted = int(np.sum(corrupted))
    initial_corruption_rate_actual = float(np.mean(corrupted))

    # No collaboration: all agents remain singleton pools.
    pools = build_decision_pools(N=N, pairs=[])

    num_corrupted_history = []
    num_honest_history = []
    corruption_rate_history = []

    task_state_history = []
    collective_decision_history = []

    raw_recommendation_accuracy_history = []
    submitted_vote_accuracy_history = []
    collective_decision_accuracy_history = []
    clean_pool_accuracy_history = []

    reversal_rate_history = []
    num_reversed_pools_history = []

    for t in range(T):
        theta = int(rng.random() < task_probability)

        next_corrupted, votes, vote_correct, period_statistics = generate_pool_outcomes(
            pools=pools,
            corrupted=corrupted,
            theta=theta,
            beta=beta,
            gamma=gamma,
            alpha=alpha,
            memory_endowment=memory_endowment,
            rng=rng,
        )

        # No collaboration means corruption cannot spread.
        if period_statistics["num_conversions"] != 0:
            raise RuntimeError("Conversions occurred in the no-collaboration simulation.")

        if not np.array_equal(next_corrupted, corrupted):
            raise RuntimeError("Corruption states changed in the no-collaboration simulation.")

        collective_decision = majority_decision(votes=votes, rng=rng)
        submitted_vote_accuracy = float(np.mean(vote_correct))
        collective_decision_correct = int(collective_decision == theta)

        corrupted = next_corrupted

        num_corrupted = int(np.sum(corrupted))
        num_honest = N - num_corrupted

        num_corrupted_history.append(num_corrupted)
        num_honest_history.append(num_honest)
        corruption_rate_history.append(num_corrupted / N)

        task_state_history.append(theta)
        collective_decision_history.append(collective_decision)

        raw_recommendation_accuracy_history.append(
            period_statistics["raw_recommendation_accuracy"]
        )
        submitted_vote_accuracy_history.append(submitted_vote_accuracy)
        collective_decision_accuracy_history.append(collective_decision_correct)
        clean_pool_accuracy_history.append(period_statistics["clean_pool_accuracy"])

        reversal_rate_history.append(period_statistics["reversal_rate"])
        num_reversed_pools_history.append(period_statistics["num_reversed_pools"])

    results = {
        "initial_num_corrupted": initial_num_corrupted,
        "initial_corruption_rate_actual": initial_corruption_rate_actual,

        "num_corrupted": np.asarray(num_corrupted_history),
        "num_honest": np.asarray(num_honest_history),
        "corruption_rate": np.asarray(corruption_rate_history),

        "task_state": np.asarray(task_state_history),
        "collective_decision": np.asarray(collective_decision_history),

        "raw_recommendation_accuracy": np.asarray(raw_recommendation_accuracy_history),
        "submitted_vote_accuracy": np.asarray(submitted_vote_accuracy_history),
        "collective_decision_accuracy": np.asarray(collective_decision_accuracy_history),
        "clean_pool_accuracy": np.asarray(clean_pool_accuracy_history),

        "reversal_rate": np.asarray(reversal_rate_history),
        "num_reversed_pools": np.asarray(num_reversed_pools_history),

        "mean_raw_recommendation_accuracy": float(
            np.mean(raw_recommendation_accuracy_history)
        ),
        "mean_submitted_vote_accuracy": float(
            np.mean(submitted_vote_accuracy_history)
        ),
        "mean_collective_decision_accuracy": float(
            np.mean(collective_decision_accuracy_history)
        ),
        "mean_clean_pool_accuracy": float(np.mean(clean_pool_accuracy_history)),
        "mean_reversal_rate": float(np.mean(reversal_rate_history)),

        "mean_collaboration_rate": 0.0,
        "mean_num_hh_pairs": 0.0,
        "mean_num_hc_pairs": 0.0,
        "mean_num_cc_pairs": 0.0,
        "total_conversions": 0,

        "final_num_corrupted": int(np.sum(corrupted)),
        "final_num_honest": int(N - np.sum(corrupted)),
        "final_corruption_rate": float(np.mean(corrupted)),
    }

    return results


def run_experiment():
    if RESUME_EXISTING and RESULTS_PATH.exists():
        existing_results = pd.read_csv(RESULTS_PATH)

        required_columns = {
            "condition",
            "N",
            "initial_corruption_rate_parameter",
            "beta",
            "collaboration_cost",
        }

        correct_setting = (
            required_columns.issubset(existing_results.columns)
            and len(existing_results) == NUM_SIMULATIONS
            and int(existing_results["N"].iloc[0]) == N
            and np.isclose(
                existing_results["initial_corruption_rate_parameter"].iloc[0],
                INITIAL_CORRUPTION_RATE,
            )
            and np.isclose(existing_results["beta"].iloc[0], BETA)
            and np.isclose(
                existing_results["collaboration_cost"].iloc[0],
                COLLABORATION_COST,
            )
        )

        if correct_setting:
            print("No-collaboration experiment already completed.")
            return existing_results

    run_records = []
    trajectory_records = []

    print(
        f"Running no-collaboration experiment: N={N}, "
        f"initial corruption={INITIAL_CORRUPTION_RATE}, "
        f"simulations={NUM_SIMULATIONS}"
    )

    for run_id in range(NUM_SIMULATIONS):
        results = one_simulation_no_collaboration(
            N=N,
            T=T,
            initial_corruption_rate=INITIAL_CORRUPTION_RATE,
            task_probability=task_probability,
            beta=BETA,
            gamma=gamma,
            alpha=alpha,
            memory_endowment=memory_endowment,
            seed=run_id,
        )

        initial_num_honest = N - results["initial_num_corrupted"]
        spread_amount = results["final_num_corrupted"] - results["initial_num_corrupted"]
        spread_rate_change = (
            results["final_corruption_rate"]
            - results["initial_corruption_rate_actual"]
        )

        spread_proportion_honest = (
            spread_amount / initial_num_honest
            if initial_num_honest > 0
            else 0.0
        )

        has_initial_corruption = results["initial_num_corrupted"] > 0
        no_spread_given_initial = spread_amount == 0 if has_initial_corruption else np.nan

        run_record = {
            "condition": "no_collaboration",
            "run_id": run_id,
            "seed": run_id,

            "N": N,
            "T": T,
            "initial_corruption_rate_parameter": INITIAL_CORRUPTION_RATE,
            "task_probability": task_probability,
            "beta": BETA,
            "gamma": gamma,
            "alpha": alpha,
            "memory_endowment": memory_endowment,
            "collaboration_cost": COLLABORATION_COST,

            "initial_num_corrupted": results["initial_num_corrupted"],
            "initial_num_honest": initial_num_honest,
            "initial_corruption_rate_actual": results[
                "initial_corruption_rate_actual"
            ],

            "final_num_corrupted": results["final_num_corrupted"],
            "final_num_honest": results["final_num_honest"],
            "final_corruption_rate": results["final_corruption_rate"],

            "spread_amount": spread_amount,
            "spread_rate_change": spread_rate_change,
            "spread_proportion_honest": spread_proportion_honest,

            "has_initial_corruption": has_initial_corruption,
            "no_spread_given_initial": no_spread_given_initial,

            "mean_collaboration_rate": results["mean_collaboration_rate"],
            "mean_raw_recommendation_accuracy": results[
                "mean_raw_recommendation_accuracy"
            ],
            "mean_submitted_vote_accuracy": results[
                "mean_submitted_vote_accuracy"
            ],
            "mean_collective_decision_accuracy": results[
                "mean_collective_decision_accuracy"
            ],
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
                    "condition": "no_collaboration",
                    "run_id": run_id,
                    "seed": run_id,
                    "period": t + 1,

                    "N": N,
                    "T": T,
                    "initial_corruption_rate_parameter": INITIAL_CORRUPTION_RATE,
                    "beta": BETA,
                    "collaboration_cost": COLLABORATION_COST,

                    "num_corrupted": results["num_corrupted"][t],
                    "corruption_rate": results["corruption_rate"][t],
                    "collaboration_rate": 0.0,

                    "raw_recommendation_accuracy": results[
                        "raw_recommendation_accuracy"
                    ][t],
                    "submitted_vote_accuracy": results[
                        "submitted_vote_accuracy"
                    ][t],
                    "collective_decision_accuracy": results[
                        "collective_decision_accuracy"
                    ][t],
                    "clean_pool_accuracy": results["clean_pool_accuracy"][t],

                    "reversal_rate": results["reversal_rate"][t],
                    "num_reversed_pools": results["num_reversed_pools"][t],

                    "num_hh_pairs": 0,
                    "num_hc_pairs": 0,
                    "num_cc_pairs": 0,
                    "num_conversions": 0,
                }

                trajectory_records.append(trajectory_record)

        completed = run_id + 1

        if completed % 500 == 0 or completed == NUM_SIMULATIONS:
            print(f"Completed {completed}/{NUM_SIMULATIONS} simulations.")

    results_df = pd.DataFrame(run_records)
    results_df.to_csv(RESULTS_PATH, index=False)

    trajectory_df = pd.DataFrame(trajectory_records)
    trajectory_df.to_csv(TRAJECTORIES_PATH, index=False)

    summary = pd.DataFrame({
        "condition": ["no_collaboration"],
        "N": [N],
        "initial_corruption_rate": [INITIAL_CORRUPTION_RATE],
        "num_simulations": [NUM_SIMULATIONS],

        "mean_final_corruption_rate": [
            results_df["final_corruption_rate"].mean()
        ],
        "mean_spread_amount": [
            results_df["spread_amount"].mean()
        ],
        "mean_spread_proportion_honest": [
            results_df["spread_proportion_honest"].mean()
        ],

        "mean_collaboration_rate": [
            results_df["mean_collaboration_rate"].mean()
        ],
        "mean_raw_recommendation_accuracy": [
            results_df["mean_raw_recommendation_accuracy"].mean()
        ],
        "mean_submitted_vote_accuracy": [
            results_df["mean_submitted_vote_accuracy"].mean()
        ],
        "mean_collective_decision_accuracy": [
            results_df["mean_collective_decision_accuracy"].mean()
        ],
        "std_collective_decision_accuracy": [
            results_df["mean_collective_decision_accuracy"].std(ddof=1)
        ],
        "mean_clean_pool_accuracy": [
            results_df["mean_clean_pool_accuracy"].mean()
        ],
        "mean_reversal_rate": [
            results_df["mean_reversal_rate"].mean()
        ],
        "mean_total_conversions": [
            results_df["total_conversions"].mean()
        ],
    })

    summary.to_csv(SUMMARY_PATH, index=False)

    print("\nNo-collaboration experiment completed.")
    print(f"Results saved to {OUTPUT_DIR}")
    print("\nSummary:")
    print(summary)

    return results_df


if __name__ == "__main__":
    run_experiment()