from pathlib import Path

from simulation import many_simulations

from config import (
    N,
    T,
    gamma,
    alpha,
    memory_endowment,
    discount_factor
)


# Parameter values

INITIAL_CORRUPTION_RATES = [
    0.20
]

BETA_VALUES = [
    0.50
]

COLLABORATION_COST_VALUES = [
    0.02
]


# Experiment settings
NUM_SIMULATIONS_PER_SETTING = 10000

# Save several complete trajectories for each parameter setting.
SAVE_TRAJECTORIES = True
NUM_TRAJECTORIES_PER_SETTING = 7

# If True, parameter settings that already have both CSV files
SKIP_EXISTING_RESULTS = True

# Main folder for the parameter experiment.
PARAMETER_SWEEP_DIR = Path("results/distribution_parameter_beta_sweep")


def format_parameter_value(value):

    return f"{value:.2f}".replace(".", "p")


def run_parameter_sweep():

    PARAMETER_SWEEP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    total_settings = (
        len(INITIAL_CORRUPTION_RATES)
        * len(BETA_VALUES)
        * len(COLLABORATION_COST_VALUES))

    completed_settings = 0
    skipped_settings = 0

    setting_number = 0

    for initial_corruption_rate in INITIAL_CORRUPTION_RATES:

        for beta in BETA_VALUES:

            for collaboration_cost in COLLABORATION_COST_VALUES:

                setting_number += 1

                # Convert parameter values into folder labels.
                initial_label = format_parameter_value(initial_corruption_rate)

                beta_label = format_parameter_value(beta)

                cost_label = format_parameter_value(collaboration_cost)

                setting_name = (f"initial_{initial_label}"f"_beta_{beta_label}"f"_cost_{cost_label}")

                # One separate folder for each parameter combination.
                setting_dir = (PARAMETER_SWEEP_DIR/ setting_name)

                setting_dir.mkdir(parents=True,exist_ok=True)

                simulation_results_csv = (setting_dir/ "simulation_results.csv")

                trajectory_results_csv = (setting_dir/ "trajectory_results.csv")

                print()
                print("=" * 70)
                print(f"Parameter setting "f"{setting_number}/{total_settings}")
                print("Initial corruption rate: "f"{initial_corruption_rate:.2f}")
                print(f"Beta: {beta:.2f}")
                print(f"Collaboration cost: {collaboration_cost:.2f}")
                print(f"Output folder: {setting_dir}")
                print("=" * 70)

                # Skip completed parameter combinations when rerunning.
                if (SKIP_EXISTING_RESULTS and simulation_results_csv.exists() and trajectory_results_csv.exists()):
                    print("Both CSV files already exist. ""This setting is skipped.")

                    skipped_settings += 1
                    continue

                # Run the simulations for this parameter combination.
                summary = many_simulations(
                    num_simulations=NUM_SIMULATIONS_PER_SETTING,
                    N=N,
                    T=T,
                    initial_corruption_rate=initial_corruption_rate,
                    beta=beta,
                    gamma=gamma,
                    alpha=alpha,
                    memory_endowment=memory_endowment,
                    collaboration_cost=collaboration_cost,
                    discount_factor=discount_factor,
                    save_csv=True,
                    csv_filename=simulation_results_csv,
                    save_trajectories=SAVE_TRAJECTORIES,
                    num_trajectories=NUM_TRAJECTORIES_PER_SETTING,
                    trajectory_csv_filename=trajectory_results_csv
                )

                completed_settings += 1

                print()
                print("Setting completed.")
                print("Mean final corruption rate: "f"{summary['mean_final_corruption_rate']:.4f}")
                print("Standard deviation: "f"{summary['std_final_corruption_rate']:.4f}")
                print("Probability of no spread: "f"{summary['prob_no_spread']:.4f}")
                print("Mean final collaboration rate: "f"{summary['mean_final_collaboration_rate']:.4f}")
                print("Variance of final collaboration rate: "f"{summary['variance_final_collaboration_rate']:.6f}")
                print("Mean full-horizon collaboration rate: "f"{summary['mean_full_horizon_collaboration_rate']:.4f}")
                print("Variance of full-horizon collaboration rate: "f"{summary['variance_full_horizon_collaboration_rate']:.6f}")

    print()
    print("=" * 70)
    print("Parameter sweep finished.")
    print(f"Newly completed settings: "f"{completed_settings}")
    print(f"Skipped existing settings: "f"{skipped_settings}")
    print(f"Total parameter settings: "f"{total_settings}")
    print("=" * 70)


if __name__ == "__main__":
    run_parameter_sweep()