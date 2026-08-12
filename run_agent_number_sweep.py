from pathlib import Path

from simulation import many_simulations

from config import (
    T,
    gamma,
    alpha,
    memory_endowment,
    discount_factor
)


# Agent-number experiment

# Only N changes across settings.
N_VALUES = [
    50,
    100,
    200,
    400,
    800
]

# Baseline parameters kept fixed.
INITIAL_CORRUPTION_RATE = 0.05
BETA = 0.50
COLLABORATION_COST = 0.02

# ============================================================
# Experiment settings

NUM_SIMULATIONS_PER_SETTING = 10000

# Save several complete trajectories for each N.
SAVE_TRAJECTORIES = True
NUM_TRAJECTORIES_PER_SETTING = 7

# Skip a setting when both output CSV files already exist.
SKIP_EXISTING_RESULTS = True

# Main output folder.
AGENT_NUMBER_SWEEP_DIR = Path(
    "results/agent_number_sweep"
)


def format_agent_number(number_of_agents):
    """Create a sortable folder label such as N_0050."""

    return f"N_{number_of_agents:04d}"


def run_agent_number_sweep():
    """Run the same baseline experiment for several values of N."""

    AGENT_NUMBER_SWEEP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    total_settings = len(N_VALUES)

    completed_settings = 0
    skipped_settings = 0

    for setting_number, current_N in enumerate(
        N_VALUES,
        start=1
    ):
        setting_name = format_agent_number(current_N)

        setting_dir = (
            AGENT_NUMBER_SWEEP_DIR
            / setting_name
        )

        setting_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        simulation_results_csv = (
            setting_dir
            / "simulation_results.csv"
        )

        trajectory_results_csv = (
            setting_dir
            / "trajectory_results.csv"
        )

        print()
        print("=" * 70)
        print(
            f"Agent-number setting "
            f"{setting_number}/{total_settings}"
        )
        print(f"N: {current_N}")
        print(
            "Initial corruption rate: "
            f"{INITIAL_CORRUPTION_RATE:.2f}"
        )
        print(f"Beta: {BETA:.2f}")
        print(
            "Collaboration cost: "
            f"{COLLABORATION_COST:.2f}"
        )
        print(f"T: {T}")
        print(f"Output folder: {setting_dir}")
        print("=" * 70)

        # Skip completed settings when rerunning the script.
        if (
            SKIP_EXISTING_RESULTS
            and simulation_results_csv.exists()
            and trajectory_results_csv.exists()
        ):
            print(
                "Both CSV files already exist. "
                "This setting is skipped."
            )

            skipped_settings += 1
            continue

        # Run simulations for the current value of N.
        summary = many_simulations(
            num_simulations=NUM_SIMULATIONS_PER_SETTING,
            N=current_N,
            T=T,
            initial_corruption_rate=(
                INITIAL_CORRUPTION_RATE
            ),
            beta=BETA,
            gamma=gamma,
            alpha=alpha,
            memory_endowment=memory_endowment,
            collaboration_cost=(
                COLLABORATION_COST
            ),
            discount_factor=discount_factor,
            save_csv=True,
            csv_filename=simulation_results_csv,
            save_trajectories=SAVE_TRAJECTORIES,
            num_trajectories=(
                NUM_TRAJECTORIES_PER_SETTING
            ),
            trajectory_csv_filename=(
                trajectory_results_csv
            )
        )

        completed_settings += 1

        print()
        print("Setting completed.")

        print(
            "Mean final corruption rate: "
            f"{summary['mean_final_corruption_rate']:.4f}"
        )

        print(
            "Standard deviation: "
            f"{summary['std_final_corruption_rate']:.4f}"
        )

        print(
            "Probability of no spread: "
            f"{summary['prob_no_spread']:.4f}"
        )

        print(
            "Mean final collaboration rate: "
            f"{summary['mean_final_collaboration_rate']:.4f}"
        )

        print(
            "Variance of final collaboration rate: "
            f"{summary['variance_final_collaboration_rate']:.6f}"
        )

        print(
            "Mean full-horizon collaboration rate: "
            f"{summary['mean_full_horizon_collaboration_rate']:.4f}"
        )

        print(
            "Variance of full-horizon collaboration rate: "
            f"{summary['variance_full_horizon_collaboration_rate']:.6f}"
        )

    print()
    print("=" * 70)
    print("Agent-number sweep finished.")
    print(
        f"Newly completed settings: "
        f"{completed_settings}"
    )
    print(
        f"Skipped existing settings: "
        f"{skipped_settings}"
    )
    print(
        f"Total agent-number settings: "
        f"{total_settings}"
    )
    print("=" * 70)


if __name__ == "__main__":
    run_agent_number_sweep()
