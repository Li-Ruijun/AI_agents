from pathlib import Path

# Basic simulation settings.
N = 400# Number of agents.
T = 100   # Time steps.
num_simulations = 600


# Model parameters.
initial_corruption_rate = 0.05
task_probability = 0.5

beta = 0.5
gamma = 0.4
alpha = 0.9
memory_endowment = 1.0
collaboration_cost = 0.02
discount_factor = 0.9


# Random seed
seed=42

# Output settings
save_csv = True
save_trajectories = True
num_trajectories = 7

experiment_dir = Path("result_obsereable_pool")
results_dir = experiment_dir/"results"
figures_dir = experiment_dir/"figures"


simulation_results_csv = results_dir/"simulation_results.csv"
trajectory_results_csv = results_dir/"trajectory_results.csv"


