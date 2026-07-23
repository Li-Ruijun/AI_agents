# AI agent contagion simulation

The current version moves from a fixed collaboration threshold to an endogenous belief cutoff.

##Files

-'config.pg': stores all parameters.
-'network.py': creates the ring work.
-'proposal_rules.py': controls how agents propose collaboration partners.
-'cutoff.py':calculates the endogenous belief cutoff.
-'voting.py':generates simplified vote correctness and performs Bayesian belief updating for observed non-partner neighbours.
-'dynamics.py': define initialization, contagion, and belief update.
-'simulation.py': contains 'one_simulation' and 'many_simulation'.
-'main.py': runs the simulation and saves CSV outputs
-'notebooks/plot_results.ipynb': reads CSV results and generates figures.