# Reinforcement Learning Framework for Experiments with Agent Learning in Matrix Games
Created by TCH Lin; github @tommyling90; tommyling79@gmail.com

## Overview

This project is a modular framework for reinforcement learning that allows for studies of agent learning behaviour in various games using different algorithms. It implements several RL algorithms, matrix games, agents, and the environment where the games are carried out.
It demonstrates simulation-based methods and scalable code structure.

This framework is designed with flexibility in mind, enabling the agility often needed in scientific research workflows — such as running experiments and generating figures or statistics under varying conditions, including different noise levels, algorithmic configurations, and other custom modifications.
Some important features include game and figure configuration by user, extending the games to multiple players with tensor, checkpointing, extending iterations/runs/games, and others.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone <repo-url>
cd project-directory

python3.10 -m venv PATH_TO_ENV/ENV_NAME
source PATH_TO_ENV/ENV_NAME/bin/activate
pip install -r ./requirements.txt
```
## Use example

1. Configure the files `config.yaml` and `graph_config.yaml` (see description below and the files for reference)
2. Use command line tools to run the experiments and print the results. See available commands and use examples below.
3. User can interrupt the experiments at any moment. The results are saved in `pkl` files. When the user resumes, the experiment will pick up where it was left off.
4. Notice that in `graph_config.yaml` user can specify a different folder than the folder in `config.yaml` to generate the figures from (hence, `generate_figures` is completely separate from `run_results`).
5. Users can freely extend runs and horizon as per needed (but cannot extend them before the experiment is complete). User can also add new games into root `config.yaml` and extend the experiment running these new games.

### 🎮 CLI options
1. `run_results` (args: --suffix_extend_games, required when running extended games, whether new runs or extending horizon/runs for them; generate `.pkl` files. Can interrupt at any moment)
2. `generate_figures` (arg: --suffix, suffix to figure name; generate figures. Can run at any moment as long as `.csv` files are present. Generates figures only for experiments with completed horizon.)
3. `prune_pkls` (arg: --path, need to give the relative path to the pkl folder containing `.pkl`s to delete; ⚠️deletes the `pkl` folder and all files contained. Use ONLY when the experiment is finalized.)
4. `add_runs` (arg: --n_runs, number of runs to add; adds more runs to the experiment in the `config.yaml` file)
5. `add_horizon` (arg: --n_horizon, number of iterations to add; adds more iterations/horizon to the experiment in the `config.yaml` file)

### 🔄 Example workflow using CLI
Basic use case
First, configure the `project_root/config.yaml` file (see configuration guide below)
`python main.py run_results` - run the experiment for the first time
`ctl-c` - interrupt the experiment at, say, run 50
`python main.py gemerate_figures` - generate figures up until run 50
`python main.py run_results` - resume the experiment until finish
`python main.py gemerate_figures` - generate figures for all runs (since it's now finished)

Extending runs and horizon
`python main.py add_runs --n_runs 20` - add 20 additional runs
`python main.py add_horizon --n_horizon 100` - add 100 additional time steps into each run
`python main.py run_results` - re-run the experiment with new params
`ctl-C` - interrupt the experiment at any moment
`python main.py gemerate_figures` - ⚠️this will get you error since the horizon for the runs before interruption is different than that after interruption
`python main.py run_results` - resume the experiment until finish. Now we can generate figures as the horizon is complete for all runs.

Extending games. First, add new games into the root `config.yaml` mentioned above
`python main.py run_results --suffix_extend_games _ext1` - run the experiment with extended games. Suffix for csv's and pkl's required.
`python main.py add_runs --n_runs 20` - add another 20 additional runs
`python main.py run_results` - run the additional 20 runs for non-extended games
`python main.py run_results --suffix_extend_games _ext1` - run the additional 20 runs for extended games with the suffix defined earlier
`python main.py gemerate_figures` - generate figures

## Documentation

### 💡General Idea
In this framework, the generation of results (`runResults.py`), statistics (`runStats.py`), and figures (`runFigures.py`) is separated and modularized.
Notice that when executing the program, the user can interrupt the experiment at any moment, and the results to date would be saved in `.pkl` files and converted to `.csv` immediately after.
When he resumes, the experiments will pick up where he left off. This is *checkpointing*.

The `pkl` files are saved to the directory `project_root/{folder}/pkl`.
Notice the `folder` in this path is provided by user in `config.yaml` - see Configuration section below.
These `pkl` files are used in order to generate the csv file - see Saving CSV below.

Along with the `pkl` files, `config.yaml` and csv files will also be saved to the folder.
`config.yaml` is saved as manifest in order to provide the user an idea of what configurations he was running in that specific experiment.

### ✅ Core Concepts
#### Agents and Metrics
Each experiment involves multiple agents interacting over a number of iterations.
During each iteration, the framework records the following metrics:

- play (action chosen by the agent)
- reward (reward for the agent at one given time step)
- regret_time (regret of the agent at one given time step)
- exploration_time (whether the agent explored at one given time step. This is boolean represented by 0 and 1)

These metrics are stored per agent and per iteration.

### ⚙️ Configuration

Key parameters are loaded from the root config.yaml file that the user needs to provide.
Refer to the existing `config.yaml` file to see what and how the parameters should be provided.
It is STRONGLY recommended to follow the same structure.

The game names available are:
- PG_WP
- PG
- PD
- SG
- CG_no
The user should also provide their own defined matrices for these games.

The `noise` parameter should always follow this pattern: `[0.0, {noise_level_tested}]`

The `matrix` parameter takes 2 forms:
1. Symmetric payoff - enter a 2-D matrix: 
`matrix: - [...,...] - [...,...]`
2. Asymmetric payoff and customized payoff matrix - enter a 3-D matrix:
`matrix: - [[...,...], [...,...]] - [[...,...], [...,...]]`

Algorithms available are:
- UCB
- KLUCB
- TS
- SoftMax

In `defaults`, the `player` parameter specifies the number of players in the game. Note that this number MUST match the length of the `algos` param in game.

In `save_folder` parameter, the path entered should always be `Figures/YOUR_FOLDER_NAME` without any `../` preceding.

### 📈 Configuration of figures

The user should also configure the file `graph_config.yaml` to specify what games to generate the figures for and from what experiment folder.
It is the user's responsibility to make sure that the games for the figures actually exist in the respective folder.
Refer to the file to see what parameters are possible.
Notice that
1. the `name` must be exactly the same as listed above
2. currently 2 possible choices for the `cumul_y` parameter - `regret` (cumulative regret graph) and `prop` (proportion of joint actions graph). These are the 2 graphs currently available.
3. all values must be in string
4. in `algos` parameter, the order matters! It must be the same order as in `algos` parameter in `config.yaml`
5. for the `regret` graph, user can choose to either have different levels of noise for one certain algorithm combo in the graph, or have different algo combos on the same noise level in the graph. However, to keep the graph easy to read, it is NOT possible to have different algo combos across different noise levels in one graph. Likewise, for `prop` graph, to keep the graph clean, it is only possible to have one algo combo on one noise level.
6. In `save_folder` parameter, the path entered should always be `Figures/YOUR_FOLDER_NAME` without any `../` preceding.
7. specify `n_actions` for the experiments you're generating the figures for.

### 📦 Checkpointing
#### Save Strategy
- Checkpoints are saved every run (regardless of the number of iterations each run). In each run, all the games in the experiment are run once for all time steps.
- At each save point, the following are recorded:
  - `run_idx`
  - A list of flattened metric dicts `delta`. Note that in `delta`, only the metrics of that certain checkpoint are recorded, in order to speed up the execution.
  - `rng_state` - random number generator state.
  - `env_state` - serialized object `env` that contains the data for each agent. This is used when extending horizon.

Each checkpoint is saved as a separate `.pkl` file named: `cp_run{r}{suffix}.pkl`

#### Saving CSV

Metrics of each checkpoint are stored in each `.pkl` file and then converted into csv at each checkpoint using the function `aggregate_metrics_from_single_pkl`.

This is especially useful if the user is running long experiments (for example, 500 runs and 10,000 iterations).
Instead of running for several hours before being able to print the graphs, the user could now interrupt after, say, 50 runs, and generate the graphs with the first 50 runs to get a first glimpse at the trends, and then continue the experiment after the checkpoint.
Notice that it is possible because as mentioned previously, each run contains the result of all games for all time steps. Therefore, the user can compare the results of different games.

Furthermore, when the experiment is finalized (that is, no more run/horizon extensions), since the csv data has been recorded for the runs before the interruption, the user can now delete the .pkl files to lessen the load for the program.

## Directory Structure

```bash
├── src/                 # Source code
├── Figures/         
    ├── Test/            # Folder specified in yaml to accommodate the relevant data for that experiment
        ├── pkl/         # Folder that contains all pkl files
        ├── prop/        # Where joint action proportion graphs are housed
            ├── run100/  # Generated joint action proportion graphs for first 100 runs come here (ditto for run200/run300...etc.)        
        ├── regret/      # Where regret graphs are housed
            ├── run100/  # Generated regret graphs for first 100 runs come here (ditto for run200/run300...etc.)        
        ├── config.yaml  # A copy of config.yaml as reference to the configurations experimented with
        ├── output/      # Folder for csv results of the experiment
            ├── run1.csv # Csv results of run1 (ditto for run2/run3...etc.)
├── main.py              # Entry file
├── config.yaml          # Original config.yaml that the user should provide
├── graph_config.yaml    # Original graph_config.yaml that the user should provide for graph generation
├── README.md            # This file
```