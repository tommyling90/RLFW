# Reinforcement Learning Framework for Experiments with Agent Learning in Matrix Games

## Overview

This project is a modular training framework for reinforcement learning and implements several RL algorithms, matrix games, agents, and the environment where the games are carried out.
It demonstrates simulation-based methods and scalable code structure.

This framework is designed with flexibility in mind, enabling the agility often needed in scientific research workflows — such as running experiments and generating figures or statistics under varying conditions, including different noise levels, algorithmic configurations, and other custom modifications.
Some important features include game configuration by user, extending the games to multiple players with tensor, checkpointing, figure configuration by user, and others.

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
2. # TODO - command line
3. Go to `main.py`. If user wishes to generate the results and the figures in one go, just run this file.
3. Or else, comment out the code from where the program loads the `graph_config.yaml` file.
4. User can interrupt the experiments at any moment. The results are saved in `pkl` files. When the user resumes, the experiment will pick up where it was left off.
5. If user wishes to only generate figures, comment out the code above where the program loads the `graph_config.yaml` file.
6. In `graph_config.yaml` user can choose a different folder than the folder in `config.yaml` to generate the figures from.

## Documentation

### 💡General Idea
In this framework, the generation of results (`runResults.py`), statistics (`runStats.py`), and figures (`runFigures.py`) is separated and modularized.
Notice that when executing the program, the user can interrupt the experiment at any moment, and the results to date would be saved in `.pkl` files and converted to `.csv`.
When he resumes, the experiments will pick up where he left off. This is called *checkpointing*.

The `pkl` files are saved to the directory `project_root/{folder}/pkl`.
Notice the `folder` in this path is provided by user in `config.yaml` - see Configuration section below.
These `pkl` files are used in order to generate the csv file - see Saving CSV below.

Along with the `pkl` files, `config.yaml` and csv files will also be saved to the folder.
`config.yaml` is saved in order to provide the user an idea of what configurations he was running in that specific experiment.

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

Key parameters are loaded from a config.yaml file that the user needs to provide.
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
  - A list of flattened metric dicts called `delta`. Note that in `delta`, only the metrics of that certain checkpoint are recorded, in order to speed up the execution.
  - `rng_state` - random number generator state.

Each checkpoint is saved as a separate `.pkl` file named: `cp_run{r}.pkl`

#### Saving CSV

Metrics of each checkpoint are stored in each `.pkl` file and then converted into csv at each checkpoint using the function `aggregate_metrics_from_single_pkl`.

This is especially useful if the user is running long experiments (for example, 500 runs and 10,000 iterations).
Instead of running for several hours before being able to print the graphs, the user could now interrupt after, say, 50 runs, and generate the graphs with the first 50 runs to get a first glimpse at the trends, and then continue the experiment after the checkpoint.
Notice that it is possible because as mentioned previously, each run contains the result of all games for all time steps. Therefore, the user can compare the results of different games.

Furthermore, after interrupting and resuming, since the csv data has been recorded for the runs before the interruption, the user can now delete the .pkl files to lessen the load for the program.

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
├── config.yaml          # Original config.yaml that the user should provide
├── graph_config.yaml    # Original graph_config.yaml that the user should provide for graph generation
├── README.md            # This file
```