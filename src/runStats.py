import numpy as np
import pandas as pd
import os
import re

def runStats(folder_path, game, n_actions):
    metrics_base = ['play', 'reward', 'regret', 'exploration']
    collected = {m: [] for m in metrics_base}
    run_ids = []
    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith(".csv"):
            match = re.search(r"run(\d+)", fname)
            run_ids.append(int(match.group(1)))

            df = pd.read_csv(os.path.join(folder_path, fname))
            df_game = df[df['title'] == game]
            df_game = df_game.sort_values(by="time_step").reset_index(drop=True)
            if df_game.empty:
                continue

            agent_cols_by_metric = {
                metric: [col for col in df.columns if col.startswith(f"{metric}_agent_")]
                for metric in metrics_base
            }

            for metric, cols in agent_cols_by_metric.items():
                values = df_game[cols].to_numpy().T  # shape: (n_agents, n_time_steps)
                collected[metric].append(values)
    if not any(collected.values()):
        raise ValueError(f"No data found for game: {game}")

    stacked = {
        metric: np.stack(collected[metric])
        for metric in metrics_base
    }

    plays_arr = stacked["play"]
    rewards_arr = stacked["reward"]
    regrets_arr = stacked["regret"]
    explorations_arr = stacked["exploration"]
    cum_regrets_arr = regrets_arr.cumsum(axis=2)

    n_ins, n_agents, n_time = plays_arr.shape

    mean_r = rewards_arr.mean(axis=0)
    std_r = rewards_arr.std(axis=0)
    mean_reg = cum_regrets_arr.mean(axis=0)
    std_reg = cum_regrets_arr.std(axis=0)
    mean_exploration = explorations_arr.mean(axis=0)

    stats = {
        'experiment': game,
        'shape': [n_actions, n_agents],
        'metrics': {}
    }

    for name, arr in [
        ('mean_reward', mean_r),
        ('std_reward', std_r),
        ('mean_cum_regret', mean_reg),
        ('std_cum_regret', std_reg),
        ('mean_exploration', mean_exploration),
    ]:
        stats['metrics'][name] = {
            f'agent_{i}': arr[i].tolist()
            for i in range(n_agents)
        }

    # Calculate joint action frequencies
    base = np.array([n_actions ** p for p in reversed(range(n_agents))])
    actions = np.moveaxis(plays_arr, 1, -1)  # (n_ins, n_time, n_agents)
    paire_action = np.tensordot(actions, base, axes=([2], [0])) + 1
    paire_action = paire_action.astype(int)
    ids = np.arange(1, n_actions ** n_agents + 1)
    vecteur_de_props = np.zeros((paire_action.shape[1], ids.size), dtype=float)

    for j in range(n_ins):
        for r in range(n_time):
            id_val = paire_action[j, r]
            vecteur_de_props[r, id_val - 1] += 1

    vecteur_de_props /= n_ins
    stats["metrics"]["vecteur_de_props"] = vecteur_de_props

    return stats, f"run{max(run_ids)}"