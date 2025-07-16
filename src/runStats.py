import numpy as np
import pandas as pd

def runStats(file_path, game):
    df = pd.read_csv(file_path)
    n_actions = np.unique(df['n_actions'].to_numpy())[0]
    df_game = df[df['title'] == game]
    grouped = df_game.groupby(['title', 'instance'])

    metrics = ['play_time', 'regret_time', 'reward_time', 'exploration_time']
    collected = {key: [] for key in metrics}

    for _, group in grouped:
        for key in metrics:
            data = group.filter(like=key, axis=1)
            collected[key].append(data)
    stacked = {key: np.stack(collected[key]) for key in metrics}
    plays_arr = stacked["play_time"]
    rewards_arr = stacked["reward_time"]
    regrets_arr = stacked["regret_time"]
    explorations_arr = stacked["exploration_time"]
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

    '''Calculer la proportion ci-bas'''
    base = np.array([n_actions**p for p in reversed(range(n_agents))])
    actions = np.moveaxis(plays_arr, 1, -1)
    paire_action = np.tensordot(actions, base, axes=([2], [0])) + 1
    paire_action = paire_action.astype(int)
    ids = np.array([i for i in range(n_actions**n_agents)]) + 1
    vecteur_de_props = np.zeros((paire_action.shape[1], ids.size), dtype=float)

    for j in range(n_ins):
        for r in range(n_time):
            id_val = paire_action[j, r]
            vecteur_de_props[r, id_val - 1] += 1

    vecteur_de_props /= n_ins
    stats["metrics"]["vecteur_de_props"] = vecteur_de_props

    return stats