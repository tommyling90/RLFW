import numpy as np
import pickle
import os
import re
import pandas as pd
from collections import defaultdict

def flatten_metrics(title, player, instance, n_actions,metrics_dict):
    row = {
        "title": title,
        "player": player,
        "instance": instance,
        "n_actions": n_actions
    }

    for key, val in metrics_dict.items():
        for t, value in enumerate(val):
            row[f"{key}{t}"] = value
    return row

def sort_metric_columns(df):
    meta_cols = ["title", "player", "instance", "n_actions"]

    metric_cols = [col for col in df.columns if col not in meta_cols]

    def sort_key(col):
        match = re.match(r"([a-zA-Z_]+)(\d+)", col)
        if match:
            return (match.group(1), int(match.group(2)))
        else:
            return (col, -1)  # fallback

    metric_cols_sorted = sorted(metric_cols, key=sort_key)

    return df[meta_cols + metric_cols_sorted]

def find_latest_checkpoint(pkl_folder):
    latest_cp = None
    latest_key = (-1, -1, -1)  # (game, run, iter)

    pattern = re.compile(r"cp_game(\d+)_run(\d+)\.pkl")

    if not os.path.exists(pkl_folder): return None
    for fname in os.listdir(pkl_folder):
        match = pattern.match(fname)
        if match:
            g, r = map(int, match.groups())
            if (g, r) > latest_key:
                latest_key = (g, r)
                latest_cp = fname

    if latest_cp:
        print(f"✅ Latest checkpoint: {latest_cp}")
        return os.path.join(pkl_folder, latest_cp)
    else:
        print("❌ No checkpoint found.")
        return None

def save_pickle_atomic(path, obj):
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp_path, path)

def save_pickle(ctx, g, r, plays, exploration_list, regrets, rewards, title, n_actions):
    delta = []
    for agent_id in range(plays.shape[0]):
        metrics_dict = {
            "play_time": plays[agent_id].tolist(),
            "reward_time": rewards[agent_id].tolist(),
            "regret_time": regrets[agent_id].tolist(),
            "exploration_time": exploration_list[agent_id].tolist(),
        }

        delta.append(flatten_metrics(
            title=title,
            player=f"agent_{agent_id}",
            instance=f"instance_{r}",
            n_actions=n_actions,
            metrics_dict=metrics_dict
        ))

    cp = {
        "game_idx": g,
        'run_idx': r+1,
        'metrics': delta,
        'rng_state': np.random.get_state(),
    }
    pkl_file = f"{ctx.cp_file}/pkl/cp_game{g+1}_run{r}.pkl"
    os.makedirs(os.path.dirname(pkl_file), exist_ok=True)
    save_pickle_atomic(pkl_file, cp)
    print(f"📝 Saved checkpoint: game={g+1}, run={r}")

def aggregate_metrics_from_pkl(path):
    merged_rows = defaultdict(dict)

    for fname in sorted(os.listdir(path)):
        if fname.endswith(".pkl") and fname.startswith("cp_"):
            with open(os.path.join(path, fname), "rb") as f:
                cp = pickle.load(f)

                for entry in cp["metrics"]:
                    key = (entry["title"], entry["player"], entry["instance"])

                    if not merged_rows[key]:
                        merged_rows[key]["title"] = entry["title"]
                        merged_rows[key]["player"] = entry["player"]
                        merged_rows[key]["instance"] = entry["instance"]
                        merged_rows[key]["n_actions"] = entry["n_actions"]

                    for k, v in entry.items():
                        if k.startswith(("play_time", "reward_time", "regret_time", "exploration_time")):
                            merged_rows[key].setdefault(k, v)

    all_rows = list(merged_rows.values())
    df = pd.DataFrame(all_rows)
    df = sort_metric_columns(df)

    parent_folder = os.path.dirname(path.rstrip("/"))
    df.to_csv(os.path.join(parent_folder, "output.csv"), index=False)
    print(f"✅ Saved aggregated CSV.")

def generate_n_player_PD(n, reward_matrix):
    # 2 pcq trahir vs trahir pas
    shape = (2,) * n
    payoffs = [np.zeros(shape) for _ in range(n)]

    for actions in np.ndindex(shape):
        betray_count = sum(actions)

        for i in range(n):
            choice_i = actions[i]
            others = list(actions[:i] + actions[i + 1:])
            others_betray = sum(others)

            if betray_count == 0:
                val = reward_matrix[0,0]  # all cooperate
            elif betray_count == n:
                val = reward_matrix[1,1]  # all betray
            elif choice_i == 1 and others_betray == 0:
                val = reward_matrix[1,0]  # lone betrayer
            elif choice_i == 0 and others_betray == n - 1:
                val = reward_matrix[0,1]  # lone cooperator
            elif choice_i == 1:
                val = (1+reward_matrix[0,0])/2  # partial betrayal: betrayer reward
            else:
                val = (reward_matrix[1,1])/2  # partial betrayal: cooperator punished

            payoffs[i][actions] = val

    return payoffs

def generate_n_player_diag(n_players, k_actions, reward_matrix):
    shape = (k_actions,) * n_players
    reward_tensor = np.zeros(shape)

    for action_combo in np.ndindex(shape):
        if all(a == action_combo[0] for a in action_combo):
            reward_tensor[action_combo] = reward_matrix[action_combo[0], action_combo[0]]
        else:
            reward_tensor[action_combo] = 0.0

    return [reward_tensor] * n_players

def generate_n_player(n_players, k_actions, reward_matrix):
    shape = (k_actions,) * n_players
    reward_tensor = np.zeros(shape)

    for action_combo in np.ndindex(shape):
        if all(a == action_combo[0] for a in action_combo):
            reward_tensor[action_combo] = reward_matrix[action_combo[0], action_combo[0]]
        elif any(a == 0 for a in action_combo) and any(a == 2 for a in action_combo):
            reward_tensor[action_combo] = 0.0
        else:
            reward_tensor[action_combo] = 0.2

    return [reward_tensor] * n_players

def is_diagonal(matrix):
    return np.allclose(matrix, np.diag(np.diagonal(matrix)))

def normalizeMatrix(matrix, etendue):
    matrix_norm = (matrix-np.min(matrix))/np.ptp(matrix)
    matrix_norm_noise = matrix_norm*(1-etendue)+etendue/2
    return matrix_norm_noise

def parse_string(s):
    parts = s.split('_')
    algos = parts[0]
    noise = parts[2]
    game = '_'.join(parts[3:])
    return algos, noise, game