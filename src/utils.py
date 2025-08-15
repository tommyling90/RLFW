import numpy as np
import pickle
import os
import re
import pandas as pd
from collections import defaultdict

def flatten_metrics(title, player, instance, n_actions, start, metrics_dict):
    row = {
        "title": title,
        "player": player,
        "instance": instance,
        "n_actions": n_actions
    }

    for key, val in metrics_dict.items():
        for t, value in enumerate(val):
            row[f"{key}{t + start}"] = value
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
    latest_run_idx = -1

    pattern = re.compile(r"cp_run(\d+)\.pkl")

    if not os.path.exists(pkl_folder): return None
    for fname in os.listdir(pkl_folder):
        match = pattern.match(fname)
        if match:
            run_idx = int(match.group(1))
            if run_idx > latest_run_idx:
                latest_run_idx = run_idx
                latest_cp = fname

    if latest_cp:
        return os.path.join(pkl_folder, latest_cp)
    else:
        print("❌ No checkpoint found.")
        return None

def save_pickle_atomic(path, obj):
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp_path, path)

def save_pickle(folder, r, all_games_metrics_for_run, env_list, suffix):
    env_list_ser = [env.serialize() for env in env_list]
    cp = {
        'run_idx': r+1,
        'metrics': all_games_metrics_for_run,
        'rng_state': np.random.get_state(),
        'env_state': env_list_ser
    }
    pkl_file = f"{folder}/pkl/cp_run{r}{suffix}.pkl"
    os.makedirs(os.path.dirname(pkl_file), exist_ok=True)
    save_pickle_atomic(pkl_file, cp)
    print(f"📝 Saved checkpoint: run={r}")

def aggregate_metrics_from_single_pkl(file_path):
    with open(file_path, "rb") as f:
        cp = pickle.load(f)
    rows_by_time = defaultdict(lambda: defaultdict(dict))

    pattern = re.compile(r"([a-zA-Z_]+)(\d+)$")

    for entry in cp["metrics"]:
        title = entry["title"]
        player = entry["player"]
        instance = entry["instance"]
        n_actions = entry["n_actions"]

        for key, value in entry.items():
            match = pattern.match(key)
            if match:
                metric_prefix, t_str = match.groups()
                t = int(t_str)
                base_metric = metric_prefix.replace("_time", "").rstrip("_")  # clean up "reward_time" → "reward"
                colname = f"{base_metric}_{player}"
                rows_by_time[(title, instance, t)][colname] = value

    tall_rows = []
    for (title, instance, t), metric_dict in sorted(rows_by_time.items()):
        row = {"title": title, "n_actions": n_actions, "time_step": t,}
        row.update(metric_dict)
        tall_rows.append(row)

    df = pd.DataFrame(tall_rows)
    df = df.sort_values(["title"])

    fname = os.path.basename(file_path)
    run_id = fname.removesuffix(".pkl").replace("cp_", "")
    output_dir = os.path.join(os.path.dirname(file_path), "..", "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{run_id}.csv")

    df.to_csv(output_path, index=False)
    print(f"📄 Saved clean tall-wide CSV: {output_path}")

def recover_last_csv(folder, last_run_id):
    pkl_path = os.path.join(folder, f'cp_run{last_run_id}.pkl')
    aggregate_metrics_from_single_pkl(pkl_path)

def get_pickle_len(pkl_path):
    with open(pkl_path, "rb") as f:
        checkpoint = pickle.load(f)

    sample_metrics = checkpoint['metrics'][0]
    iter_reward = [k for k in sample_metrics.keys() if re.match(r"reward_time\d+$", k)]
    iter_regret = [k for k in sample_metrics.keys() if re.match(r"regret_time\d+$", k)]
    iter_play = [k for k in sample_metrics.keys() if re.match(r"play_time\d+$", k)]
    iter_exp = [k for k in sample_metrics.keys() if re.match(r"exploration_time\d+$", k)]
    return len(iter_reward), len(iter_regret), len(iter_play), len(iter_exp)

def is_csv_complete(csv_file, num_games, expected_iterations):
    if not csv_file.exists():
        return False
    actual_lines = get_csv_line_count(csv_file)
    return actual_lines == num_games * expected_iterations

def set_rng_for_run(run_id, seed_base, pkl_path):
    run_pkl = f"{pkl_path}/pkl/cp_run{run_id}.pkl"
    if os.path.exists(run_pkl):
        with open(run_pkl, "rb") as f:
            cp = pickle.load(f)
        np.random.set_state(cp['rng_state'])
    else:
        np.random.seed(seed_base + run_id)

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

def get_csv_line_count(csv_file):
    with open(csv_file, "r") as f:
        return sum(1 for _ in f) - 1