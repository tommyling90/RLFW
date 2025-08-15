import yaml
import sys
from pathlib import Path

from src.execute import Execute
from src.utils import *
from src.environment import Environnement

root = Path(__file__).resolve().parent.parent
LAST_ACTIVE_RUN = Path("last_active_run.txt")

def get_last_active_run():
    try:
        with open("last_active_run.txt", "r") as f:
            return int(f.read().strip()) - 1
    except FileNotFoundError:
        return None

def open_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    defaults = config['defaults']
    games = config['games']

    horizon = defaults['horizon']
    runs = defaults['runs']
    player = defaults['player']
    seed = defaults["seed"]
    folder = f"{root}/{defaults['save_folder']}"
    return games, horizon, runs, player, seed, folder, config

def run_results(suffix):
    config_path = root / "config.yaml"
    games, horizon, runs, player, seed, folder, config = open_config(config_path)
    last_run_csv = Path(f"../Figures/{folder}/output/run{runs-1}.csv")
    manifest = Path(f"../Figures/{folder}/config.yaml")

    extend_games = False

    if manifest.exists() and last_run_csv.exists():
        m_games, m_horizon, m_runs, m_player, m_seed, m_folder, m_config = open_config(manifest)
        m_values = tuple(m_games.values())
        filtered_v = [v for v in games.values() if v not in m_values]
        if len(filtered_v) > 0:
            if suffix == '':
                raise RuntimeError('⚠️ Must give a suffix to csv when extending games!')
            extend_games = True
            games = {f"game{i}": v for i, v in enumerate(filtered_v, 1)}

    if os.path.isdir(folder):
        choice = input("⚠️ Folder already exists.\n"
               "If you're continuing an experiment that was interrupted or running more runs/horizon/games, press Y to continue.\n"
               "Otherwise press Q to quit and rename the folder in config.yaml.\n"
               "[Y/Q]").strip().upper()
        if choice == "Y":
            print("✅ Continuing...")
        elif choice == "Q":
            print("❌ Exiting.")
            sys.exit(0)
        else:
            print("❗ Invalid input. Exiting.")
            sys.exit(1)
    else:
        os.makedirs(folder, exist_ok=True)

    if extend_games:
        next_idx = len(m_games) + 1
        for v in games.values():
            m_games[f"game{next_idx}"] = v
            next_idx += 1
        m_config['games'] = m_games
        with open(manifest, "w") as f:
            yaml.safe_dump(m_config, f, sort_keys=False, allow_unicode=True)
    else:
        with open(f"{folder}/config.yaml", 'w') as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    last_run_id = get_last_active_run()
    if last_run_id is not None:
        print(f"Regenerating CSV for run {last_run_id} before resuming...")
        recover_last_csv(f"{folder}/pkl/", last_run_id)
    else:
        print("No previous run to recover.")

    for r in range(runs):
        set_rng_for_run(r, seed_base=seed, pkl_path=folder)
        # Récupérer le csv pour le dernier run pour s'assurer des données complètes et correctes
        with open(LAST_ACTIVE_RUN, "w") as f:
            f.write(str(r))

        # logique pour soit une nouvelle expérience soit une extension d'une expérience
        pkl_file = Path(folder) / "pkl" / f"cp_run{r}{suffix}.pkl"
        csv_file = Path(folder) / "output" / f"run{r}{suffix}.csv"

        lengths = get_pickle_len(pkl_file) if pkl_file.exists() else (0,0,0,0)
        rew_len, reg_len, play_len, exp_len = lengths
        csv_complete = is_csv_complete(csv_file, len(games), horizon)

        if all(x >= horizon for x in lengths):
            if not csv_complete:
                aggregate_metrics_from_single_pkl(str(pkl_file))
            continue
        # étendre l'horizon
        if pkl_file.exists() and not extend_games:
            with open(pkl_file, "rb") as f:
                state = pickle.load(f)
            start_iter = rew_len
            env_state_list = [Environnement.from_serialized(env_state) for env_state in state['env_state']]
            all_games_metrics_for_run = state['metrics']
        else:
            start_iter = 0
            env_state_list = [None] * len(games)
            all_games_metrics_for_run = []

        env_list = []
        for g in range(len(games)):
            game = games[f'game{g + 1}']
            raw = game["matrix"]
            arr = np.array(raw, dtype=float)

            if arr.ndim == 2:
                matrix = arr
                n_actions = matrix.shape[1]
                matrices = generate_n_player_diag(player, n_actions, matrix) if is_diagonal(matrix) else generate_n_player(
                    player, n_actions, matrix)
            elif arr.ndim == 3:
                matrices = [np.array(m, dtype=float) for m in raw]
                n_actions = matrices[0].shape[1]

            matrices_norm = [normalizeMatrix(mat, 0) for mat in matrices]

            regrets, rewards, plays, exploration_list, title, env = (
                Execute(runs, horizon, player, [None] * player, game['name'], n_actions).run_one_game(start_iter, env_state_list[g], matrices_norm, game['algos'], 'normal', game['noise'][0]))
            env_list.append(env)
            for agent_id in range(player):
                metrics_dict = {
                    "play_time": plays[agent_id].tolist(),
                    "reward_time": rewards[agent_id].tolist(),
                    "regret_time": regrets[agent_id].tolist(),
                    "exploration_time": exploration_list[agent_id].tolist(),
                }

                index = player * g + agent_id
                if len(all_games_metrics_for_run) < (len(games) * player):
                    flattened = flatten_metrics(
                        title=title,
                        player=f"agent_{agent_id}",
                        instance=r,
                        n_actions=n_actions,
                        start=start_iter,
                        metrics_dict=metrics_dict
                    )
                    all_games_metrics_for_run.append(flattened)
                else:
                    for key, val in metrics_dict.items():
                        for t, value in enumerate(val):
                            all_games_metrics_for_run[index][f"{key}{t + start_iter}"] = value

        save_pickle(folder, r, all_games_metrics_for_run, env_list, suffix=suffix)
        aggregate_metrics_from_single_pkl(str(pkl_file))
    if LAST_ACTIVE_RUN.exists():
        LAST_ACTIVE_RUN.unlink()