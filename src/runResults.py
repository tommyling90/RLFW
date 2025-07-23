import yaml
import sys
from pathlib import Path

from src.execute import Execute
from src.utils import *

root = Path(__file__).resolve().parent.parent

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

def run_results():
    config_path = root / "config.yaml"
    games, horizon, runs, player, seed, folder, config = open_config(config_path)

    if os.path.isdir(folder):
        choice = input("⚠️ Folder already exists.\n"
               "If you're continuing an experiment that was interrupted or running more runs, press Y to continue.\n"
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

    with open(f"{folder}/config.yaml", 'w') as f:
        yaml.dump(config, f)

    rng_state = None

    pkl_path = f"{folder}/pkl"
    latest_file = find_latest_checkpoint(pkl_path)
    if latest_file:
        with open(latest_file, "rb") as f:
            cp = pickle.load(f)
        run_idx = cp['run_idx']
        rng_state = cp['rng_state']
        recover_last_csv(f"{folder}/pkl/")
    else:
        run_idx = 0

    np.random.set_state(rng_state) if rng_state is not None else np.random.seed(seed)

    for r in range(run_idx, runs):
        all_games_metrics_for_run = []
        for g in range(0, len(games)):
            game = games[f'game{g + 1}']
            matrix = np.array(game['matrix'])
            n_actions = len(matrix[0])
            matrices = generate_n_player_diag(player, n_actions, matrix) if is_diagonal(matrix) else generate_n_player(
                player, n_actions, matrix)

            matrices_norm = [normalizeMatrix(mat, 0) for mat in matrices]

            regrets, rewards, plays, exploration_list, title = (
                Execute(runs, horizon, player, [None] * player, game['name'], n_actions).run_one_game(matrices_norm, game['algos'], 'normal', game['noise'][0]))

            for agent_id in range(player):
                metrics_dict = {
                    "play_time": plays[agent_id].tolist(),
                    "reward_time": rewards[agent_id].tolist(),
                    "regret_time": regrets[agent_id].tolist(),
                    "exploration_time": exploration_list[agent_id].tolist(),
                }
                all_games_metrics_for_run.append(flatten_metrics(
                    title=title,
                    player=f"agent_{agent_id}",
                    instance=r,
                    n_actions=n_actions,
                    metrics_dict=metrics_dict
                ))

        save_pickle(folder, r, all_games_metrics_for_run)
        aggregate_metrics_from_single_pkl(f"{folder}/pkl/cp_run{r}.pkl")
