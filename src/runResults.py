import yaml
import sys

from execute import Execute
from utils import *
from pickleContext import PickleContext

with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)

defaults = config['defaults']
games = config['games']

runs = defaults['runs']
horizon = defaults['horizon']
player = defaults['player']
folder = f"../{defaults['save_folder']}"

def run_results():
    if os.path.isdir(folder):
        choice = input("⚠️ Folder already exists.\n"
               "If you're continuing an experiment that was interrupted, press Y to continue.\n"
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
        game_idx = cp['game_idx']
        run_idx = cp['run_idx']
        rng_state = cp['rng_state']
    else:
        game_idx = run_idx = 0

    np.random.set_state(rng_state) if rng_state is not None else np.random.seed(defaults['seed'])
    ctx = PickleContext(game_idx, run_idx, folder)

    for g in range(game_idx, len(games)):
        game = games[f'game{g+1}']
        matrix = np.array(game['matrix'])
        n_actions = len(matrix[0])
        matrices = generate_n_player_diag(player, n_actions, matrix) if is_diagonal(matrix) else generate_n_player(
            player, n_actions, matrix)
        Execute(runs, horizon, player, [None] * player, game['name'], n_actions).get_one_game_result(
            matrices, game['algos'], ctx, g, 'normal', game['noise'][0])
        ctx.reset_after_game()