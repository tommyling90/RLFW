import argparse, shutil

from src.runResults import *
from src.runFigures import *
from src.utils import get_csv_line_count

def generate_figures(suffix):
    with open("graph_config.yaml", "r") as f:
        fig_config = yaml.safe_load(f)
    fig_defaults = fig_config['defaults']
    fig_games = fig_config['games']
    fig_root = Path(__file__).resolve().parent

    for key, val in fig_games.items():
        if len(val['algos']) > 1 and len(val['noise']) > 1 :
            raise "Too many pairs to compare. Either compare different algo combos on one noise level or different noise levels on one algo combo."
        cumul_y = val['cumul_y']
        generate_fig(cumul_y, val['algos'], val['noise'], val['name'], fig_defaults['n_actions'], f"{fig_root}/{fig_defaults['graph_folder']}", suffix)

def prune_pkls(pkl_folder):
    choice = input("⚠️ You're going to delete pkl files.\n"
                   "If you're certain that the experiment is FINALIZED and that you WILL NOT be extending/resuming, press Y to delete.\n"
                   "Otherwise press Q to quit and rename the folder in config.yaml.\n"
                   "[Y/Q]").strip().upper()
    if choice == "Y":
        print("✅ Deleting...")
        pkl_folder = Path(pkl_folder)
        if pkl_folder.exists() and pkl_folder.is_dir():
            shutil.rmtree(pkl_folder)
        print(f"Deleted folder: {pkl_folder}")
    elif choice == "Q":
        print("❌ Exiting.")
        sys.exit(0)
    else:
        print("❗ Invalid input. Exiting.")
        sys.exit(1)

def add(param, n):
    with open('config.yaml', "r") as f:
        config = yaml.safe_load(f)
    if param == 'runs':
        i = config['defaults']['runs']
        new_i = i + int(n)
        config['defaults']['runs'] = new_i
    elif param == 'horizon':
        i = config['defaults']['horizon']
        new_i = i + int(n)
        config['defaults']['horizon'] = new_i
    with open('config.yaml', "w") as f:
        yaml.safe_dump(config, f)
    print(f"[Config] Updated {param} from {i} to {new_i}")
    return config

def add_horizon(n):
    with open('config.yaml', "r") as f:
        config = yaml.safe_load(f)
    n_games = len(config['games'])
    folder = config['defaults']['save_folder']
    runs = config['defaults']['runs']
    horizon = config['defaults']['horizon']
    csv_files = list(Path(f'{folder}/output').glob("run*.csv"))
    lines = get_csv_line_count(f'{folder}/output/run{runs-1}.csv')
    if Path(folder) / "output" / f"run{runs-1}.csv" not in csv_files or lines != horizon * n_games:
        print("‼️Experiment not complete. Can extend horizon only to a complete experiment.")
        return
    add('horizon', n)

def main():
    parser = argparse.ArgumentParser(
        description="RL Framework CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    # Command 1: run_results
    parser_run = subparsers.add_parser(
        "run_results", help="Run experiments based on a YAML config"
    )
    parser_run.add_argument("--suffix_extend_games", required=False, default='', help="Add suffix when running extended games")

    # Command 2: generate_figures
    parser_graph = subparsers.add_parser(
        "generate_figures", help="Generate plots from a YAML figure config"
    )
    parser_graph.add_argument("--suffix", required=False, default=None, help="Add suffix to figure name")

    # Command 3: prune_pkls
    parser_prune = subparsers.add_parser(
        "prune_pkls", help="Deletes all pkl files except for the last one"
    )
    parser_prune.add_argument("--path", "-p", required=True, help="Path to pkl folder")

    # Command 4: add_runs
    parser_add_runs = subparsers.add_parser(
        "add_runs", help="Add more runs into the experiments"
    )
    parser_add_runs.add_argument("--n_runs", required=True, help="Number of runs to add")

    # Command 5: add_horizon
    parser_add_horizon = subparsers.add_parser(
        "add_horizon", help="Add more time steps into the experiments"
    )
    parser_add_horizon.add_argument("--n_horizon", required=True, help="Number of runs to add")

    args = parser.parse_args()
    if args.command == "run_results":
        run_results(args.suffix_extend_games)
    elif args.command == "generate_figures":
        generate_figures(args.suffix)
    elif args.command == "prune_pkls":
        prune_pkls(args.path)
    elif args.command == "add_runs":
        add('runs', args.n_runs)
    elif args.command == "add_horizon":
        add_horizon(args.n_horizon)

if __name__ == "__main__":
    main()
