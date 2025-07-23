import argparse

from src.runResults import *
from src.runFigures import *

def generate_figures():
    with open("graph_config.yaml", "r") as f:
        fig_config = yaml.safe_load(f)
    fig_defaults = fig_config['defaults']
    fig_games = fig_config['games']
    fig_root = Path(__file__).resolve().parent

    for key, val in fig_games.items():
        if len(val['algos']) > 1 and len(val['noise']) > 1 :
            raise "Too many pairs to compare. Either compare different algo combos on one noise level or different noise levels on one algo combo."
        cumul_y = val['cumul_y']
        generate_fig(cumul_y, val['algos'], val['noise'], val['name'], fig_defaults['n_actions'], f"{fig_root}/{fig_defaults['graph_folder']}")

def prune_pkls(pkl_folder):
    pkl_files = list(Path(pkl_folder).glob("cp_run*.pkl"))
    if not pkl_files:
        print("[Prune] Pkl folder empty.")
        return

    run_indices = []
    pattern = re.compile(r"cp_run(\d+)\.pkl")
    for f in pkl_files:
        match = pattern.match(f.name)
        if match:
            run_indices.append((int(match.group(1)), f))

    if len(run_indices) <= 1:
        print("[Prune] Must have the latest pkl file. Cannot delete.")
        return

    run_indices.sort()
    *to_delete, last = run_indices
    for _, f in to_delete:
        try:
            f.unlink()
            print(f"[Prune] Deleted: {f}")
        except Exception as e:
            print(f"[Prune] Failed to delete: {e}")
    print(f"[Prune] Kept latest: {last[1]}")

def add_runs(n):
    with open('config.yaml', "r") as f:
        config = yaml.safe_load(f)
    runs = config['defaults']['runs']
    new_runs = runs + int(n)
    config['defaults']['runs'] = new_runs
    with open('config.yaml', "w") as f:
        yaml.safe_dump(config, f)
    print(f"[Config] Updated runs from {runs} to {new_runs}")
    return config

def main():
    parser = argparse.ArgumentParser(
        description="RL Framework CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    # Command 1: run_results
    subparsers.add_parser(
        "run_results", help="Run experiments based on a YAML config"
    )

    # Command 2: generate_figures
    subparsers.add_parser(
        "generate_figures", help="Generate plots from a YAML figure config"
    )

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

    args = parser.parse_args()
    if args.command == "run_results":
        run_results()
    elif args.command == "generate_figures":
        generate_figures()
    elif args.command == "prune_pkls":
        prune_pkls(args.path)
    elif args.command == "add_runs":
        add_runs(args.n_runs)

if __name__ == "__main__":
    main()
