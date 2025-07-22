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

    args = parser.parse_args()
    if args.command == "run_results":
        run_results()
    elif args.command == "generate_figures":
        generate_figures()

if __name__ == "__main__":
    main()
