import os
import itertools
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.utils import parse_string
from src.runStats import runStats

sns.set_theme(style="whitegrid", palette="colorblind")
sns.despine(trim=True)

def plot_results(games, save_folder, compare):
    plt.figure(figsize=(5, 3))
    for game in games:
        title = game['experiment']
        mean = np.array(game['metrics']['mean_cum_regret']['agent_0'])
        std = np.array(game['metrics']['std_cum_regret']['agent_0'])
        algos, noise, game = parse_string(title)
        algos_arr = algos.split('x')
        n_rounds = len(mean)
        x = np.arange(n_rounds)
        sep = r"$\times$"
        noiseLegend = f'noise {noise}'
        line = plt.plot(
            x,
            mean,
            label=f"{sep.join(algos_arr) if compare == 'algos' else noiseLegend}",
            linewidth=1,
        )
        plt.plot(
            x,
            mean + std,
            linestyle='--',
            alpha=0.5,
            color=line[0].get_color(),
            linewidth=1,
        )

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0)
    ax.set_xlim(0, n_rounds)
    plt.xlabel(f"Round (t) {sep.join(algos_arr) if compare == 'noise' else ''}")
    plt.ylabel("Mean cumulative regret R(t)")
    legend = plt.legend()
    legend.get_frame().set_linewidth(0)
    legend.get_frame().set_facecolor('white')
    plt.tight_layout()
    fileN = f"{game}_{noise}" if compare == 'algos' else f"{game}_compare_noise{algos}"
    plt.savefig(f"{save_folder}/{fileN}.pdf", dpi=300)
    plt.close()

def plot_action_prop(games, save_folder):
    plt.figure(figsize=(5, 3))
    for game in games:
        title = game['experiment']
        props = game['metrics']['vecteur_de_props']
        [n_actions, n_agents] = game['shape']
        labels = list(itertools.product(range(1, n_actions + 1), repeat=n_agents))
        algos, noise, game = parse_string(title)
        algos_arr = algos.split('x')
        n_rounds = props.shape[0]
        x = np.arange(n_rounds)

        for i in range(props.shape[1]):
            plt.plot(
                x,
                props[:, i],
                label=labels[i],
                linewidth=1,
        )

    sep = r"$\times$"
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0)
    ax.set_xlim(0, n_rounds)
    plt.title(sep.join(algos_arr))
    plt.xlabel(f"Round (t)")
    plt.ylabel("Proportion of actions")
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.22),
        ncol=4,
        frameon=False
    )
    plt.tight_layout()
    plt.savefig(f"{save_folder}/{game}_{algos}_{noise}.pdf", dpi=300)
    plt.close()

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "figure.dpi": 300,
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8
})

def generate_fig(cumul_y, algos, noise, name, n_actions, folder):
    games, games_result = [], []
    for algo in algos:
        for n in noise:
            algoCombo = "×".join(algo)
            games.append(f"{algoCombo}_0.0_{n}_{name}")
    for g in games:
        res, subDir = runStats(f"{folder}/output/", g, n_actions)
        games_result.append(res)
    newDir = f"{folder}/{cumul_y}/{subDir}"
    os.makedirs(newDir, exist_ok=True)
    if cumul_y == 'regret':
        plot_results(games_result, newDir, 'algos' if len(algos) > 1 else 'noise')
    elif cumul_y == 'prop':
        plot_action_prop(games_result, newDir)