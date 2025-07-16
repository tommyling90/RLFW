import numpy as np

from agentSpace import AgentSpace
from learningAlgo import LearningAlgo
from agent import Agent
from environment import Environnement
from utils import normalizeMatrix, save_pickle

class Execute:
    def __init__(self, n_instance, T, n_agents, const, title, n_actions):
        self.n_instance = n_instance
        self.T = T
        self.n_agents = n_agents
        self.const = const
        self.title = title
        self.n_actions = n_actions

    def run_one_experiment(self, matrices, algo, noise_dist, noise_params, ctx, g, r):
        env = Environnement(matrices, noise_dist, noise_params)
        for agent in range(0, self.n_agents):
            a_space = AgentSpace(self.n_actions)
            learning_algo = LearningAlgo(self.const[agent], algo[agent], a_space, noise_params[1])
            env.ajouter_agents(Agent(a_space, learning_algo))

        title = f"{'×'.join(algo)}_{'_'.join(str(n) for n in noise_params)}_{self.title}"

        plays = np.zeros((self.n_agents, self.T))
        exploration_list = plays.copy()
        for i in range(0, self.T):
            actions, explorations = env.step()
            plays[:, i] = actions
            exploration_list[:, i] = explorations
            regrets = np.array([env.agents[k].regret for k in range(self.n_agents)])
            rewards = np.array([env.agents[k].reward for k in range(self.n_agents)])
        return regrets, rewards, plays, exploration_list, title

    def get_one_game_result(self, matrices, algo, ctx, g, noise_dist, noise_params):
        matrices_norm = [normalizeMatrix(mat,0) for mat in matrices]

        for r in range(ctx.run_idx if ctx.game_idx == g else 0, self.n_instance):
            regrets, rewards, plays, exploration_list, title = self.run_one_experiment(matrices_norm, algo, noise_dist, noise_params, ctx, g, r)
            save_pickle(ctx, g, r, plays, exploration_list, regrets, rewards, title, self.n_actions)
            ctx.reset_after_run()