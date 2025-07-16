import numpy as np

from agent import Agent

class Environnement:
    def __init__(self, matrices, noise_dist, noise_params):
        self.agents = []
        self.matrices = matrices
        self.noise_dist = noise_dist
        self.noise_params = noise_params

    def ajouter_agents(self, agent):
        self.agents.append(agent)

    def sample_noise(self):
        if self.noise_dist == 'normal':
            mean, var = self.noise_params
            std = np.sqrt(var)
            return np.random.normal(mean, std)
        else:
            raise ValueError(f"Unknown noise distribution: {self.noise_dist}")

    def updateStep(self, actions):
        rewards = []
        for i in range(len(self.agents)):
            rewards.append(self.matrices[i][*actions] + self.sample_noise())

        min_matrix = np.minimum(self.matrices[0], self.matrices[1])
        max_val = np.max(min_matrix)
        regret_matrix = max_val - min_matrix
        regret = regret_matrix[*actions]

        for i in range(len(self.agents)):
            self.agents[i].update(actions[i], rewards[i], regret)

    def step(self):
        acts, explorations = [], []
        for i in range(len(self.agents)):
            action, exp = self.agents[i].train()
            acts.append(action)
            explorations.append(exp)
        self.updateStep(acts)
        return acts, explorations

    def serialize(self):
        return {
            'matrices': self.matrices,
            'noise_dist': self.noise_dist,
            'noise_params': self.noise_params,
            'agents': [agent.serialize() for agent in self.agents]
        }

    @staticmethod
    def from_serialized(data):
        env = Environnement(
            data['matrices'],
            data['noise_dist'],
            data['noise_params']
        )
        env.agents = [
            Agent.from_serialized(agent_data)
            for i, agent_data in enumerate(data['agents'])
        ]
        return env