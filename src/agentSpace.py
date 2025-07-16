import numpy as np

class AgentSpace:
    def __init__(self, n_arms):
        self.target_plays = np.zeros(n_arms, dtype=int)
        self.plays = np.zeros(n_arms, dtype=int)
        self.avg_reward = np.zeros(n_arms)
        self.sums = np.zeros(n_arms)
        self.t = 0
        self.n_arms = n_arms

    def serialize(self):
        return {
            'target_plays': self.target_plays.tolist(),
            'plays': self.plays.tolist(),
            'avg_reward': self.avg_reward.tolist(),
            'sums': self.sums.tolist(),
            't': self.t,
            'n_arms': self.n_arms
        }

    @staticmethod
    def from_serialized(data):
        obj = AgentSpace(data['n_arms'])
        obj.target_plays = np.array(data['target_plays'])
        obj.plays = np.array(data['plays'])
        obj.avg_reward = np.array(data['avg_reward'])
        obj.sums = np.array(data['sums'])
        obj.t = data['t']
        return obj