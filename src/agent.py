import numpy as np

from agentSpace import AgentSpace
from learningAlgo import LearningAlgo

class Agent:
    def __init__(self, a_space: AgentSpace, algo):

        self.learning_algo = algo
        self.a_space = a_space
        self.regret = []
        self.reward = []

    def update(self, action, step_reward, step_regret):
        self.a_space.plays[action] += 1
        self.a_space.sums[action] += step_reward
        self.a_space.avg_reward = np.divide(
            self.a_space.sums,
            self.a_space.plays,
            out=np.zeros_like(self.a_space.sums, dtype=float),
            where=self.a_space.plays != 0
        )
        self.regret.append(step_regret)
        self.reward.append(step_reward)

    def train(self):
        self.a_space.t += 1
        action, exploration = self.learning_algo.getAction()

        return action, exploration

    def serialize(self):
        return {
            'regret': self.regret,
            'reward': self.reward,
            'a_space': self.a_space.serialize(),
            'learning_algo': self.learning_algo.serialize(),
        }

    @staticmethod
    def from_serialized(data):
        learning_algo = LearningAlgo.from_serialized(data['learning_algo'])
        a_space = AgentSpace.from_serialized(data['a_space'])
        agent = Agent(a_space, learning_algo)
        agent.regret = data['regret']
        agent.reward = data['reward']
        return agent