import numpy as np
from agentSpace import AgentSpace

class LearningAlgo:
    def __init__(self, constant, algo_name, a_space: AgentSpace, noise_param):
        self.constant = constant
        self.algo_name = algo_name
        self.a_space = a_space
        self.init_sequence = np.random.permutation(a_space.n_arms)
        self.noise_param = noise_param

    def getInitialState(self):
        first_time = False
        action = 0
        if self.a_space.t <= self.a_space.n_arms:
            action = self.init_sequence[self.a_space.t-1]
            first_time = True

        return {'action': action, 'first_time': first_time }

    def getUCBAction(self, first_time, action):
        exploration = 1
        if not first_time:
            var = max(self.noise_param + .25, 1e-2)
            est_opt = np.sqrt(8 * var * np.log(self.a_space.t) / self.a_space.plays)
            action_val = self.a_space.avg_reward + est_opt

            best = np.flatnonzero(action_val == action_val.max())
            if best.size == 1:
                action = int(best[0])
            else:
                action = int(np.random.choice(best))

            best_greedy = np.flatnonzero(self.a_space.avg_reward == self.a_space.avg_reward.max())
            exploration = int(action not in best_greedy)

        return action, exploration
    
    def getTSAction(self, first_time, action):
        mu_0 = 1
        var_0 = 1
        var = max(self.noise_param + .25, 1e-2)
        exploration = 1
        if not first_time:
            mu_post = (mu_0/var_0 + self.a_space.sums/var) / (1/var_0 + self.a_space.plays/var)
            var_post = 1 / (1 / var_0 + self.a_space.plays / var)
            samples = np.random.normal(mu_post, np.sqrt(var_post))

            best = np.flatnonzero(samples == samples.max())
            if best.size == 1:
                action = int(best[0])
            else:
                action = int(np.random.choice(best))

            best_greedy = np.flatnonzero(mu_post == mu_post.max())
            exploration =  int(action not in best_greedy)

        return action, exploration

    def getKLUCBAction(self, first_time, action):
        var = max(self.noise_param + 0.25, 1e-2)
        c = 3
        exploration=1

        if not first_time:
            means = self.a_space.sums / self.a_space.plays
            f_t = 2 * var * (np.log(self.a_space.t) + c * np.log(np.log(self.a_space.t)))
            ucbs = means + np.sqrt(f_t / self.a_space.plays)

            best = np.flatnonzero(ucbs == ucbs.max())
            if best.size == 1:
                action = int(best[0])
            else:
                action = int(np.random.choice(best))

            best_greedy = np.flatnonzero(self.a_space.avg_reward == self.a_space.avg_reward.max())
            exploration =  int(action not in best_greedy)

        return action, exploration

    def getSoftMaxAction(self, first_time, action):
        tau = 1 / np.log(self.a_space.t + 1)
        exploration = 1

        if not first_time:
            total = np.sum(np.exp(self.a_space.avg_reward / tau))

            rand = np.random.random()
            probabilities = np.exp(self.a_space.avg_reward / tau) / total
            cumulative_probabilities = np.cumsum(probabilities)
            action = np.searchsorted(cumulative_probabilities, rand)
            best_greedy = np.flatnonzero(self.a_space.avg_reward == self.a_space.avg_reward.max())
            exploration = int(action not in best_greedy)

        return action, exploration

    def getAction(self):
        res = self.getInitialState()
        first_time = res['first_time']
        action = res['action']

        match self.algo_name:
            case "UCB":
                return self.getUCBAction(first_time, action)
            case "TS":
                return self.getTSAction(first_time, action)
            case "KLUCB":
                return self.getKLUCBAction(first_time, action)
            case "SoftMax":
                return self.getSoftMaxAction(first_time, action)
            case _:
                return None

    def serialize(self):
        return {
            'constant': self.constant,
            'algo_name': self.algo_name,
            'a_space': self.a_space.serialize(),
            'init_sequence': self.init_sequence.tolist(),
            'noise_param': self.noise_param
        }

    @staticmethod
    def from_serialized(data):
        a_space = AgentSpace.from_serialized(data['a_space'])
        algo = LearningAlgo(
            data['constant'],
            data['algo_name'],
            a_space,
            data['noise_param']
        )
        algo.init_sequence = np.array(data['init_sequence'])
        return algo