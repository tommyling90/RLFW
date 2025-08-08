import pickle
import re

with open('../Figures/SM0.1-ComparaisonUCBTS_tauLog_100r50h/pkl/cp_run17.pkl', "rb") as f:
    checkpoint = pickle.load(f)

sample_metrics = checkpoint['metrics'][0]
iter_reward = [k for k in sample_metrics.keys() if re.match(r"reward_time\d+$", k)]
iter_regret = [k for k in sample_metrics.keys() if re.match(r"regret_time\d+$", k)]
iter_play = [k for k in sample_metrics.keys() if re.match(r"play_time\d+$", k)]
iter_exp = [k for k in sample_metrics.keys() if re.match(r"exploration_time\d+$", k)]
print(len(iter_reward), len(iter_regret), len(iter_play), len(iter_exp))