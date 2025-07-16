import pickle

with open("../Figures/Test_Pickle_interrupt/pkl/cp_game7_run4_iter25.pkl", "rb") as f:
    cp = pickle.load(f)

for entry in cp["metrics"]:
    print(entry)

with open("../Figures/Test_Pickle_interrupt/pkl/cp_game7_run4_iter50.pkl", "rb") as f:
    cp = pickle.load(f)

for entry in cp["metrics"]:
    print(entry)