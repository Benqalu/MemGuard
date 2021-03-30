import sys

# Parse args value
if len(sys.argv) > 1:
    data = sys.argv[1]
else:
    data = "adult"

# Read dataset
import pandas as pd

csv = pd.read_csv(f"./data/{data}.csv")
dataset = csv.to_numpy()
column = list(csv.columns)

# Add randomness
import numpy as np

np.random.shuffle(dataset)

# Scaler to [0,1]
from sklearn.preprocessing import MinMaxScaler

dataset = MinMaxScaler().fit_transform(dataset)

# Get X,y,s
X = dataset[:, :-1]
y = dataset[:, -1].reshape(-1)
num_classes = len(set(y))
s_race = dataset[:, column.index("race")].reshape(-1)
s_sex = dataset[:, column.index("sex")].reshape(-1)
size = X.shape[0]
# multiplier = size // 9
multiplier = 100

# Rewrite config file
f = open("config_template.txt")
content = "".join(f.readlines())
f.close()
base = tuple(
    (
        np.array([0, 2, 2, 4, 0, 2, 2, 4, 2, 0, 2, 4, 6, 2, 2, 6, 7, 8, 9]) * multiplier
    ).tolist()
    + [num_classes]
)
content = content % base
f = open("config.ini", "w")
f.write(content)
f.close()

# Clean files
try:
	os.remove("./data/location/data_complete.npz")
except:
	pass
try:
	os.remove("./result/location/code_publish/attack/mia_results.npz")
except:
	pass

# Generate replacement npz
np.savez("./data/location/data_complete.npz", x=X, y=y)

# Run MemGuard
import os
cmd = "python train_user_classification_model.py -dataset location"
os.system(cmd)
cmd = "python train_defense_model_defensemodel.py -dataset location"
os.system(cmd)
cmd = "python defense_framework.py -dataset location -qt evaluation"
os.system(cmd)
cmd = "python train_attack_shadow_model.py -dataset location -adv adv1"
os.system(cmd)
cmd = "python evaluate_nn_attack.py -dataset location -scenario full -version v0"
os.system(cmd)

# Prepare final result
result = np.load("./result/location/code_publish/attack/mia_results.npz")
final_s_race = np.hstack(
    [s_race[0 * multiplier : 2 * multiplier], s_race[4 * multiplier : 6 * multiplier]]
)
final_s_sex = np.hstack(
    [s_sex[0 * multiplier : 2 * multiplier], s_sex[4 * multiplier : 6 * multiplier]]
)
final_l_true = result["true"]
final_l_origin = result["origin"]
final_l_defense = result["defense"]

# Save to file with timestamp
import time

ts = int(time.time())
np.savez(
    f"./result/{data}_{ts}.npz",
    s_race=final_s_race,
    s_sex=final_s_sex,
    l_true=final_l_true,
    l_origin=final_l_origin,
    l_defense=final_l_defense,
)
