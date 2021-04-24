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

# Re-org dataset
# multiplier = size // 9
multiplier = int(min(100,dataset.shape[0]//4))
dataset = np.vstack([
	dataset[0*multiplier:2*multiplier,:],
	dataset[2*multiplier:4*multiplier,:],
])

# Get X,y,s
X = dataset[:, :-1]
y = dataset[:, -1].reshape(-1)
num_classes = len(set(y))
s_race = dataset[:, column.index("race")].reshape(-1)
s_sex = dataset[:, column.index("sex")].reshape(-1)
size = X.shape[0]
print('>>>>> Multiplier = %d <<<<<'%multiplier)

# Rewrite config file
f = open("config_template.txt")
content = "".join(f.readlines())
f.close()
base = tuple(
	(
		np.array([0, 2, 2, 4, 0, 2, 2, 4, 2, 0, 2, 2, 4, 2, 2, 0, 2, 2, 4]) * multiplier
	).tolist()
	+ [num_classes]
)
content = content % base
f = open("config.ini", "w")
f.write(content)
f.close()

# Clean files
import os
try:
	os.remove("./data/location/shuffle_index.npz")
except FileNotFoundError:
	pass
try:
	os.remove("./data/location/data_complete.npz")
except FileNotFoundError:
	pass
try:
	os.remove("./result/location/code_publish/attack/mia_results.npz")
except FileNotFoundError:
	pass

# Generate replacement npz
np.savez("./data/location/data_complete.npz", x=X, y=y)
np.savez("./data/location/shuffle_index.npz", x=np.arange(0, X.shape[0]))

# Run MemGuard

cmd = "python train_user_classification_model.py -dataset location"
print(">>> Running >>>", cmd)
os.system(cmd)
cmd = "python train_defense_model_defensemodel.py -dataset location"
print(">>> Running >>>", cmd)
os.system(cmd)
cmd = "python defense_framework.py -dataset location -qt evaluation"
print(">>> Running >>>", cmd)
os.system(cmd)
cmd = "python train_attack_shadow_model.py -dataset location -adv adv1"
print(">>> Running >>>", cmd)
os.system(cmd)
cmd = "python evaluate_nn_attack.py -dataset location -scenario full -version v0"
print(">>> Running >>>", cmd)
os.system(cmd)

# Prepare final result
result = np.load("./result/location/code_publish/attack/mia_results.npz", allow_pickle=True)['report'].reshape(1)[0]
final_s_race = np.hstack(
	[s_race[0 * multiplier : 2 * multiplier], s_race[2 * multiplier : 4 * multiplier]]
)
final_s_sex = np.hstack(
	[s_sex[0 * multiplier : 2 * multiplier], s_sex[2 * multiplier : 4 * multiplier]]
)

# Save to file with timestamp
import time

ts = int(time.time())
np.savez(
	f"./result/{data}_{ts}.npz",
	y_true=result["y_true"],
	y_origin=result["y_origin"],
	y_defense=result["y_defense"],
	s_race=final_s_race,
	s_sex=final_s_sex,
	l_true=result["l_true"],
	l_origin=result["l_pred_orig"],
	l_defense=result["l_pred_defense"],
	intensities=result['intensities'],
)
