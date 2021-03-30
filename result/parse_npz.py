import numpy as np
import os

fnames = os.listdir('.')

for fname in fnames:
	if '.npz' not in fname:
		continue
	z = np.load(fname)
	true = z['l_true']
	pred = (z['l_origin']>0.5).astype(int)
	pred_def = (z['l_defense']>0.5).astype(int)
	print(fname, (true==pred).sum()/true.shape[0], (true==pred_def).sum()/true.shape[0],)