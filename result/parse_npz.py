import numpy as np
import os
from metric import Metric

fnames = os.listdir('.')

for dataname in ['adult','broward','hospital']:
	for fname in fnames:
		if '.npz' not in fname:
			continue
		if dataname not in fname:
			continue
		z = np.load(fname)

		metric = Metric(true=-z['y_true'], pred=z['y_origin'][:,1])
		if metric.accuracy() < 0.5:
			metric = Metric(true=-z['y_true'], pred=z['y_origin'][:,0])
		print(fname, metric.accuracy())
