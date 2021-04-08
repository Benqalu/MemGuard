import numpy as np
import os, json
from metric import Metric

fnames = os.listdir('.')
intensities_params = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]

for attr in ['race', 'sex']:
	for dataname in ['adult','broward','hospital', 'compas']:
		res = {
			'target':{
				'accuracy':0.0,
				'precision_disparity':0.0,
				'recall_disparity':0.0,
			},
			'mia':{
				'accuracy':0.0,
				'precision_disparity':0.0,
				'recall_disparity':0.0,
			},
			'mia_defense':{
				'accuracy':0.0,
				'precision_disparity':0.0,
				'recall_disparity':0.0,
			},
			'data':dataname,
			'attr':attr,
			'intensities':np.array([0.,0.,0.,0.,0.,0.]),
			'count':0,
		}
		for fname in fnames:
			if '.npz' not in fname:
				continue
			if dataname not in fname:
				continue
			z = np.load(fname, allow_pickle=True)

			res['count']+=1

			metric = Metric(true=-z['y_true'], pred=z['y_origin'][:,1])
			if metric.accuracy() < 0.5:
				metric = Metric(true=-z['y_true'], pred=z['y_origin'][:,0])
			res['target']['accuracy'] += metric.accuracy()
			res['target']['precision_disparity'] += metric.precision_disparity(s=z[f's_{attr}'])
			res['target']['recall_disparity'] += metric.recall_disparity(s=z[f's_{attr}'])

			metric = Metric(true=z['l_true'], pred=z['l_origin'])
			res['mia']['accuracy'] += metric.accuracy()
			res['mia']['precision_disparity'] += metric.precision_disparity(s=z[f's_{attr}'])
			res['mia']['recall_disparity'] += metric.recall_disparity(s=z[f's_{attr}'])
		
			metric = Metric(true=z['l_true'], pred=z['l_defense'])
			res['mia_defense']['accuracy'] += metric.accuracy()
			res['mia_defense']['precision_disparity'] += metric.precision_disparity(s=z[f's_{attr}'])
			res['mia_defense']['recall_disparity'] += metric.recall_disparity(s=z[f's_{attr}'])
			
			intensities = z['intensities'].reshape(1)[0]

			res['intensities']+=np.array([intensities[item] for item in intensities_params])
		
		if res['count']>0:
			for item in res:
				if type(res[item]) is dict:
					for jtem in res[item]:
						res[item][jtem]/=res['count']
				if item=='intensities':
					res[item]/=res['count']
					res[item]=res[item].tolist()
			res['average_of']=res['count']
			del res['count']

			print(json.dumps(res))


