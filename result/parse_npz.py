import numpy as np
import os, json
from metric import Metric

fnames = os.listdir('.')
intensities_params = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]

for attr in ['race', 'sex']:
	for dataname in ['adult','broward','hospital', 'compas']:
		res = {
			'target_train':{
				'accuracy':0.0,
				'precision':0.0,
				'recall':0.0,
				'accuracy_groups': np.array([0.0, 0.0]),
				'precision_groups': np.array([0.0, 0.0]),
				'recall_groups': np.array([0.0, 0.0]),
			},
			'target_test':{
				'accuracy':0.0,
				'precision':0.0,
				'recall':0.0,
				'accuracy_groups': np.array([0.0, 0.0]),
				'precision_groups': np.array([0.0, 0.0]),
				'recall_groups': np.array([0.0, 0.0]),
			},
			'mia':{
				'accuracy':0.0,
				'precision':0.0,
				'recall':0.0,
				'accuracy_groups': np.array([0.0, 0.0]),
				'precision_groups': np.array([0.0, 0.0]),
				'recall_groups': np.array([0.0, 0.0]),
			},
			'mia_defense':{
				'accuracy':0.0,
				'precision':0.0,
				'recall':0.0,
				'accuracy_groups': np.array([0.0, 0.0]),
				'precision_groups': np.array([0.0, 0.0]),
				'recall_groups': np.array([0.0, 0.0]),
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

			dividor = z['y_true'].shape[0] // 2

			# target_train
			metric = Metric(true=-z['y_true'][:dividor], pred=z['y_origin'][:,1][:dividor])
			if metric.accuracy() < 0.5:
				metric = Metric(true=-z['y_true'][:dividor], pred=z['y_origin'][:,0][:dividor])
			res['target_train']['accuracy'] += metric.accuracy()
			res['target_train']['precision'] += metric.precision()
			res['target_train']['recall'] += metric.recall()
			res['target_train']['accuracy_groups'] += metric.accuracy_groups(s=z[f's_{attr}'][:dividor])
			res['target_train']['precision_groups'] += metric.precision_groups(s=z[f's_{attr}'][:dividor])
			res['target_train']['recall_groups'] += metric.recall_groups(s=z[f's_{attr}'][:dividor])

			# target_test
			metric = Metric(true=-z['y_true'][dividor:], pred=z['y_origin'][:,1][dividor:])
			if metric.accuracy() < 0.5:
				metric = Metric(true=-z['y_true'][dividor:], pred=z['y_origin'][:,0][dividor:])
			res['target_test']['accuracy'] += metric.accuracy()
			res['target_test']['precision'] += metric.precision()
			res['target_test']['recall'] += metric.recall()
			res['target_test']['accuracy_groups'] += metric.accuracy_groups(s=z[f's_{attr}'][dividor:])
			res['target_test']['precision_groups'] += metric.precision_groups(s=z[f's_{attr}'][dividor:])
			res['target_test']['recall_groups'] += metric.recall_groups(s=z[f's_{attr}'][dividor:])

			# mia original
			metric = Metric(true=z['l_true'], pred=z['l_origin'])
			res['mia']['accuracy'] += metric.accuracy()
			res['mia']['precision'] += metric.precision()
			res['mia']['recall'] += metric.recall()
			res['mia']['accuracy_groups'] += metric.accuracy_groups(s=z[f's_{attr}'])
			res['mia']['precision_groups'] += metric.precision_groups(s=z[f's_{attr}'])
			res['mia']['recall_groups'] += metric.recall_groups(s=z[f's_{attr}'])
		
			metric = Metric(true=z['l_true'], pred=z['l_defense'])
			res['mia_defense']['accuracy'] += metric.accuracy()
			res['mia_defense']['precision'] += metric.precision()
			res['mia_defense']['recall'] += metric.recall()
			res['mia_defense']['accuracy_groups'] += metric.accuracy_groups(s=z[f's_{attr}'])
			res['mia_defense']['precision_groups'] += metric.precision_groups(s=z[f's_{attr}'])
			res['mia_defense']['recall_groups'] += metric.recall_groups(s=z[f's_{attr}'])
			
			intensities = z['intensities'].reshape(1)[0]

			res['intensities']+=np.array([intensities[item] for item in intensities_params])
		
		if res['count']>0:
			for item in res:
				if type(res[item]) is dict:
					for jtem in res[item]:
						res[item][jtem]/=res['count']
						if type(res[item][jtem]) is np.ndarray:
							res[item][jtem] = res[item][jtem].tolist()
				if item=='intensities':
					res[item]/=res['count']
					res[item]=res[item].tolist()
			res['average_of']=res['count']
			del res['count']

			print(json.dumps(res))


