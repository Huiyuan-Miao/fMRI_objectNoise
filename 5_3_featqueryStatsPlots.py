#!/usr/bin/python
'''
runs featquery analysis
'''

import os
import pickle
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# get scan info from experiment file
from fMRI.analysis.scripts.experiment import experiment

# get conditions
# relOri = ['-90', '-45', '-15', '0', '15', '45', '90'] #-for M012 and M015
# relOri_num = [-90,-45,-15,0,15,45,90] #-for M012 and M015
relOri = ['-90', '-30', '-15', '0', '15', '30', '90']
relOri_num = [-90,-30,-15,0,15,30,90]

for condition in ['center','surround']:
	dataFile = f'/home/tonglab/Miao/fMRI/figureGround/fMRI/data/group/{condition}_featquery.csv'
	data = pd.read_csv(dataFile, index_col = False)

	# add columns for region, hemisphere and voxel counts
	regions, hemispheres, nVoxels = [[], [], []]
	for row in data.index:
		regions.append(data['roi'][row].split(sep='_')[0])
		hemispheres.append(data['roi'][row].split(sep='_')[1])
		nVoxels.append(data['roi'][row].split(sep='_')[2])
	data['region'] = regions
	data['hemisphere'] = hemispheres
	data['nVoxels'] = nVoxels

	for topup in ['topUp']:#, 'noTopUp']:
		for b0 in ['noB0']:#,'b0']:
			for HRFmodel in ['doubleGamma']:#, 'singleGamma', 'singleGamma034']:
				for subject in list(experiment['scanInfo'].keys()):
					for session in experiment['scanInfo'][subject].keys():

						# EXPERIMENT
						scan = 'figure_ground'
						outDir = os.path.join('/home/tonglab/Miao/fMRI/figureGround/fMRI/analysis/results', topup, b0, HRFmodel, subject, session, scan,condition)
						os.makedirs(outDir, exist_ok=True)
						dataConf = data[(data['topup'] == topup) &
										(data['b0'] == b0) &
										(data['HRFmodel'] == HRFmodel) &
										(data['subject'] == subject) &
										(data['session'] == int(session)) &
										(data['scan'] == scan)]

						# make plots for each configuration of parameters
						for region in dataConf['region'].unique():
							dataRegion = dataConf[dataConf['region'] == region]
							for size in dataRegion['nVoxels'].unique():
								dataSize = dataRegion[dataRegion['nVoxels'] == size]
								dataSize =dataSize[['PSC','condition']]
								dataMeans = dataSize.groupby(['condition']).mean()
								dataSems = dataSize.groupby(['condition']).sem()

								means = []
								sems = []
								condsFull = []
								plt.figure(figsize=(5, 3))
								for o, ori in enumerate(relOri):
									condsFull.append(f'{ori}')
									means.append(dataMeans['PSC'][dataMeans.index==ori].item())
									sems.append(dataSems['PSC'][dataMeans.index==ori].item())
								# barColors = ['red', 'darkred', 'blue', 'darkblue']
								plt.errorbar(relOri_num,
											 means,
											 yerr=sems,
											 marker='.',
											 markersize=5)
								plt.xticks(relOri_num)
								plt.xlabel('relative orientation')
								plt.ylabel('signal change (%)')
								plt.tight_layout()  # ensure everything is placed in the canvas
								plt.savefig(f'{outDir}/{region}_{size}_PSC.png')
								plt.show()
								plt.close()
print('Done.')




