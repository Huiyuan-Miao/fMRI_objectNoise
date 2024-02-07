'''
runs featquery analysis
'''

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# get scan info from experiment file
from fMRI.analysis.scripts.experiment import experiment
buffer = 3 # 3 volumes either side of epoch

# get conditions
# relOri = ['-90', '-45', '-15', '0', '15', '45', '90'] #-for M012 and M015
# relOri_num = [-90,-45,-15,0,15,45,90] #-for M012 and M015
relOri = ['-90', '-30', '-15', '0', '15', '30', '90']
relOri_num = [-90,-30,-15,0,15,30,90]

figSize = (4, 3)
for condition in ['center','surround']:
	dataFile = f'/home/tonglab/Miao/fMRI/figureGround/fMRI/data/group/{condition}_timeseries.csv'
	data = pd.read_csv(dataFile)

	# add columns for region, hemisphere and voxel counts
	regions, hemispheres, nVoxels = [[], [], []]
	for row in data.index:
		regions.append(data['region'][row].split(sep='_')[0])
		hemispheres.append(data['region'][row].split(sep='_')[1])
		nVoxels.append(data['region'][row].split(sep='_')[2])
	data['region'] = regions
	data['hemisphere'] = hemispheres
	data['nVoxels'] = nVoxels

	# normed PSC
	baseWindow=np.array([-3,-2,-1]).astype(int)


	for topup in ['noTopUp']:#, 'topUp']:
		for b0 in ['noB0']:#,'b0']:
			for HRFmodel in ['doubleGamma']:#, 'singleGamma', 'singleGamma034']:
				for subject in list(experiment['scanInfo'].keys()):
					for session in experiment['scanInfo'][subject].keys():


						'''
						# debugging
						distCor = 'topUp'
						HRFmodel = 'doubleGamma'
						subject = 'M015'
						session = 201215
						'''

						# EXPERIMENT
						scan = 'figure_ground'
						params = experiment['design']['figure_ground']['params']
						epochDur = int((params['blockDuration'] + params['IBI']) / params['TR'])
						buffer = 3  # 3 volumes either side of epoch
						xticks = np.arange(-buffer, epochDur + buffer + 1) * params[
							'TR']  # time points fall in between measurements
						x_pos = xticks[:-1]
						# x_pos = xticks[:-1] + 1  # points are shifted to centre on middle of TR


						outDir = os.path.join('/home/tonglab/Miao/fMRI/figureGround/fMRI/analysis/results', topup, b0, HRFmodel, subject, session, scan, condition)
						os.makedirs(outDir, exist_ok=True)
						dataConf = data[(data['topup'] == topup) &
										(data['b0'] == b0) &
										(data['HRFmodel'] == HRFmodel) &
										(data['subject'] == subject) &
										(data['session'] == int(session)) &
										(data['scan'] == scan)].reset_index()

						# baseline currently no baseline correction
						lastRow = 0
						PSCbaselined = []
						while lastRow < len(dataConf):
							theseRows = np.arange(lastRow, lastRow + 15)
							baselineVals = dataConf['PSC'][theseRows[tuple([baseWindow + 3])]].astype(float)
							baseline = np.mean(baselineVals)
							for row in theseRows:
								if dataConf['PSC'][row] != 'None':
									PSCbaselined.append(float(dataConf['PSC'][row]) - baseline)
									# PSCbaselined.append(float(dataConf['PSC'][row]))
								else:
									PSCbaselined.append('None')
							lastRow = row + 1
						dataConf = dataConf.assign(PSCbaselined=PSCbaselined)


						# make plots for each configuration of parameters
						for region in dataConf['region'].unique():
							dataRegion = dataConf[dataConf['region'] == region]
							for size in dataRegion['nVoxels'].unique():
								dataSize = dataRegion[dataRegion['nVoxels'] == size]

								plt.figure(figsize=figSize)

								for ori in relOri:
									means = np.empty([epochDur + buffer * 2])
									errorbar = np.empty([epochDur + buffer * 2])
									for timepoint in range(epochDur + buffer * 2):
										values_exp = np.array(dataSize['PSCbaselined'][(dataSize['condition']==ori) & (dataSize['TR'] == timepoint - buffer)])
										values_exp = np.delete(values_exp, [x == 'None' for x in values_exp])
										values_exp = values_exp.astype(float)
										means[timepoint] = np.mean(values_exp)
										errorbar[timepoint] = stats.sem(values_exp)
									plt.errorbar(x_pos,
												 means,
												 yerr=errorbar,
												 marker='.',
												 markersize=5,
												 label=ori)
								plt.legend(bbox_to_anchor=(1.04, .75), borderaxespad=0)  # put legend outside plot
								plt.xticks(xticks)
								plt.xlabel('time (s)')
								plt.ylabel('signal change (%)')
								plt.title(f'timeseries: {region}, {size} voxels')
								#plt.tight_layout(rect=[0, 0, 0.75, 1])  # ensure everything is placed in the canvas
								plt.subplots_adjust(right=1.4)  # allow space for legend
								plt.savefig(os.path.join(outDir, f'{region}_{size}_timeseries.png'), bbox_inches='tight')
								plt.show()
								plt.close()

								# PARAMETER ESTIMATES FROM TIMESERIES
								# alternative to featquery analyses
								# timeWindow = (2,12)  # in seconds, inclusive
								timeWindow = (2,12)  # in seconds, inclusive
								volWindow = np.array(timeWindow) / params['TR']
								vols2use = np.arange(volWindow[0], volWindow[1] + 1)
								dataWindow = dataSize[(dataSize['TR'].isin(vols2use))]
								dataWindow = dataWindow.astype({'PSCbaselined': 'float'})  # convert PSC column to float
								#dataWindowMeanAcrossReps = dataWindow.groupby(['orientation', 'subject']).mean()
								#dataWindowMeanAcrossReps = dataWindowMeanAcrossReps[['PSC']]
								dataWindow = dataWindow[['PSCbaselined','condition']]
								dataWindowMeans = dataWindow.groupby(['condition']).mean()
								dataWindowSems = dataWindow.groupby(['condition']).sem()

								means = []
								sems = []
								condsFull = []
								plt.figure(figsize=(5,3))
								# for o, ori in enumerate(dataWindowMeans.index):
								for ori in relOri:
									condsFull.append(f'{ori}')
									means.append(dataWindowMeans['PSCbaselined'][ori])
									sems.append(dataWindowSems['PSCbaselined'][ori])
								#barColors = ['red', 'darkred', 'blue', 'darkblue']
								plt.errorbar(relOri_num,
											 means,
											 yerr=sems,
											 marker='.',
											 markersize=5)
								plt.xticks(relOri_num)
								plt.xlabel('relative orientation')
								plt.ylabel('signal change (%)')
								plt.title(f'timeseries: {region}, {size} voxels')
								plt.tight_layout()  # ensure everything is placed in the canvas
								plt.savefig(f'{outDir}/{region}_{size}_PSCts.png')
								plt.show()
								plt.close()

						# LOCALIZER
						scan = 'localizer'
						params = experiment['design']['localizer']['params']
						epochDur = int((params['blockDuration'] + params['IBI']) / params['TR'])
						buffer = 3  # 3 volumes either side of epoch
						xticks = np.arange(-buffer, epochDur + buffer + 1) * params[
							'TR']  # time points fall in between measurements
						x_pos = xticks[:-1] + 1  # points are shifted to centre on middle of TR

						outDir = os.path.join('/home/tonglab/Miao/fMRI/figureGround/fMRI/analysis/results', topup, b0, HRFmodel, subject, session, scan,condition)
						os.makedirs(outDir, exist_ok=True)
						dataConf = data[(data['topup'] == topup) &
										(data['b0'] == b0) &
										(data['HRFmodel'] == HRFmodel) &
										(data['subject'] == subject) &
										(data['session'] == int(session)) &
										(data['scan'] == scan)].reset_index()

						# baseline currently no baseline correction
						lastRow = 0
						PSCbaselined = []
						while lastRow < len(dataConf):
							theseRows = np.arange(lastRow, lastRow + 18)
							baselineVals = dataConf['PSC'][theseRows[tuple([baseWindow + 3])]].astype(float)
							baseline = np.mean(baselineVals)
							for row in theseRows:
								if dataConf['PSC'][row] != 'None':
									PSCbaselined.append(float(dataConf['PSC'][row]) - baseline)
									# PSCbaselined.append(float(dataConf['PSC'][row]))
								else:
									PSCbaselined.append('None')
							lastRow = row + 1
						dataConf = dataConf.assign(PSCbaselined=PSCbaselined)

						# make plots for each configuration of parameters
						for region in dataConf['region'].unique():
							dataRegion = dataConf[dataConf['region'] == region]
							for size in dataRegion['nVoxels'].unique():
								dataSize = dataRegion[dataRegion['nVoxels'] == size]

								plt.figure(figsize=figSize)
								for stiLoc in ['center','surround']:
									means = np.empty([epochDur + buffer * 2])
									errorbar = np.empty([epochDur + buffer * 2])
									for timepoint in range(epochDur + buffer * 2):
										values_loc = np.array(dataSize['PSCbaselined'][(dataSize['condition']==stiLoc)&(dataSize['TR'] == timepoint - buffer)])
										values_loc = np.delete(values_loc, [x == 'None' for x in values_loc])
										values_loc = values_loc.astype(float)
										means[timepoint] = np.mean(values_loc)
										errorbar[timepoint] = stats.sem(values_loc)
									plt.errorbar(x_pos,
												 means,
												 yerr=errorbar,
												 marker='.',
												 markersize=5,
												 label = stiLoc)
							plt.legend(bbox_to_anchor=(1.04, .75), borderaxespad=0)  # put legend outside plot
							plt.xticks(xticks)
							plt.xlabel('time (s)')
							plt.ylabel('signal change (%)')
							plt.title(f'timeseries: {region}, {size} voxels')
							#plt.tight_layout(rect=[0, 0, 0.75, 1])  # ensure everything is placed in the canvas
							plt.subplots_adjust(right=1.4)  # allow space for legend
							plt.savefig(os.path.join(outDir, f'{region}_{size}_timeseries.png'), bbox_inches='tight')
							plt.show()
							plt.close()


print('Done.')
