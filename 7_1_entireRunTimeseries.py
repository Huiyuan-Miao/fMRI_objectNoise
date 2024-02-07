#!/usr/bin/python
"""
get data for timeseries of each condition (in % signal change) and place in csv file

-convert filtered_func_data to % signal change
-separate by condition based on event files
-get timeseries for each condition * mask

"""

import os
import sys
import glob
import pickle
import datetime
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

overwrite = True

# get scan info from experiment file
from fMRI.analysis.scripts.experiment import experiment
figSize = (4, 3)
# set system to process NIFTI_GZ
os.system('export FSLOUTPUTTYPE=NIFTI_GZ')
resultsFile = f'/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/group/timeseries_entireRun.csv'
results = open(resultsFile, 'w')
results.write('topup,b0,HRFmodel,scan,subject,session,mask,region,hemi,numVoxel,run,TR,PSC\n')

for topup in ['topUp', 'noTopUp']:#, 'noTopUp']:
	for b0 in ['noB0']:#,'b0']:
		for HRFmodel in ['doubleGamma']:#, 'singleGamma', 'singleGamma034']:
			for scan in experiment['design'].keys():#['localizer']:#
				if scan == 'checkerboard_loc':
					params = experiment['design'][scan]['params']
					for subject in experiment['scanInfo'].keys():
						for session in experiment['scanInfo'][subject].keys():
							sessDir = f'/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/individual/{subject}/{session}'
							for run in range(len(experiment['scanInfo'][subject][session]['funcScans'][scan])):

								'''
								# for debugging
								distCor = 'topUp'
								HRFmodel = 'doubleGamma'
								scan = 'rapid_event_related_object_noise_v3'
								params = experiment['design'][scan]['params']
								subject = 'M015'
								session = '201215'
								run=0
								'''

								print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | topup : {topup} | b0: {b0} | HRF model: {HRFmodel} | Scan: {scan} | Subject: {subject} | Session: {session} | Run: {run+1}')
								runDir = os.path.join('/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/individual', subject, session, 'functional', scan,
													  f'run{run+1:02}/outputs', topup, b0, HRFmodel, 'firstLevel.feat')
								os.makedirs(os.path.join(runDir, 'timeseries'), exist_ok=True)

								if not os.path.exists(f'{runDir}/timeseries/filtered_func_percentSC.nii.gz') or overwrite:
									print('calculating mean...')
									os.system(f'fslmaths {runDir}/filtered_func_data -Tmean {runDir}/timeseries/Tmean')
									print('demeaning...')
									os.system(f'fslmaths {runDir}/filtered_func_data -sub {runDir}/timeseries//Tmean {runDir}/timeseries/filtered_func_demeaned')
									print('converting to percent MR signal...')
									os.system(f'fslmaths {runDir}/timeseries/filtered_func_demeaned -div {runDir}/timeseries/Tmean -mul 100 {runDir}/timeseries/filtered_func_percentSC')


								# create timeseries
								print('creating timeseries...')
								masks = sorted(glob.glob(os.path.join(sessDir, f'masksNative/checkerboard_loc_firstLevel',topup,b0,HRFmodel,'*.nii.gz')))
								for mask in masks:
									maskName = os.path.basename(mask)[:-7]
									print(maskName)

									# get mean time-series of mask for whole scan
									if not os.path.exists(f'{runDir}/timeseries/{maskName}.txt') or overwrite:
										os.system(f'fslmeants -i {runDir}/timeseries/filtered_func_percentSC -o {runDir}/timeseries/{maskName}.txt -m {mask}')
									f = open(f'{runDir}/timeseries/{maskName}.txt')
									ts_output = f.readlines()
									f.close()
									region = maskName.split(sep='_')[0]
									hemispheres=maskName.split(sep='_')[1]
									nVoxels=maskName.split(sep='_')[2]
									for i in range(len(ts_output)):
										results.write(
											f'{topup},{b0},{HRFmodel},{scan},{subject},{session},{maskName},{region},{hemispheres},{nVoxels},{run+1},{i},{ts_output[i]}\n')
results.close()
print('FINISHED')

