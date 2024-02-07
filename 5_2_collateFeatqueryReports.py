#!/usr/bin/python
'''
runs featquery analysis
'''

import os
import glob
import shutil
import numpy as np

overwrite = True

# get scan info from experiment file
from fMRI.analysis.scripts.experiment import experiment
for condition in ['center','surround']:
	resultsFile = f'/home/tonglab/Miao/fMRI/figureGround/fMRI/data/group/{condition}_featquery.csv'
	os.makedirs('/home/tonglab/Miao/fMRI/figureGround/fMRI/data/group/', exist_ok=True)

	results = open(resultsFile, 'w')
	results.write('topup,b0,HRFmodel,subject,session,scan,run,roi,condition,PSC,lower,upper\n')

	for topup in ['topUp']:#, 'noTopUp']:
		for b0 in ['noB0']:#,'b0']:
			for HRFmodel in ['doubleGamma']:#, 'singleGamma', 'singleGamma034']:
				for subject in experiment['scanInfo'].keys():
					for session in experiment['scanInfo'][subject].keys():
						sessDir = os.path.join('/home/tonglab/Miao/fMRI/figureGround/fMRI/data/individual', subject, session)
						for scan in experiment['design'].keys():

							if scan == 'figure_ground':
								# conds = ['-90','-45','-15','0','15','45','90', 'allConds'] #-forM012&M015
								conds = ['-90','-30','-15','0','15','30','90', 'allConds']
							elif scan == 'localizer':
								conds = ['center','surround','allConds','centerSubSurround']

							for run in range(len(experiment['scanInfo'][subject][session]['funcScans'][scan])):
								runDir = os.path.join('/home/tonglab/Miao/fMRI/figureGround/fMRI/data/individual', subject, session, 'functional', scan,
													  f'run{run + 1:02}/outputs', topup, b0, HRFmodel, 'firstLevel.feat')
								masks = sorted(glob.glob(os.path.join(sessDir, f'masksNative/{condition}/*.nii.gz')))
								for mask in masks:
									maskName = os.path.basename(mask)[:-7]
									reportFile = os.path.join(runDir, f'featquery/{condition}', maskName, 'report.txt')
									if os.path.isfile(reportFile):

										report = open(reportFile, 'r')
										report = report.readlines()
										for c, cond in enumerate(conds):
											info = report[c].split(' ')
											lower, PSC, upper = np.array(info)[[4,5,7]]
											results.write(f'{topup},{b0},{HRFmodel},{subject},{session},{scan},{run},{maskName},{cond},{PSC},{lower},{upper}\n')
	results.close()

