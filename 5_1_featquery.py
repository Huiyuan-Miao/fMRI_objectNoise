#!/usr/bin/python
'''
runs featquery analysis
'''

import os
import glob
import shutil
import datetime

overwrite = True

# get scan info from experiment file
from fMRI.analysis.scripts.experiment import experiment

for topup in ['topUp']:#, 'noTopUp']:
	for b0 in ['noB0']:#,'b0']:
		for HRFmodel in ['doubleGamma']:#, 'singleGamma', 'singleGamma034']:
			nMasksRemoved = 0
			for subject in experiment['scanInfo'].keys():
				for session in experiment['scanInfo'][subject].keys():
					sessDir = os.path.join('/home/tonglab/Miao/fMRI/figureGround/fMRI/data/individual', subject, session)
					for scan in list(experiment['design'].keys())[0:2]:

						if scan == 'figure_ground':
							# conds = ['-90','-45','-15','0','15','45','90', 'allConds'] #-For M012&M015
							conds = ['-90','-30','-15','0','15','30','90', 'allConds']
						elif scan == 'localizer':
							conds = ['center','surround','allConds','centerSubSurround']

						for run in range(len(experiment['scanInfo'][subject][session]['funcScans'][scan])):

							'''
							distCor = 'topUp'
							HRFmodel = 'doubleGamma'
							subject = 'M015'
							session = '201215'
							sessDir = os.path.join('data/fMRI/individual', subject, session)
							scan = 'figureGround_v3'
							run=0
							'''
							runDir = os.path.join('/home/tonglab/Miao/fMRI/figureGround/fMRI/data/individual', subject, session, 'functional', scan,
												  f'run{run + 1:02}/outputs', topup, b0, HRFmodel, 'firstLevel.feat')
							for condition in ['surround','center']:
								''' if you want to use mask in the standard space, change the masksNative to masks'''
								masks = sorted(glob.glob(os.path.join(sessDir, f'masksNative/{condition}/*.nii.gz')))
								for mask in masks:
									maskName = os.path.basename(mask)[:-7]

									# double check that mask has > 0 voxels
									minVal = float(os.popen(f'fslstats {mask} -R').read().split()[0])
									if minVal == 1:
										nMasksRemoved += 1 # if there is no voxel in the mask, the entire volume is 1
									elif minVal < 1:
										print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | topup : {topup} | b0: {b0} | HRFmodel: {HRFmodel} | Subject: {subject} | Session: {session} | Scan: {scan} | Run: {run+1} | Region: {maskName} ')
										if not os.path.exists(os.path.join(runDir, f'featquery/{condition}')):
											os.makedirs(os.path.join(runDir, f'featquery/{condition}'), exist_ok=True)

										outDir = os.path.join(runDir, f'featquery/{condition}', maskName)

										nCopes = len(glob.glob(os.path.join(runDir, 'stats/cope*.nii.gz')))
										# fqCommand = f'featquery 1 {os.path.join(os.getcwd(),runDir)} {nCopes}'
										fqCommand = f'featquery 1 {runDir} {nCopes}'
										for cope in range(nCopes):
											fqCommand += f' stats/cope{cope+1}'
										# fqCommand += f' featquery/{condition}/{maskName} -p -s {os.getcwd()}/{mask}'
										fqCommand += f' featquery/{condition}/{maskName} -p -s {mask}'

										if not os.path.isfile(f'{outDir}/report.txt') or overwrite:
											try:
												shutil.rmtree(outDir)
											except:
												pass
											os.system(fqCommand)
print(nMasksRemoved)
print('FINISHED')



