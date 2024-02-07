'''
makes flood-filled masks
'''

import os
import sys
import glob
import datetime
import shutil
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/home/tonglab/Miao/fMRI/masterScripts')
from splitByHemi import splitByHemi
from makeFloodfillMasks import makeFloodfillMasks

# get scan info from experiment file
from fMRI.analysis.scripts.experiment import experiment

sizes = {'V1': [16, 64, 256],'V2': [4, 16, 64, 256],'V3': [4, 16, 64, 256],'hV4': [4, 16, 64, 256]}
# sizes = {'V1': [16, 64, 256]}
thresh = 2.34
stdMaskDir = '/home/tonglab/Miao/fMRI/masks'

for subject in experiment['scanInfo'].keys():
	for s, session in enumerate(experiment['scanInfo'][subject].keys()):

		sessDir = os.path.join('/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/individual', subject, session)
		scan = 'checkerboard_loc'
		condition = scan + '_secondLevel'
		for topup in ['topUp','noTopUp']:#,'noTopUp'
			for b0 in ['noB0']:
				for HRFmodel in ['doubleGamma']:#, 'singleGamma']:
					estimateDir = f'{sessDir}/masksNative/{condition}/{topup}/{b0}/{HRFmodel}/floodfill/estimates/'
					# localiser activation is based on a second level analysis across runs
					locDir = os.path.join(sessDir, f'functional/{scan}/allRuns/{topup}/{b0}/{HRFmodel}/secondLevel_withSmoothing.gfeat/cope4.feat')
					actMapStd = os.path.join(locDir, 'stats/zstat1.nii.gz')

					# get a first level directory for example func and reg
					runDir = f'{sessDir}/functional/{scan}/run01/outputs/{topup}/{b0}/{HRFmodel}/firstLevel_withSmoothing.feat'

					# single native mask directory since all first levels use the same native reference image
					natMaskDir = os.path.join('/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/individual', subject, session, 'masksNative',condition,topup,b0,HRFmodel)
					os.makedirs(natMaskDir, exist_ok=True)

					# convert activation map into native space
					floodfillDir = os.path.join(natMaskDir, 'floodfill')
					os.makedirs(floodfillDir, exist_ok=True)
					runVol = os.path.join(f'{runDir}/example_func.nii.gz')
					actMapNat = os.path.join(floodfillDir, 'actMap.nii.gz')
					transMat = os.path.join(runDir, 'reg/standard2example_func.mat')
					os.system(f'flirt -in {actMapStd} -ref {runVol} -out {actMapNat} -init {transMat} -applyxfm -interp trilinear')

					reportFile = os.path.join(floodfillDir, 'maskInfo.txt')
					mi = open(reportFile, 'a+')
					mi.write('region,peakX,peakY,peakZ,Nvox(target),Nvox(actual),minZ\n') # write header (actual data is added by floodfillMasks.py)
					mi.close()

					# for ROI plots
					exampleFunc = os.path.join(runDir, 'example_func.nii.gz')
					EFrange = os.popen(f'fslstats {exampleFunc} -R')
					EFmax = float(EFrange.read().split()[1])
					os.makedirs(os.path.join(natMaskDir, 'plots'), exist_ok=True)

					for region in list(sizes.keys()):
						regionSizes = sizes[region]

						# for V1, make copy of mask and split by hemisphere
						if (region == 'V1')|(region == 'V2')|(region == 'V3')|(region == 'hV4'):
							ROImaskIn = sorted(glob.glob(os.path.join(stdMaskDir, f'*/{region}.nii.gz')))[0]
							ROImaskOut = os.path.join(sessDir, f'masksNative/{condition}/{topup}/{b0}/{HRFmodel}/floodfill/estimates/{region}.nii.gz')
							os.makedirs(os.path.join(sessDir, f'masksNative/{condition}/{topup}/{b0}/{HRFmodel}/floodfill/estimates'), exist_ok=True)
							shutil.copyfile(ROImaskIn, ROImaskOut)
							splitByHemi(ROImaskOut)
							os.remove(ROImaskOut)

						for hemi in ['lh','rh']:
							estimateStd = os.path.join(estimateDir, f'{region}_{hemi}.nii.gz')

							if os.path.isfile(estimateStd):

								# convert estimate to native space
								estimateNat = os.path.join(floodfillDir, f'{region}_{hemi}.nii.gz')
								os.system(f'flirt -in {estimateStd} -ref {runVol} -out {estimateNat} -init {transMat} -applyxfm -interp nearestneighbour')

								# mask the native activation by the native ROI estimate
								maskedActivationDir = os.path.join(floodfillDir, 'masked_activations')
								os.makedirs(maskedActivationDir, exist_ok=True)
								maskedActivation = os.path.join(maskedActivationDir, f'{region}_{hemi}.nii.gz')
								os.system(f'fslmaths {actMapNat} -abs {actMapNat}')
								os.system(f'fslmaths {actMapNat} -mul {estimateNat} {maskedActivation}')

								# make distribution plot of t values in masked activation
								actData = nib.load(maskedActivation).get_fdata().flatten()
								actData.sort()

								nVoxels = np.count_nonzero(np.maximum(0,actData)) # reLu
								y = actData[:-(nVoxels+1):-1]
								x = np.arange(1,nVoxels+1)

								plt.plot(x, y)
								plt.xlabel('voxels')
								plt.ylabel('t value')
								plt.title(f'subject: {subject}, session: {s+1}. region: {region}, hemisphere: {hemi}')
								plt.grid(True)
								plt.savefig(os.path.join(maskedActivationDir, f'{region}_{hemi}_tvals.png'))
								plt.show()
								plt.close()

								# get location of peak activation
								coords = os.popen(f'fslstats {maskedActivation} -x')
								coords = coords.read()[:-2]

								# run flood-fill algorithm
								for size in regionSizes:
									maskOutFile = os.path.join(natMaskDir, f'{region}_{hemi}_{size:05}_voxels.nii.gz')
									print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | Top Up: {topup} |  B0: {b0} | HRF model: {HRFmodel} Subject: {subject} | Session: {session} | Region: {region} | Hemi: {hemi} | Size: {size}')
									# makeFloodfillMasks(maskedActivation,size,coords,thresh,maskOutFile,reportFile)
									clustersize, xopt = makeFloodfillMasks(maskedActivation,size,coords,thresh,maskOutFile)

									reportFile = os.path.join(floodfillDir, 'maskInfo.txt')
									mi = open(reportFile, 'a+')
									mi.write(region+'_'+hemi+' '+coords+' '+str(size)+' '+str(clustersize)+' ' +str(xopt)+'\n')  # write header (actual data is added by floodfillMasks.py)
									mi.close()
									# make brain plots showing ROI
									plotFile = os.path.join(natMaskDir, f'plots/{region}_{hemi}_{size:05}_voxels.png')

									activation = os.path.join(natMaskDir, f'floodfill/actMap.nii.gz')
									maxAct = os.popen(f'fslstats {activation} -R')
									maxAct = float(maxAct.read().split()[1])

									maskedActivation = os.path.join(natMaskDir, f'floodfill/masked_activations/{region}_{hemi}.nii.gz')
									coords = os.popen(f'fslstats {maskedActivation} -x')
									coords = coords.read()[:-2]

									fsleyesCommand = f'fsleyes render --outfile {plotFile} --size 3200 600 --scene ortho --autoDisplay -vl {coords} {exampleFunc} -dr 0 {EFmax} {activation} -dr 2.34 {maxAct} -cm red-yellow {maskedActivation} -dr 2.34' \
													 f' {maxAct} -cm blue-lightblue {maskOutFile} -dr 0 1 -cm green'
									os.system(fsleyesCommand)
print('Done.')
