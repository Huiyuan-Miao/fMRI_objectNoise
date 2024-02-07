import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats
import scipy.io as sio
import matplotlib.pyplot as plt
import nibabel as nib

import os
from os.path import join, exists, split
import sys
import time
import urllib.request
import copy
import warnings
from tqdm import tqdm
from pprint import pprint
from scipy.io import loadmat
import glob
import nibabel as nib
warnings.filterwarnings('ignore')

import sys
sys.path.append('/home/tonglab/Miao/fMRI/GLMsingle-main/')
import glmsingle
from glmsingle.glmsingle import GLM_single
import matplotlib.pyplot as plt
from fMRI.analysis.scripts.experiment import experiment

# first level analysis
for topup, topupString in zip(['topUp', 'noTopUp'],['_topUp','']):
    # for b0, b0String in zip(['noB0', 'b0'],['', '_b0']):
    for b0, b0String in zip(['noB0'],['']):
        for HRFmodel in ['doubleGamma']:#,'singleGamma']:
            for subject in experiment['scanInfo'].keys():
                for session in experiment['scanInfo'][subject].keys():
                    sessDir = os.path.join(
                        f'/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/individual', subject,
                        session)
                    for scan in experiment['design'].keys():
                        if (scan != 'resting_state') & (scan != 'FHO_loc') & (scan != 'checkerboard_loc'): #& (scan != 'object_noise_mseq')& (scan != 'object_noise')
                            designs = []
                            datas = []
                            for r, run in enumerate(experiment['scanInfo'][subject][session]['funcScans'][scan]):
                                if scan == 'object_noise':
                                    condNames = experiment['design'][scan]['conditions']['imageID']
                                    dynamics = experiment['design'][scan]['params']['nDynamics']
                                    TR = experiment['design'][scan]['params']['TR']
                                    stimdur = experiment['design'][scan]['params']['imageDuration']
                                    # design = np.zeros((dynamics,len(condNames)+1))
                                    design = np.zeros((dynamics,len(condNames)))
                                    eventDir = os.path.join(sessDir, 'events', scan, f'3column/run{r + 1:02}')
                                    os.makedirs(eventDir, exist_ok=True)
                                    logFile = glob.glob(os.path.join(sessDir, 'events', scan, f'logFiles/*_rn{r + 1}_*'))[0]
                                    logData = loadmat(logFile)
                                    eventData = logData['onsetTime'][0]
                                    for c, cond in enumerate(condNames):
                                        design[int(eventData[c]/2),c] = 1
                                    # design[:,-1] = np.sum(design[:,:len(condNames)],axis = 1)
                                    designs.append(design)
                                    inputName = 'filtered_func_data.nii.gz'
                                    # outName = 'filtered_func_data_nativeAnatomical.nii.gz'
                                    # inputDir = os.path.join(
                                    #     f'/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/individual',
                                    #     subject, session, 'functional', scan, f'run{r + 1:02}/outputs/',topup,b0,HRFmodel,'firstLevel.feat', inputName)
                                    inputDir = os.path.join(
                                        f'/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/individual',
                                        subject, session, 'functional', scan, f'run{r + 1:02}/outputs/',topup,b0,'preprocessing.feat', inputName)
                                    # outputDir = os.path.join(
                                    #     f'/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/individual',
                                    #     subject, session, 'functional', scan, f'run{r + 1:02}/outputs/',topup,b0,HRFmodel,'firstLevel.feat', outName)
                                    # transformMat = os.path.join(
                                    #     f'/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/individual',
                                    #     subject, session, 'functional', scan, f'run{r + 1:02}/outputs/',topup,b0,HRFmodel,'firstLevel.feat/reg/example_func2highres.mat')
                                    img = nib.load(inputDir)
                                    data = img.get_fdata()
                                    datas.append(data)
                                    print(f'There are {len(datas)} runs in total\n')
                                    print(f'N = {datas[r].shape[3]} TRs per run\n')
                                    print(f'The dimensions of the data for each run are: {datas[0].shape}\n')
                                    print(
                                        f'XYZ dimensionality is: {datas[r].shape[:3]} (one slice only in this example)\n')
                                    print(f'Numeric precision of data is: {type(datas[r][0, 0, 0, 0])}\n')
                                if scan == 'object_noise_mseq':
                                    condNames = experiment['design'][scan]['conditions']['imageID']
                                    dynamics = experiment['design'][scan]['params']['nDynamics']
                                    TR = experiment['design'][scan]['params']['TR']
                                    stimdur = experiment['design'][scan]['params']['imageDuration']
                                    # design = np.zeros((dynamics,len(condNames)+1))
                                    design = np.zeros((dynamics,len(condNames)))
                                    eventDir = os.path.join(sessDir, 'events', scan, f'3column/run{r + 1:02}')
                                    os.makedirs(eventDir, exist_ok=True)
                                    logFile = glob.glob(os.path.join(sessDir, 'events', scan, f'logFiles/*_rn{r + 1}_*'))[0]
                                    logData = loadmat(logFile)
                                    eventData = logData['onsetTime'][0]
                                    for c, cond in enumerate(condNames):
                                        design[int(eventData[c*2]/2),c] = 1
                                        design[int(eventData[c*2+1]/2), c] = 1
                                    # design[:,-1] = np.sum(design[:,:len(condNames)],axis = 1)
                                    designs.append(design)
                                    inputName = 'filtered_func_data.nii.gz'
                                    inputDir = os.path.join(
                                        f'/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/individual',
                                        subject, session, 'functional', scan, f'run{r + 1:02}/outputs/', topup, b0,
                                        HRFmodel, 'firstLevel.feat', inputName)
                                    img = nib.load(inputDir)
                                    data = img.get_fdata()
                                    datas.append(data)
                                    print(f'There are {len(datas)} runs in total\n')
                                    print(f'N = {datas[r].shape[3]} TRs per run\n')
                                    print(f'The dimensions of the data for each run are: {datas[0].shape}\n')
                                    print(
                                        f'XYZ dimensionality is: {datas[r].shape[:3]} (one slice only in this example)\n')
                                    print(f'Numeric precision of data is: {type(datas[r][0, 0, 0, 0])}\n')
                            # create a directory for saving GLMsingle outputs
                            outDir = os.path.join(
                                f'/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/individual',
                                subject, session, 'functional', scan, 'GLM_single', topup, b0, HRFmodel,
                                'GLM_single/')
                            os.makedirs(os.path.dirname(outDir), exist_ok=True)

                            opt = dict()

                            # set important fields for completeness (but these would be enabled by default)
                            opt['wantlibrary'] = 1
                            opt['wantglmdenoise'] = 1
                            opt['wantfracridge'] = 1

                            # for the purpose of this example we will keep the relevant outputs in memory
                            # and also save them to the disk
                            opt['wantfileoutputs'] = [1, 1, 1, 1]
                            opt['wantmemoryoutputs'] = [1, 1, 1, 1]

                            # running python GLMsingle involves creating a GLM_single object
                            # and then running the procedure using the .fit() routine
                            glmsingle_obj = GLM_single(opt)

                            # visualize all the hyperparameters
                            pprint(glmsingle_obj.params)

                            start_time = time.time()

                            if not exists(outDir+'TYPED_FITHRF_GLMDENOISE_RR.npy'):

                                print(f'running GLMsingle...')

                                # run GLMsingle
                                results_glmsingle = glmsingle_obj.fit(
                                    designs,
                                    datas,
                                    stimdur,
                                    TR,
                                    outputdir=outDir,
                                figuredir = outDir+'figure')

                                # we assign outputs of GLMsingle to the "results_glmsingle" variable.
                                # note that results_glmsingle['typea'] contains GLM estimates from an ONOFF model,
                                # where all images are treated as the same condition. these estimates
                                # could be potentially used to find cortical areas that respond to
                                # visual stimuli. we want to compare beta weights between conditions
                                # therefore we are not going to include the ONOFF betas in any analyses of
                                # voxel reliability

                            else:
                                print(f'loading existing GLMsingle outputs from directory:\n\t{outDir}')

                                # load existing file outputs if they exist
                                results_glmsingle = dict()
                                results_glmsingle['typea'] = np.load(join(outDir, 'TYPEA_ONOFF.npy'),
                                                                     allow_pickle=True).item()
                                results_glmsingle['typeb'] = np.load(join(outDir, 'TYPEB_FITHRF.npy'),
                                                                     allow_pickle=True).item()
                                results_glmsingle['typec'] = np.load(
                                    join(outDir, 'TYPEC_FITHRF_GLMDENOISE.npy'), allow_pickle=True).item()
                                results_glmsingle['typed'] = np.load(
                                    join(outDir, 'TYPED_FITHRF_GLMDENOISE_RR.npy'),
                                    allow_pickle=True).item()
                                affine_nifti = img.affine
                                img_nifti= results_glmsingle['typea']['onoffR2']
                                img_nifti[np.isnan(img_nifti)] = 0
                                create_nifti = nib.Nifti1Image(img_nifti,affine_nifti)
                                nib.save(create_nifti,os.path.join(outDir,'onoffR2.nii.gz'))
                            elapsed_time = time.time() - start_time

                            print(
                                '\telapsed time: ',
                                f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}'
                            )




# hrflibrary_norm = np.zeros(np.shape(params['hrflibrary']))
# plt.figure(figsize=(3, 16.8))
# for i in range(20):
#     hrflibrary_norm[:,i] = normalisemax(
#                 params['hrflibrary'][:,i],
#                 dim='global')
#     plt.subplot(21,1,i+1)
#     plt.plot(params['hrflibrary'][:,i])
# plt.subplot(21,1,21)
# plt.plot(params['hrftoassume'],label ='default')
# # plt.plot(np.mean(hrflibrary_norm,axis =1),label ='mean library') # v3
# plt.plot(normalisemax(np.mean(params['hrflibrary'],axis =1),dim='global'),label ='mean library') # v4
# plt.legend(fontsize=6)
# plt.savefig('GLM_single_prfLibrary4.png')

