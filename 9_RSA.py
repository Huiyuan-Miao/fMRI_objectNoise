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

overwrite = 0
sizes = {'V1': [16, 64, 256],'V2': [4, 16, 64, 256],'V3': [4, 16, 64, 256],'hV4': [4, 16, 64, 256]}
HojinPlosBioHumanData = loadmat('/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/group/HojinHumanRSA.mat')
averageHumanRSA = {}
HojinPlosBioHumanData['V1'] = np.arctanh(np.mean(HojinPlosBioHumanData['RSA'][:,0,:,:],axis = 0))
HojinPlosBioHumanData['V2'] = np.arctanh(np.mean(HojinPlosBioHumanData['RSA'][:,1,:,:],axis = 0))
HojinPlosBioHumanData['V3'] = np.arctanh(np.mean(HojinPlosBioHumanData['RSA'][:,2,:,:],axis = 0))
HojinPlosBioHumanData['hV4'] = np.arctanh(np.mean(HojinPlosBioHumanData['RSA'][:,3,:,:],axis = 0))
HojinPlosBioCNNData = loadmat('/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/group/HojinCNNRSA.mat')['RSA']
HojinPlosBioNoiseCNNData = loadmat('/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/group/HojinNoiseCNNRSA.mat')['RSA']

def removeDiag(mat):
    t = (1-np.eye(mat.shape[0]))
    t[t == 0] = np.nan
    mat_new = mat * t
    mat_new = mat_new.reshape(-1)
    mat_new = mat_new[~np.isnan(mat_new)]
    return mat_new

sizes = {'V1': [16, 64, 256],'V2': [4, 16, 64, 256],'V3': [4, 16, 64, 256],'hV4': [4, 16, 64, 256]}
for topup, topupString in zip(['topUp'],['_topUp']):#zip(['topUp', 'noTopUp'],['_topUp','']):
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
                            # create a directory for saving GLMsingle outputs
                            outDir = os.path.join(
                                f'/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/individual',
                                subject, session, 'functional', scan, 'GLM_single', topup, b0, HRFmodel,
                                'GLM_single/')
                            for analysisType in ['GLM_single_noHRF_fitting']:#['GLM_single','firstLevel', 'secondLevel',]:
                                if (analysisType == 'firstLevel') | (analysisType == 'firstLevel'):
                                    maskTypes = ['firstLevel','secondLevel']
                                elif  (analysisType == 'GLM_single'):
                                    maskTypes = ['firstLevel','GLM_single']
                                elif  (analysisType == 'GLM_single_noHRF_fitting'):
                                    maskTypes = ['firstLevel']
                                for maskType in maskTypes:
                                    if maskType == 'firstLevel':
                                        maskFolder =  os.path.join('/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/individual', subject, session, 'masksNative','checkerboard_loc_firstLevel',topup,b0,HRFmodel)
                                        maskName = 'checkerboard_loc_firstLevel'
                                    else:
                                        maskFolder =  os.path.join('/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/individual', subject, session, 'masksNative',scan + '_'+maskType,topup,b0,HRFmodel)
                                        maskName = scan + '_'+maskType
                                    RSAFolder = os.path.join('/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/individual', subject, session, 'RSA',scan,'analysis_'+analysisType,'mask_'+maskName,topup,b0,HRFmodel)
                                    os.makedirs(RSAFolder, exist_ok=True)
                                    RSADataFolder = os.path.join(RSAFolder,'betaByRegion')
                                    os.makedirs(RSADataFolder, exist_ok=True)
                                    RSAFigFolder = os.path.join(RSAFolder,'figure')
                                    os.makedirs(RSAFigFolder, exist_ok=True)
                                    RSAMatFolder = os.path.join(RSAFolder,'RSM')
                                    os.makedirs(RSAMatFolder, exist_ok=True)
                                    for region in list(sizes.keys()):
                                        regionSizes = sizes[region]
                                        for size in regionSizes:
                                            dataMats = []
                                            dataName = f'{region}_{size:05}_voxels.npy'
                                            figName = f'{region}_{size:05}_voxels.png'
                                            if (not exists(RSADataFolder+'/'+dataName))|(overwrite):
                                                for hemi in ['lh', 'rh']:
                                                    whichMask = maskFolder + f'/{region}_{hemi}_{size:05}_voxels.nii.gz'
                                                    img = nib.load(whichMask)
                                                    mask = img.get_fdata()
                                                    nVol = np.sum(mask==1)
                                                    nRun = len(experiment['scanInfo'][subject][session]['funcScans'][scan])
                                                    condNames = experiment['design'][scan]['conditions']['imageID']
                                                    if analysisType =='firstLevel':
                                                        dataMat_temp = np.zeros(
                                                            (nVol, len(condNames), nRun))
                                                        for r, run in enumerate(
                                                            experiment['scanInfo'][subject][session]['funcScans'][scan]):
                                                            for i in range(len(condNames)):
                                                                runDir = f'{sessDir}/functional/{scan}/run0{r+1}/outputs/{topup}/{b0}/{HRFmodel}/firstLevel.feat/stats/cope{i+1}.nii.gz'
                                                                img = nib.load(runDir)
                                                                data = img.get_fdata()
                                                                dataMat_temp[:nVol,i,r] = data[mask==1]
                                                                # dataMat_temp[nVol:,i,r] = np.nan
                                                        # dataMats.append(dataMat_temp)
                                                    if analysisType == 'secondLevel':
                                                        dataMat_temp = np.zeros(
                                                            (nVol, len(condNames)))
                                                        for i in range(len(condNames)):
                                                            actMapStd = f'{sessDir}/functional/{scan}/allRuns/{topup}/{b0}/{HRFmodel}/secondLevel.gfeat/cope{i + 1}.feat/stats/cope1.nii.gz'
                                                            actMapNat = f'{sessDir}/functional/{scan}/allRuns/{topup}/{b0}/{HRFmodel}/secondLevel.gfeat/cope{i + 1}.feat/stats/cope1_Nat.nii.gz'
                                                            runDir = f'{sessDir}/functional/{scan}/run01/outputs/{topup}/{b0}/{HRFmodel}/firstLevel.feat'
                                                            runVol = os.path.join(f'{runDir}/example_func.nii.gz')
                                                            transMat = os.path.join(runDir,
                                                                                    'reg/standard2example_func.mat')
                                                            if not exists(actMapNat):
                                                                os.system(
                                                                    f'flirt -in {actMapStd} -ref {runVol} -out {actMapNat} -init {transMat} -applyxfm -interp trilinear')
                                                            img = nib.load(actMapNat)
                                                            data = img.get_fdata()
                                                            dataMat_temp[:nVol, i] = data[mask == 1]
                                                            # dataMat_temp[nVol:, i] = np.nan
                                                        # dataMats.append(dataMat_temp)
                                                    if (analysisType == 'GLM_single')|(analysisType == 'GLM_single_noHRF_fitting'):
                                                        if scan == 'object_noise':
                                                            dataMat_temp = np.zeros(
                                                                (nVol, len(condNames),nRun,3))
                                                            design_reshape = np.zeros((len(designs) *len(condNames),))
                                                            for i in range(len(designs)):
                                                                design = designs[i]
                                                                design_reshape[(i*len(condNames)):((i+1)*len(condNames))]= np.argwhere(design == 1)[:,1]
                                                        elif scan == 'object_noise_mseq':
                                                            dataMat_temp = np.zeros(
                                                                (nVol, len(condNames), 2 * nRun, 3))
                                                            design_reshape = np.zeros((len(designs) * len(condNames)*2,))
                                                            for i in range(len(designs)):
                                                                design = designs[i]
                                                                design_reshape[(i*len(condNames)*2):((i+1)*len(condNames)*2)]= np.argwhere(design == 1)[:,1]
                                                        runDir = f'{sessDir}/functional/{scan}/GLM_single/{topup}/{b0}/{HRFmodel}/{analysisType}/'
                                                        results_glmsingle = dict()
                                                        results_glmsingle['typea'] = np.load(
                                                            join(runDir, 'TYPEA_ONOFF.npy'),
                                                            allow_pickle=True).item()
                                                        results_glmsingle['typeb'] = np.load(
                                                            join(runDir, 'TYPEB_FITHRF.npy'),
                                                            allow_pickle=True).item()
                                                        results_glmsingle['typec'] = np.load(
                                                            join(runDir, 'TYPEC_FITHRF_GLMDENOISE.npy'),
                                                            allow_pickle=True).item()
                                                        results_glmsingle['typed'] = np.load(
                                                            join(runDir, 'TYPED_FITHRF_GLMDENOISE_RR.npy'),
                                                            allow_pickle=True).item()
                                                        for i in range(3):
                                                            if i == 0:
                                                                results_glmsingle_sub = results_glmsingle['typeb']['betasmd']
                                                            elif i == 1:
                                                                results_glmsingle_sub = results_glmsingle['typec']['betasmd']
                                                            elif i == 2:
                                                                results_glmsingle_sub = results_glmsingle['typed']['betasmd']
                                                            count1 = np.zeros((len(condNames),))
                                                            for j in range(results_glmsingle_sub.shape[3]):
                                                                data = results_glmsingle_sub[:,:,:,j]
                                                                w = int(design_reshape[j])
                                                                dataMat_temp[:nVol, w, int(count1[w]),i] = data[mask == 1]
                                                                # dataMat_temp[nVol:, w, int(count1[w]),i] = np.nan
                                                                count1[w]+=1
                                                    dataMats.append(dataMat_temp)
                                                print(RSADataFolder+'/'+dataName)
                                                np.save(RSADataFolder+'/'+dataName, np.concatenate((dataMats[0],dataMats[1]),axis =0))
                                                dataMat = np.concatenate((dataMats[0],dataMats[1]),axis =0)
                                            else:
                                                dataMat = np.load(RSADataFolder+'/'+dataName,allow_pickle=True)
                                                print(dataMat.shape)
                                                if dataMat.shape[0] > 512:
                                                    print(RSADataFolder+'/'+dataName)

                                            '''correlation to the entire mat'''
                                            if (analysisType == 'firstLevel'):
                                                dataMat_reshape = dataMat[:,:32,:].reshape(dataMat.shape[0],-1)
                                                c = np.corrcoef(dataMat_reshape.T) #*(1-np.eye(dataMat_reshape.shape[1]))
                                                plt.imshow(c,vmin= 0,vmax = 1);plt.colorbar(ticks = [0,0.25,0.5,0.75,1])
                                                plt.savefig(RSAFigFolder+'/'+figName);plt.close()
                                                np.save(RSAMatFolder + '/' + dataName, c)
                                                dataMat_reshape = np.mean(dataMat[:,:32,:],axis = 2)
                                                c = np.corrcoef(
                                                    dataMat_reshape.T)  # *(1-np.eye(dataMat_reshape.shape[1]))
                                                plt.figure(figsize=(5, 8))
                                                plt.subplot(2, 1, 1)
                                                plt.imshow(c, vmin=0, vmax=1);
                                                plt.colorbar(ticks=[0, 0.25, 0.5, 0.75, 1])
                                                c = np.arctanh(c)
                                                if scan == 'object_noise_mseq':
                                                    cHojin = HojinPlosBioHumanData[region][:c.shape[0], :c.shape[0]]
                                                else:
                                                    cHojin = HojinPlosBioHumanData[region][:32, :32]
                                                s = np.corrcoef(removeDiag(c), removeDiag(cHojin))[0, 1]
                                                plt.title('corr to Hojin is ' + str(s))
                                                ax2 = plt.subplot(2, 1, 2)
                                                sCNN = np.zeros((19,))
                                                sNoiseCNN = np.zeros((19,))
                                                for i in range(19):
                                                    if scan == 'object_noise_mseq':
                                                        cCNNHojin = HojinPlosBioCNNData[i, :c.shape[0], :c.shape[0]]
                                                        cNoiseCNNHojin = HojinPlosBioNoiseCNNData[i, :c.shape[0],
                                                                         :c.shape[0]]
                                                    else:
                                                        cCNNHojin = HojinPlosBioCNNData[i, :32, :32]
                                                        cNoiseCNNHojin = HojinPlosBioNoiseCNNData[i, :32, :32]
                                                    sCNN[i] = np.corrcoef(removeDiag(c), removeDiag(cCNNHojin))[
                                                        0, 1]
                                                    sNoiseCNN[i] = \
                                                    np.corrcoef(removeDiag(c), removeDiag(cNoiseCNNHojin))[0, 1]
                                                ax2.plot(np.arange(19), sCNN, color='r', label='CNN')
                                                ax2.plot(np.arange(19), sNoiseCNN, color='b', label='NoiseCNN')
                                                ax2.set_xlabel('Layer')
                                                ax2.set_ylabel('Correlation')
                                                plt.title(region)
                                                ax2.legend()
                                                plt.savefig(RSAFigFolder + '/average_beta_' + figName);
                                                plt.close()
                                                np.save(RSAMatFolder + '/average_beta_' + dataName, c)
                                            elif (analysisType == 'secondLevel'):
                                                dataMat_reshape = dataMat[:,:32].reshape(dataMat.shape[0],-1)
                                                c = np.corrcoef(dataMat_reshape.T) #*(1-np.eye(dataMat_reshape.shape[1]))
                                                plt.figure(figsize=(5, 8))
                                                plt.subplot(2, 1, 1)
                                                plt.imshow(c, vmin=0, vmax=1);
                                                plt.colorbar(ticks=[0, 0.25, 0.5, 0.75, 1])
                                                c = np.arctanh(c)
                                                if scan == 'object_noise_mseq':
                                                    cHojin = HojinPlosBioHumanData[region][:c.shape[0], :c.shape[0]]
                                                else:
                                                    cHojin = HojinPlosBioHumanData[region][:32,:32]
                                                s = np.corrcoef(removeDiag(c), removeDiag(cHojin))[0, 1]
                                                plt.title('corr to Hojin is ' + str(s))
                                                ax2 = plt.subplot(2, 1, 2)
                                                sCNN = np.zeros((19,))
                                                sNoiseCNN = np.zeros((19,))
                                                for i in range(19):
                                                    if scan == 'object_noise_mseq':
                                                        cCNNHojin = HojinPlosBioCNNData[i, :c.shape[0], :c.shape[0]]
                                                        cNoiseCNNHojin = HojinPlosBioNoiseCNNData[i, :c.shape[0],
                                                                         :c.shape[0]]
                                                    else:
                                                        cCNNHojin = HojinPlosBioCNNData[i, :32, :32]
                                                        cNoiseCNNHojin = HojinPlosBioNoiseCNNData[i, :32, :32]
                                                    sCNN[i] = np.corrcoef(removeDiag(c), removeDiag(cCNNHojin))[
                                                        0, 1]
                                                    sNoiseCNN[i] = \
                                                    np.corrcoef(removeDiag(c), removeDiag(cNoiseCNNHojin))[0, 1]
                                                ax2.plot(np.arange(19), sCNN, color='r', label='CNN')
                                                ax2.plot(np.arange(19), sNoiseCNN, color='b', label='NoiseCNN')
                                                ax2.set_xlabel('Layer')
                                                ax2.set_ylabel('Correlation')
                                                plt.title(region)
                                                ax2.legend()
                                                plt.savefig(RSAFigFolder + '/' + figName);
                                                plt.close()
                                                np.save(RSAMatFolder + '/' + dataName, c)
                                            else:
                                                dataMat_reshape = dataMat[:,:32,:,0].reshape(dataMat.shape[0], -1)
                                                c = np.corrcoef(
                                                    dataMat_reshape.T)  # *(1-np.eye(dataMat_reshape.shape[1]))
                                                plt.imshow(c,vmin= 0,vmax = 1);plt.colorbar(ticks = [0,0.25,0.5,0.75,1])
                                                plt.savefig(RSAFigFolder + '/model_typeB_' + figName);plt.close()
                                                np.save(RSAMatFolder + '/model_typeB_' + dataName, c)
                                                dataMat_reshape = dataMat[:, :32, :, 1].reshape(dataMat.shape[0], -1)
                                                c = np.corrcoef(
                                                    dataMat_reshape.T)  # *(1-np.eye(dataMat_reshape.shape[1]))
                                                plt.imshow(c,vmin= 0,vmax = 1);plt.colorbar(ticks = [0,0.25,0.5,0.75,1])
                                                plt.savefig(RSAFigFolder + '/model_typeC_' + figName);plt.close()
                                                np.save(RSAMatFolder + '/model_typeC_' + dataName, c)
                                                dataMat_reshape = dataMat[:, :32, :, 2].reshape(dataMat.shape[0], -1)
                                                c = np.corrcoef(
                                                    dataMat_reshape.T)  # *(1-np.eye(dataMat_reshape.shape[1]))
                                                plt.imshow(c,vmin= 0,vmax = 1);plt.colorbar(ticks = [0,0.25,0.5,0.75,1])
                                                plt.savefig(RSAFigFolder + '/model_typeD_' + figName);plt.close()
                                                np.save(RSAMatFolder + '/model_typeD_' + dataName, c)

                                                dataMat_reshape = np.mean(dataMat[:,:32,:,0], axis=2)
                                                c = np.corrcoef(
                                                    dataMat_reshape.T)  # *(1-np.eye(dataMat_reshape.shape[1]))
                                                plt.figure(figsize=(5, 8))
                                                plt.subplot(2, 1, 1)
                                                plt.imshow(c, vmin=0, vmax=1);
                                                plt.colorbar(ticks=[0, 0.25, 0.5, 0.75, 1])
                                                c = np.arctanh(c)
                                                if scan == 'object_noise_mseq':
                                                    cHojin = HojinPlosBioHumanData[region][:c.shape[0], :c.shape[0]]
                                                else:
                                                    cHojin = HojinPlosBioHumanData[region][:32,:32]
                                                s = np.corrcoef(removeDiag(c), removeDiag(cHojin))[0, 1]
                                                plt.title('corr to Hojin is ' + str(s))
                                                ax2 = plt.subplot(2, 1, 2)
                                                sCNN = np.zeros((19,))
                                                sNoiseCNN = np.zeros((19,))
                                                for i in range(19):
                                                    if scan == 'object_noise_mseq':
                                                        cCNNHojin = HojinPlosBioCNNData[i, :c.shape[0], :c.shape[0]]
                                                        cNoiseCNNHojin = HojinPlosBioNoiseCNNData[i, :c.shape[0],
                                                                         :c.shape[0]]
                                                    else:
                                                        cCNNHojin = HojinPlosBioCNNData[i, :32, :32]
                                                        cNoiseCNNHojin = HojinPlosBioNoiseCNNData[i, :32, :32]
                                                    sCNN[i] = np.corrcoef(removeDiag(c), removeDiag(cCNNHojin))[0, 1]
                                                    sNoiseCNN[i] = \
                                                    np.corrcoef(removeDiag(c), removeDiag(cNoiseCNNHojin))[0, 1]
                                                ax2.plot(np.arange(19), sCNN, color='r', label='CNN')
                                                ax2.plot(np.arange(19), sNoiseCNN, color='b', label='NoiseCNN')
                                                ax2.set_xlabel('Layer')
                                                ax2.set_ylabel('Correlation')
                                                plt.title(region)
                                                ax2.legend()
                                                plt.savefig(RSAFigFolder + '/average_beta_model_typeB_' + figName);
                                                plt.close()
                                                np.save(RSAMatFolder + '/average_beta_model_typeB_' + dataName, c)
                                                dataMat_reshape = np.mean(dataMat[:, :32, :, 1], axis=2)
                                                c = np.corrcoef(
                                                    dataMat_reshape.T)  # *(1-np.eye(dataMat_reshape.shape[1]))
                                                plt.figure(figsize=(5, 8))
                                                plt.subplot(2, 1, 1)
                                                plt.imshow(c, vmin=0, vmax=1);
                                                plt.colorbar(ticks=[0, 0.25, 0.5, 0.75, 1])
                                                c = np.arctanh(c)
                                                if scan == 'object_noise_mseq':
                                                    cHojin = HojinPlosBioHumanData[region][:c.shape[0], :c.shape[0]]
                                                else:
                                                    cHojin = HojinPlosBioHumanData[region][:32,:32]
                                                s = np.corrcoef(removeDiag(c), removeDiag(cHojin))[0, 1]
                                                plt.title('corr to Hojin is ' + str(s))
                                                ax2 = plt.subplot(2, 1, 2)
                                                sCNN = np.zeros((19,))
                                                sNoiseCNN = np.zeros((19,))
                                                for i in range(19):
                                                    if scan == 'object_noise_mseq':
                                                        cCNNHojin = HojinPlosBioCNNData[i, :c.shape[0], :c.shape[0]]
                                                        cNoiseCNNHojin = HojinPlosBioNoiseCNNData[i, :c.shape[0],
                                                                         :c.shape[0]]
                                                    else:
                                                        cCNNHojin = HojinPlosBioCNNData[i, :32, :32]
                                                        cNoiseCNNHojin = HojinPlosBioNoiseCNNData[i, :32, :32]
                                                    sCNN[i] = np.corrcoef(removeDiag(c), removeDiag(cCNNHojin))[0, 1]
                                                    sNoiseCNN[i] = \
                                                    np.corrcoef(removeDiag(c), removeDiag(cNoiseCNNHojin))[0, 1]
                                                ax2.plot(np.arange(19), sCNN, color='r', label='CNN')
                                                ax2.plot(np.arange(19), sNoiseCNN, color='b', label='NoiseCNN')
                                                ax2.set_xlabel('Layer')
                                                ax2.set_ylabel('Correlation')
                                                plt.title(region)
                                                ax2.legend()
                                                plt.savefig(RSAFigFolder + '/average_beta_model_typeC_' + figName);
                                                plt.close()
                                                np.save(RSAMatFolder + '/average_beta_model_typeC_' + dataName, c)
                                                dataMat_reshape = np.mean(dataMat[:, :32, :, 2], axis=2)
                                                c = np.corrcoef(
                                                    dataMat_reshape.T)  # *(1-np.eye(dataMat_reshape.shape[1]))
                                                plt.figure(figsize=(5, 8))
                                                plt.subplot(2, 1, 1)
                                                plt.imshow(c, vmin=0, vmax=1);
                                                plt.colorbar(ticks=[0, 0.25, 0.5, 0.75, 1])
                                                c = np.arctanh(c)
                                                if scan == 'object_noise_mseq':
                                                    cHojin = HojinPlosBioHumanData[region][:c.shape[0], :c.shape[0]]
                                                else:
                                                    cHojin = HojinPlosBioHumanData[region][:32,:32]
                                                s = np.corrcoef(removeDiag(c), removeDiag(cHojin))[0, 1]
                                                plt.title('corr to Hojin is ' + str(s))
                                                ax2 = plt.subplot(2, 1, 2)
                                                sCNN = np.zeros((19,))
                                                sNoiseCNN = np.zeros((19,))
                                                for i in range(19):
                                                    if scan == 'object_noise_mseq':
                                                        cCNNHojin = HojinPlosBioCNNData[i, :c.shape[0], :c.shape[0]]
                                                        cNoiseCNNHojin = HojinPlosBioNoiseCNNData[i, :c.shape[0],
                                                                         :c.shape[0]]
                                                    else:
                                                        cCNNHojin = HojinPlosBioCNNData[i, :32, :32]
                                                        cNoiseCNNHojin = HojinPlosBioNoiseCNNData[i, :32, :32]
                                                    sCNN[i] = np.corrcoef(removeDiag(c), removeDiag(cCNNHojin))[0, 1]
                                                    sNoiseCNN[i] = \
                                                    np.corrcoef(removeDiag(c), removeDiag(cNoiseCNNHojin))[0, 1]
                                                ax2.plot(np.arange(19), sCNN, color='r', label='CNN')
                                                ax2.plot(np.arange(19), sNoiseCNN, color='b', label='NoiseCNN')
                                                ax2.set_xlabel('Layer')
                                                ax2.set_ylabel('Correlation')
                                                plt.title(region)
                                                ax2.legend()
                                                plt.savefig(RSAFigFolder + '/average_beta_model_typeD_' + figName);
                                                plt.close()
                                                np.save(RSAMatFolder + '/average_beta_model_typeD_' + dataName, c)


                                                # plt.show()















