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


def removeDiag(mat):
    t = (1-np.eye(mat.shape[0]))
    t[t == 0] = np.nan
    mat_new = mat * t
    mat_new = mat_new.reshape(-1)
    mat_new = mat_new[~np.isnan(mat_new)]
    return mat_new
def plotHist(mat1,mat2,leg1,leg2):
    plt.hist(mat1, bins=np.arange(-0.5, 20.5, 1), alpha=0.5, label=leg1)
    plt.hist(mat2, bins=np.arange(-0.5, 20.5, 1), alpha=0.5, label=leg2)
    plt.title('average difference '+ str(np.round(np.mean(np.abs(mat1-mat2)),2)))
    plt.xlabel('hrf id (---> hrf shift to right)')
    plt.ylabel('counts')
    plt.xlim((-0.5,20))
    plt.ylim((0,len(mat1)))
    plt.legend()


def plotCorr(mat1,mat2,leg1,leg2):
    # plt.scatter(mat1,mat2,alpha= 0.1)
    plt.scatter(mat1 + np.random.rand(np.size(mat1)) * np.random.choice([-1,1],np.size(mat1)) / 2, mat2 + np.random.rand(np.size(mat1)) * np.random.choice([-1,1],np.size(mat1)) / 2, alpha=0.3)
    plt.title('corr '+ str(np.round(np.corrcoef(mat1,mat2)[0,1],2)))
    plt.xlabel(leg1)
    plt.ylabel(leg2)
    plt.xlim((-0.5,19.5))
    plt.ylim((-0.5,19.5))
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
sizes = {'V1': [16, 64, 256],'V2': [4, 16, 64, 256],'V3': [4, 16, 64, 256],'hV4': [4, 16, 64, 256]}
for topup, topupString in zip(['noTopUp'],['']):#zip(['topUp', 'noTopUp'],['_topUp','']):
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
                            for analysisType in ['GLM_single']:
                                if analysisType != 'GLM_single':
                                    maskTypes = ['firstLevel']
                                else:
                                    maskTypes = ['firstLevel']
                                for maskType in maskTypes:
                                    if maskType == 'firstLevel':
                                        maskFolder =  os.path.join('/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/individual', subject, session, 'masksNative','checkerboard_loc_firstLevel',topup,b0,HRFmodel)
                                        maskName = 'checkerboard_loc_firstLevel'
                                    else:
                                        maskFolder =  os.path.join('/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/individual', subject, session, 'masksNative',scan + '_'+maskType,topup,b0,HRFmodel)
                                        maskName = scan + '_'+maskType
                                    HRFConsistencyDir = os.path.join(
                                        f'/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/individual',
                                        subject, session, 'functional', scan, 'GLM_single', topup, b0, HRFmodel,
                                        'HRFConsistency')
                                    os.makedirs(HRFConsistencyDir, exist_ok=True)

                                    for region in list(sizes.keys()):
                                        regionSizes = sizes[region]
                                        for size in regionSizes:
                                            dataMats = {}
                                            dataName = f'{region}_{size:05}_voxels.npy'
                                            figName = f'{region}_{size:05}_voxels.png'
                                            if (not exists(HRFConsistencyDir+'/'+dataName))|(overwrite):
                                                for hemi in ['lh', 'rh']:
                                                    whichMask = maskFolder + f'/{region}_{hemi}_{size:05}_voxels.nii.gz'
                                                    img = nib.load(whichMask)
                                                    mask = img.get_fdata()
                                                    nVol = np.sum(mask==1)
                                                    nRun = len(experiment['scanInfo'][subject][session]['funcScans'][scan])
                                                    condNames = experiment['design'][scan]['conditions']['imageID']
                                                    if analysisType == 'GLM_single':
                                                        if len(designs) == 3:
                                                            combs = [[1], [2], [3], [1, 2], [1, 3], [2, 3]]
                                                        elif len(designs) == 4:
                                                            combs = [[1],[2],[3],[4],[1, 4], [2, 4], [3, 4], [1, 2], [1, 3], [2, 3]]
                                                        elif len(designs) == 5:
                                                            # combs = [[1, 4], [2, 4], [3, 4], [1, 2], [1, 3], [2, 3],
                                                            #          [1, 5], [2, 5], [3, 5],
                                                            #          [4, 5],
                                                            #          [1, 2, 3], [1, 2, 4], [1, 2, 5], [1, 3, 4],
                                                            #          [1, 3, 5], [1, 4, 5],
                                                            #          [2, 3, 4], [2, 3, 5], [2, 4, 5], [3, 4, 5]]
                                                            combs = [[1, 4], [2, 4], [3, 4], [1, 2], [1, 3], [2, 3],
                                                                     [1, 5], [2, 5], [3, 5],
                                                                     [4, 5]]
                                                        for comb in combs:
                                                            s = ''
                                                            for i in range(len(comb)):
                                                                s = s + str(comb[i])
                                                            outDir = os.path.join(
                                                                f'/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/individual',
                                                                subject, session, 'functional', scan, 'GLM_single',
                                                                topup, b0, HRFmodel,
                                                                'GLM_single' + s + '/')
                                                            results_glmsingle = dict()
                                                            results_glmsingle['typeb'] = np.load(
                                                                join(outDir, 'TYPEB_FITHRF.npy'),
                                                                allow_pickle=True).item()
                                                            dataMat = np.zeros((nVol,))
                                                            results_glmsingle_sub = results_glmsingle['typeb']['HRFindex']
                                                            dataMat = results_glmsingle_sub[mask == 1]
                                                            if hemi == 'lh':
                                                                dataMats[s] = dataMat
                                                            else:
                                                                dataMats[s] = np.concatenate((dataMats[s],dataMat),axis =0)
                                                print(HRFConsistencyDir+'/'+dataName)
                                                np.save(HRFConsistencyDir+'/'+dataName, dataMats)
                                            else:
                                                dataMats = np.load(HRFConsistencyDir+'/'+dataName,allow_pickle=True).item()
                                                HRFConsistencyFigDir = os.path.join(
                                                    f'/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/individual',
                                                    subject, session, 'functional', scan, 'GLM_single', topup, b0,
                                                    HRFmodel,
                                                    'HRFConsistency/figure')
                                                os.makedirs(HRFConsistencyFigDir, exist_ok=True)
                                                if len(dataMats) == 6:
                                                    if '1' in dataMats.keys():
                                                        plt.figure(figsize=(12, 4))
                                                        # plt.subplot(131)
                                                        # plotHist(dataMats['14'], dataMats['23'], 'run 1,4', 'run 2,3')
                                                        # plt.subplot(132)
                                                        # plotHist(dataMats['13'], dataMats['24'], 'run 1,3', 'run 2,4')
                                                        # plt.subplot(133)
                                                        # plotHist(dataMats['12'], dataMats['34'], 'run 1,2', 'run 3,4')
                                                        plt.subplot(131)
                                                        plotHist(dataMats['1'], dataMats['23'], 'run 1', 'run 2,3')
                                                        plt.subplot(132)
                                                        plotHist(dataMats['2'], dataMats['13'], 'run 2', 'run 1,3')
                                                        plt.subplot(133)
                                                        plotHist(dataMats['3'], dataMats['12'], 'run 3', 'run 1,2')
                                                        # plt.show()
                                                        plt.savefig(HRFConsistencyFigDir + '/' + figName);
                                                        plt.close()

                                                        plt.figure(figsize=(14, 4))
                                                        # plt.subplot(131)
                                                        # plotHist(dataMats['14'], dataMats['23'], 'run 1,4', 'run 2,3')
                                                        # plt.subplot(132)
                                                        # plotHist(dataMats['13'], dataMats['24'], 'run 1,3', 'run 2,4')
                                                        # plt.subplot(133)
                                                        # plotHist(dataMats['12'], dataMats['34'], 'run 1,2', 'run 3,4')
                                                        plt.subplot(131)
                                                        plotCorr(dataMats['1'], dataMats['23'], 'run 1', 'run 2,3')
                                                        plt.subplot(132)
                                                        plotCorr(dataMats['2'], dataMats['13'], 'run 2', 'run 1,3')
                                                        plt.subplot(133)
                                                        plotCorr(dataMats['3'], dataMats['12'], 'run 3', 'run 1,2')
                                                        # plt.show()
                                                        plt.savefig(HRFConsistencyFigDir + '/corr_' + figName);
                                                        plt.close()
                                                elif len(dataMats) ==10:
                                                        plt.figure(figsize=(12, 4))
                                                        plt.subplot(131)
                                                        plotHist(dataMats['14'], dataMats['23'], 'run 14', 'run 2,3')
                                                        plt.subplot(132)
                                                        plotHist(dataMats['24'], dataMats['13'], 'run 24', 'run 1,3')
                                                        plt.subplot(133)
                                                        plotHist(dataMats['34'], dataMats['12'], 'run 34', 'run 1,2')
                                                        # plt.show()
                                                        plt.savefig(HRFConsistencyFigDir + '/' + figName);
                                                        plt.close()

                                                        plt.figure(figsize=(14, 4))
                                                        plt.subplot(131)
                                                        plotCorr(dataMats['14'], dataMats['23'], 'run 14', 'run 2,3')
                                                        plt.subplot(132)
                                                        plotCorr(dataMats['24'], dataMats['13'], 'run 24', 'run 1,3')
                                                        plt.subplot(133)
                                                        plotCorr(dataMats['34'], dataMats['12'], 'run 34', 'run 1,2')
                                                        # plt.show()
                                                        plt.savefig(HRFConsistencyFigDir + '/corr_' + figName);
                                                        plt.close()
                                                elif len(dataMats) == 20:
                                                    plt.figure(figsize=(16, 9))
                                                    count = 1
                                                    for i in range(1,6):
                                                        for j in range(i+1, 6):
                                                            id1 = str(i) + str(j)
                                                            leg1 = 'run ' + str(i) + ',' + str(j)
                                                            id2 = ''
                                                            leg2 = 'run '
                                                            for k in range(1,6):
                                                                if (k != i) & (k != j):
                                                                    id2 += str(k)
                                                                    if len(id2) < 3:
                                                                        leg2 += str(k)
                                                                        leg2 += ','
                                                                    else:
                                                                        leg2 += str(k)
                                                            plt.subplot(2,5,count)
                                                            plotHist(dataMats[id1],dataMats[id2],leg1,leg2)
                                                            count+=1
                                                    # plt.show()
                                                    plt.savefig(HRFConsistencyFigDir + '/' + figName);
                                                    plt.close()

                                                    plt.figure(figsize=(24, 9))
                                                    count = 1
                                                    for i in range(1, 6):
                                                        for j in range(i + 1, 6):
                                                            id1 = str(i) + str(j)
                                                            leg1 = 'run ' + str(i) + ',' + str(j)
                                                            id2 = ''
                                                            leg2 = 'run '
                                                            for k in range(1, 6):
                                                                if (k != i) & (k != j):
                                                                    id2 += str(k)
                                                                    if len(id2) < 3:
                                                                        leg2 += str(k)
                                                                        leg2 += ','
                                                                    else:
                                                                        leg2 += str(k)
                                                            plt.subplot(2, 5, count)
                                                            plotCorr(dataMats[id1], dataMats[id2], leg1, leg2)
                                                            count += 1
                                                    # plt.show()
                                                    plt.savefig(HRFConsistencyFigDir + '/corr_' + figName);
                                                    plt.close()



















