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
# sizes = {'V1': [256],'V2': [256],'V3': [256],'hV4': [256]}


def removeDiag(mat):
    t = (1-np.eye(mat.shape[0]))
    t[t == 0] = np.nan
    mat_new = mat * t
    mat_new = mat_new.reshape(-1)
    mat_new = mat_new[~np.isnan(mat_new)]
    return mat_new
def plotHist(c,leg1,leg2):
    plt.hist(c, bins=np.arange(-1.05, 1.05, 0.1))
    plt.title('run ' + leg1 + ' vs. run ' + leg2 +', mean(r) ' + str(
        np.round(c.mean(), 2)))
    plt.xlabel('R')
    plt.ylabel('counts')
    plt.xlim((-1.05, 1.05))
    plt.ylim((0, len(c)))

def plotHist2(c,c2,leg1,leg2):
    x,bins,p = plt.hist(c, bins=np.arange(-1.05, 1.05, 0.1),density=True,label = 'diag, mean r = ' +str(np.round(c.mean(), 2)),alpha = 0.3)
    for item in p:
        item.set_height(item.get_height()/sum(x))
    x2, bins2, p2 = plt.hist(c2, bins=np.arange(-1.05, 1.05, 0.1), density=True,label = 'off diag, mean r = '+str(np.round(c2.mean(), 2)),alpha = 0.3)
    for item in p2:
        item.set_height(item.get_height() / sum(x2))
    plt.title('run ' + leg1 + ' vs. run ' + leg2 )
    plt.xlabel('R')
    plt.ylabel('counts')
    plt.legend()
    plt.xlim((-1.05, 1.05))
    plt.ylim((0, 1))

def normMat(mat):
    mat_mean = np.mean(mat,axis = 1)
    mat = mat - mat_mean.reshape(-1,1)
    return mat
def calCorr(mat1,mat2):
    c = np.zeros((32,))
    t1 = normMat(mat1.mean(axis=2))
    t2 = normMat(mat2.mean(axis=2))
    for i in range(32):
        c[i] = np.corrcoef(t1[:, i],t2[:, i])[0, 1]
        # c[i] = np.corrcoef(mat1[:, i, :].mean(axis=1), mat2[:, i, :].mean(axis=1))[0, 1]
    return c
def calCorr2(mat1,mat2):
    c = np.zeros((32*31))
    t1 = normMat(mat1.mean(axis=2))
    t2 = normMat(mat2.mean(axis=2))
    count = 0
    for i in range(32):
        for j in range(32):
            if i != j:
                c[count] = np.corrcoef(t1[:, i],t2[:, j])[0, 1]
                # c[count] = np.corrcoef(mat1[:, i, :].mean(axis=1),mat2[:, j, :].mean(axis=1))[0, 1]
                count = count + 1
    return c

def calCorr3(mat1,mat2):
    c = np.zeros((32,32))
    t1 = normMat(mat1.mean(axis=2))
    t2 = normMat(mat2.mean(axis=2))
    for i in range(32):
        for j in range(32):
            c[i,j] = np.corrcoef(t1[:, i],t2[:, j])[0, 1]
            # c[i,j] = np.corrcoef(mat1[:, i, :].mean(axis=1),mat2[:, j, :].mean(axis=1))[0, 1]
    return c

# sizes = {'V1': [16, 64, 256],'V2': [4, 16, 64, 256],'V3': [4, 16, 64, 256],'hV4': [4, 16, 64, 256]}
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
                                    maskTypes = ['firstLevel','secondLevel']
                                else:
                                    maskTypes = ['firstLevel']
                                for maskType in maskTypes:
                                    if maskType == 'firstLevel':
                                        maskFolder =  os.path.join('/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/individual', subject, session, 'masksNative','checkerboard_loc_firstLevel',topup,b0,HRFmodel)
                                        maskName = 'checkerboard_loc_firstLevel'
                                    else:
                                        maskFolder =  os.path.join('/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/individual', subject, session, 'masksNative',scan + '_'+maskType,topup,b0,HRFmodel)
                                        maskName = scan + '_'+maskType
                                    RepresentationStabilityDir = os.path.join(
                                        f'/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/individual',
                                        subject, session, 'functional', scan, 'GLM_single', topup, b0, HRFmodel,
                                        'RepresentationStability')
                                    os.makedirs(RepresentationStabilityDir, exist_ok=True)

                                    for region in list(sizes.keys()):
                                        regionSizes = sizes[region]
                                        for size in regionSizes:
                                            dataMats = {}
                                            dataName = f'{region}_{size:05}_voxels.npy'
                                            figName = f'{region}_{size:05}_voxels.png'
                                            dataMats = {}
                                            if (not exists(RepresentationStabilityDir+'/'+dataName))|(overwrite):
                                                for hemi in ['lh', 'rh']:
                                                    whichMask = maskFolder + f'/{region}_{hemi}_{size:05}_voxels.nii.gz'
                                                    img = nib.load(whichMask)
                                                    mask = img.get_fdata()
                                                    nVol = np.sum(mask==1)
                                                    nRun = len(experiment['scanInfo'][subject][session]['funcScans'][scan])
                                                    condNames = experiment['design'][scan]['conditions']['imageID']

                                                    if scan == 'object_noise':
                                                        dataMat_temp = np.zeros(
                                                            (nVol, len(condNames), 2))
                                                        design_reshape = np.zeros((len(designs) * len(condNames),))
                                                        for i in range(len(designs)):
                                                            design = designs[i]
                                                            design_reshape[(i * len(condNames)):(
                                                                        (i + 1) * len(condNames))] = np.argwhere(
                                                                design == 1)[:, 1]
                                                    elif scan == 'object_noise_mseq':
                                                        dataMat_temp = np.zeros(
                                                            (nVol, len(condNames), 2 * 2))
                                                        design_reshape = np.zeros((len(designs) * len(condNames) * 2,))
                                                        for i in range(len(designs)):
                                                            design = designs[i]
                                                            design_reshape[(i * len(condNames) * 2):(
                                                                        (i + 1) * len(condNames) * 2)] = np.argwhere(
                                                                design == 1)[:, 1]

                                                    if analysisType == 'GLM_single':
                                                        if len(designs) == 3:
                                                            combs = [[1, 2], [2, 3], [1, 3]]
                                                        elif len(designs) == 4:
                                                            combs = [[1, 4], [2, 4], [3, 4], [1, 2], [1, 3], [2, 3]]
                                                        elif len(designs) == 5:
                                                            combs = [[1, 4], [2, 4], [3, 4], [1, 2], [1, 3], [2, 3],]
                                                                     # [1, 5], [2, 5], [3, 5], [4, 5]]
                                                        for comb in combs:
                                                            s = ''
                                                            for i in range(len(comb)):
                                                                s = s + str(comb[i])
                                                                if i == 0:
                                                                    design_reshape_sub = design_reshape[int(len(design_reshape)/len(designs)*(comb[i]-1)):int(len(design_reshape)/len(designs)*(comb[i]))]
                                                                else:
                                                                    design_reshape_sub = np.concatenate((design_reshape_sub,design_reshape[int(len(design_reshape)/len(designs)*(comb[i]-1)):int(len(design_reshape)/len(designs)*(comb[i]))]),axis = 0)

                                                            outDir = os.path.join(
                                                                f'/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/individual',
                                                                subject, session, 'functional', scan, 'GLM_single',
                                                                topup, b0, HRFmodel,
                                                                'GLM_single' + s + '/')
                                                            results_glmsingle = dict()
                                                            results_glmsingle['typeb'] = np.load(
                                                                join(outDir, 'TYPEB_FITHRF.npy'),
                                                                allow_pickle=True).item()
                                                            results_glmsingle['typec'] = np.load(
                                                                join(outDir, 'TYPEC_FITHRF_GLMDENOISE.npy'),
                                                                allow_pickle=True).item()
                                                            results_glmsingle['typed'] = np.load(
                                                                join(outDir, 'TYPED_FITHRF_GLMDENOISE_RR.npy'),
                                                                allow_pickle=True).item()
                                                            for i in range(3):
                                                                if i == 0:
                                                                    results_glmsingle_sub = results_glmsingle['typeb'][
                                                                        'betasmd']
                                                                    nm = 'typeb_'+s
                                                                elif i == 1:
                                                                    results_glmsingle_sub = results_glmsingle['typec'][
                                                                        'betasmd']
                                                                    nm = 'typec_'+s
                                                                elif i == 2:
                                                                    results_glmsingle_sub = results_glmsingle['typed'][
                                                                        'betasmd']
                                                                    nm = 'typed_'+s
                                                                count1 = np.zeros((len(condNames),))
                                                                for j in range(results_glmsingle_sub.shape[3]):
                                                                    data = results_glmsingle_sub[:,:,:,j]
                                                                    w = int(design_reshape_sub[j])
                                                                    dataMat_temp[:nVol, w, int(count1[w])] = data[
                                                                        mask == 1]
                                                                    # dataMat_temp[nVol:, w, int(count1[w]),i] = np.nan
                                                                    count1[w] += 1

                                                                if hemi == 'lh':
                                                                    dataMats[nm] = dataMat_temp
                                                                else:
                                                                    dataMats[nm] = np.concatenate((dataMats[nm],dataMat_temp),axis =0)
                                                print(RepresentationStabilityDir+'/'+dataName)
                                                np.save(RepresentationStabilityDir+'/'+dataName, dataMats)
                                            else:
                                                dataMats = np.load(RepresentationStabilityDir+'/'+dataName,allow_pickle=True).item()
                                                RepresentationStabilityFigDir = os.path.join(
                                                    f'/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/individual',
                                                    subject, session, 'functional', scan, 'GLM_single', topup, b0,
                                                    HRFmodel,
                                                    'RepresentationStability/figure')
                                                os.makedirs(RepresentationStabilityFigDir, exist_ok=True)
                                                if len(dataMats) == 18:
                                                    accs = np.zeros((3, 6, 3))
                                                    for pp in range(3):
                                                        if pp == 0:
                                                            p = 'typeb_'
                                                        elif pp == 1:
                                                            p = 'typec_'
                                                        elif pp == 2:
                                                            p = 'typed_'
                                                        dataMat_reshape_12 = dataMats[p+'12'][:, :32, :]
                                                        dataMat_reshape_13 = dataMats[p+'13'][:, :32, :]
                                                        dataMat_reshape_14 = dataMats[p+'14'][:, :32, :]
                                                        dataMat_reshape_23 = dataMats[p+'23'][:, :32, :]
                                                        dataMat_reshape_24 = dataMats[p+'24'][:, :32, :]
                                                        dataMat_reshape_34 = dataMats[p+'34'][:, :32, :]

                                                        plt.figure(figsize=(12, 4))
                                                        plt.subplot(131)
                                                        c = calCorr(dataMat_reshape_12,dataMat_reshape_34)
                                                        plotHist(c, '12','34')
                                                        plt.subplot(132)
                                                        c = calCorr(dataMat_reshape_13, dataMat_reshape_24)
                                                        plotHist(c, '13', '24')
                                                        plt.subplot(133)
                                                        c = calCorr(dataMat_reshape_14, dataMat_reshape_23)
                                                        plotHist(c, '14', '23')
                                                        # plt.show()
                                                        plt.savefig(RepresentationStabilityFigDir + '/'+p + figName);
                                                        plt.close()

                                                        plt.figure(figsize=(12, 4))
                                                        plt.subplot(131)
                                                        c = calCorr(dataMat_reshape_12, dataMat_reshape_34)
                                                        c2 = calCorr2(dataMat_reshape_12, dataMat_reshape_34)
                                                        plotHist2(c, c2, '12', '34')
                                                        plt.subplot(132)
                                                        c = calCorr(dataMat_reshape_13, dataMat_reshape_24)
                                                        c2 = calCorr2(dataMat_reshape_13, dataMat_reshape_24)
                                                        plotHist2(c, c2, '13', '24')
                                                        plt.subplot(133)
                                                        c = calCorr(dataMat_reshape_14, dataMat_reshape_23)
                                                        c2 = calCorr2(dataMat_reshape_14, dataMat_reshape_23)
                                                        plotHist2(c, c2, '14', '23')
                                                        # plt.show()
                                                        plt.savefig(RepresentationStabilityFigDir + '/' + p + 'ver2_'+figName);
                                                        plt.close()

                                                        ## classification by correlation
                                                        acc = np.zeros((3,6))
                                                        c = calCorr3(dataMat_reshape_12, dataMat_reshape_34)
                                                        t1 = c[:16,:16]
                                                        t2 = c[16:,16:]
                                                        for i in range(16):
                                                            if np.argwhere(t1[:,i] == np.max(t1[:,i] )) == i:
                                                                acc[0,0] +=1
                                                            if np.argwhere(t2[:,i] == np.max(t2[:,i] )) == i:
                                                                acc[1,0] +=1
                                                            if np.argwhere(t1[i,:] == np.max(t1[i,:] )) == i:
                                                                acc[0,1] +=1
                                                            if np.argwhere(t2[i,:] == np.max(t2[i,:] )) == i:
                                                                acc[1,1] +=1
                                                            if (np.argwhere(c[:,i] == np.max(c[:,i] )) == i)|(np.argwhere(c[:,i] == np.max(c[:,i] )) == i+16):
                                                                acc[2,0] +=1
                                                            if (np.argwhere(c[:, i + 16] == np.max(c[:, i + 16])) == i)|(np.argwhere(c[:, i + 16] == np.max(c[:, i + 16])) == i+16):
                                                                acc[2,0] +=1
                                                            if (np.argwhere(c[i, :] == np.max(c[i, :])) == i)|(np.argwhere(c[i, :] == np.max(c[i, :])) == i+16):
                                                                acc[2,1] +=1
                                                            if (np.argwhere(c[i + 16, :] == np.max(c[i + 16, :])) == i)|(np.argwhere(c[i + 16, :] == np.max(c[i + 16, :])) == i+16):
                                                                acc[2,1] +=1
                                                        c = calCorr3(dataMat_reshape_13, dataMat_reshape_24)
                                                        t1 = c[:16,:16]
                                                        t2 = c[16:,16:]
                                                        for i in range(16):
                                                            if np.argwhere(t1[:,i] == np.max(t1[:,i] )) == i:
                                                                acc[0,2] +=1
                                                            if np.argwhere(t2[:,i] == np.max(t2[:,i] )) == i:
                                                                acc[1,2] +=1
                                                            if np.argwhere(t1[i,:] == np.max(t1[i,:] )) == i:
                                                                acc[0,3] +=1
                                                            if np.argwhere(t2[i,:] == np.max(t2[i,:] )) == i:
                                                                acc[1,3] +=1
                                                            if (np.argwhere(c[:,i] == np.max(c[:,i] )) == i)|(np.argwhere(c[:,i] == np.max(c[:,i] )) == i+16):
                                                                acc[2,2] +=1
                                                            if (np.argwhere(c[:, i + 16] == np.max(c[:, i + 16])) == i)|(np.argwhere(c[:, i + 16] == np.max(c[:, i + 16])) == i+16):
                                                                acc[2,2] +=1
                                                            if (np.argwhere(c[i, :] == np.max(c[i, :])) == i)|(np.argwhere(c[i, :] == np.max(c[i, :])) == i+16):
                                                                acc[2,3] +=1
                                                            if (np.argwhere(c[i + 16, :] == np.max(c[i + 16, :])) == i)|(np.argwhere(c[i + 16, :] == np.max(c[i + 16, :])) == i+16):
                                                                acc[2,3] +=1
                                                        c = calCorr3(dataMat_reshape_14, dataMat_reshape_23)
                                                        t1 = c[:16, :16]
                                                        t2 = c[16:, 16:]
                                                        for i in range(16):
                                                            if np.argwhere(t1[:, i] == np.max(t1[:, i])) == i:
                                                                acc[0, 4] += 1
                                                            if np.argwhere(t2[:, i] == np.max(t2[:, i])) == i:
                                                                acc[1, 4] += 1
                                                            if np.argwhere(t1[i, :] == np.max(t1[i, :])) == i:
                                                                acc[0, 5] += 1
                                                            if np.argwhere(t2[i, :] == np.max(t2[i, :])) == i:
                                                                acc[1, 5] += 1
                                                            if (np.argwhere(c[:,i] == np.max(c[:,i] )) == i)|(np.argwhere(c[:,i] == np.max(c[:,i] )) == i+16):
                                                                acc[2, 4] += 1
                                                            if (np.argwhere(c[:, i + 16] == np.max(c[:, i + 16])) == i)|(np.argwhere(c[:, i + 16] == np.max(c[:, i + 16])) == i+16):
                                                                acc[2, 4] += 1
                                                            if (np.argwhere(c[i, :] == np.max(c[i, :])) == i)|(np.argwhere(c[i, :] == np.max(c[i, :])) == i+16):
                                                                acc[2, 5] += 1
                                                            if (np.argwhere(c[i + 16, :] == np.max(c[i + 16, :])) == i)|(np.argwhere(c[i + 16, :] == np.max(c[i + 16, :])) == i+16):
                                                                acc[2, 5] += 1
                                                        # plt.show()
                                                        acc[0:2,:]/=16
                                                        acc[2,:]/=32
                                                        accs[:,:,pp]=acc
                                                        plt.figure(figsize=(6, 6))
                                                        plt.bar([0,1,2],acc.mean(axis = 1),width = 0.5)
                                                        plt.hlines(y = 1/16,xmin =-1,xmax=3,color = [0.5,0.5,0.5],linestyles = 'dashed')
                                                        plt.errorbar([0,1,2],acc.mean(axis = 1),np.std(acc,axis = 1)/(6**0.5),color = 'k',ls= 'none')
                                                        plt.title('16-way classification \nbased on Haxby corr method')
                                                        plt.xticks([0,1,2],['Clean','Gaussian \nnoise', 'across clean \n and Gaussian'])
                                                        plt.ylabel('accuracy')
                                                        plt.xlim((-1,3))
                                                        plt.ylim((0, 1))
                                                        # plt.show()
                                                        plt.savefig(RepresentationStabilityFigDir + '/' + p + 'classification_'+figName);
                                                        plt.close()
                                                    plt.figure(figsize=(6, 6))
                                                    plt.bar([0,1,2],accs[:,:,0].mean(axis = 1),width = 0.15,label = 'GLM')
                                                    plt.bar([0.25,1.25,2.25],accs[:,:,1].mean(axis = 1),width = 0.15,label = 'GLM_denoise')
                                                    plt.bar([0.5,1.5,2.5],accs[:,:,2].mean(axis = 1),width = 0.15,label = 'GLM_denoise+ridge')
                                                    plt.errorbar([0,1,2],accs[:,:,0].mean(axis = 1),np.std(accs[:,:,0],axis = 1)/(6**0.5),color = 'k',ls= 'none')
                                                    plt.errorbar([0.25,1.25,2.25],accs[:,:,1].mean(axis = 1),np.std(accs[:,:,1],axis = 1)/(6**0.5),color = 'k',ls= 'none')
                                                    plt.errorbar([0.5,1.5,2.5],accs[:,:,2].mean(axis = 1),np.std(accs[:,:,2],axis = 1)/(6**0.5),color = 'k',ls= 'none')
                                                    plt.hlines(y = 1/16,xmin =-1,xmax=3,color = [0.5,0.5,0.5],linestyles = 'dashed')
                                                    plt.title('16-way classification \nbased on Haxby corr method')
                                                    plt.xticks([0.25,1.25,2.25],['Clean','Gaussian \nnoise', 'across clean \n and Gaussian'])
                                                    plt.ylabel('accuracy')
                                                    plt.xlim((-0.5,3))
                                                    plt.ylim((0, 1))
                                                    plt.legend()
                                                    # plt.show()
                                                    plt.savefig(RepresentationStabilityFigDir + '/classification_'+figName);
                                                    plt.close()

