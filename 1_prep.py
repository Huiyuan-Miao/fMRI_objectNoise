#!/usr/bin/python
'''
prepares directory structure, runs brain extractions, prepares b0 fieldmaps,
and all other steps necessary before FSL design files are created
'''

import sys
import os
import glob
import pickle
import itertools
import numpy as np
from scipy.io import loadmat
from argparse import Namespace
import matplotlib.pyplot as plt
import datetime
import shutil

# get scan info from experiment file
from fMRI.analysis.scripts.experiment import experiment
from fMRI.analysis.scripts.apply_TOPUP import apply_topup

for subject in experiment['scanInfo']:
    for s, session in enumerate(experiment['scanInfo'][subject]):
        print(f'\n\nSubject: {subject}, Session: {session}\n\n')

        # dir info
        sessID = experiment['scanInfo'][subject][session]['sessID']
        sessDir = os.path.join(f'/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/individual', subject, session)
        rawDir = os.path.join(sessDir, 'rawData')
        anatDir = os.path.join(sessDir, 'anatomical')
        os.makedirs(anatDir, exist_ok=True)

        # run segmentation if a new subject
        freesurferSubjectDir = f'/home/tonglab/freesurfer/subjects/{subject}'
        if not os.path.isdir(os.path.join(freesurferSubjectDir)):
            print('Running surface segmentation...')
            anatScan = experiment['scanInfo'][subject][session]['anatScan']
            inFile = glob.glob(os.path.join(rawDir, f'*{sessID}.{anatScan:02}*.nii'))[0]
            freesurferCommand = f'recon-all -subject {subject} -all -i {inFile}'
            # os.system(f'bash /mnt/HDD12TB/masterScripts/fMRI/callFreesurferFunction.sh -s "{freesurferCommand}"')
            os.system(freesurferCommand)

        # convert mgz to nifti
        if not os.path.isfile(f'{freesurferSubjectDir}/mri/orig/anatomical.nii'):
            freesurferCommand = f'mri_convert {freesurferSubjectDir}/mri/orig/001.mgz {freesurferSubjectDir}/mri/orig/anatomical.nii'
            # os.system(f'bash /mnt/HDD12TB/masterScripts/fMRI/callFreesurferFunction.sh -s "{freesurferCommand}"')
            os.system(freesurferCommand)

        # extract brain
        if not os.path.isfile(f'{freesurferSubjectDir}/mri/orig/anatomical_brain.nii.gz'):
            os.system(f'bet {freesurferSubjectDir}/mri/orig/anatomical.nii {freesurferSubjectDir}/mri/orig/anatomical_brain.nii.gz')

        # copy to session dir
        if not os.path.isfile(f'{anatDir}/anatomical.nii'):
            shutil.copyfile(f'{freesurferSubjectDir}/mri/orig/anatomical.nii', f'{anatDir}/anatomical.nii')
        if not os.path.isfile(f'{anatDir}/anatomical_brain.nii'):
            shutil.copyfile(f'{freesurferSubjectDir}/mri/orig/anatomical_brain.nii.gz', f'{anatDir}/anatomical_brain.nii.gz')

        # b0 field map preprocessing
        if 'b0Scan' in experiment['scanInfo'][subject][session].keys():  # only run if b0 map exists

            b0dir = os.path.join(sessDir, 'b0')
            os.makedirs(b0dir, exist_ok=True)
            finalFile = os.path.join(b0dir, 'b0_realFieldMapImage_rads_reg_unwrapped.nii.gz')

            # Preprocessing real field map image
            realFile = glob.glob(os.path.join(rawDir, f'*{sessID}.{experiment["scanInfo"][subject][session]["b0Scan"]:02}*e2*.nii'))[0]
            realFileCopy = os.path.join(b0dir, 'b0_realFieldMapImage.nii.gz')
            shutil.copyfile(realFile, realFileCopy)
            os.system(f'fslmaths {realFileCopy} -div 500 -mul 3.1415 {realFileCopy[:-7]}_rads.nii.gz') # current range is -500:500, rescale to +/- pi otherwise fugue will barf
            os.system(f'fugue --loadfmap={realFileCopy[:-7]}_rads.nii.gz -m --savefmap={realFileCopy[:-7]}_rads_reg.nii.gz')  # regularization

            # Preprocessing magnitude image
            magnitudeFile = glob.glob(os.path.join(rawDir, f'*{sessID}.{experiment["scanInfo"][subject][session]["b0Scan"]:02}*e1*.nii'))[0]
            magnitudeFileCopy = os.path.join(b0dir, 'b0_magnitudeImage.nii.gz')
            shutil.copyfile(magnitudeFile, magnitudeFileCopy)
            os.system(f'bet {magnitudeFileCopy} {magnitudeFileCopy[:-7]}_brain.nii.gz')  # brain extraction
            os.system(f'fslmaths {magnitudeFileCopy[:-7]}_brain.nii.gz -ero {magnitudeFileCopy[:-7]}_brain_ero.nii.gz')  # erode/remove one voxel from all edges

            # b0 unwarping using fugue
            te = 0.00226  # echo time
            wfs = 40.284  # water fat shift
            acc = 1  # acceleration factor
            npe = 160  # phase encoding steps
            fstrength = 7  # field strength
            wfd_ppm = 3.4  # water fat density
            g_ratio_mhz_t = 42.57

            etl = 54 #npe / acc echo train length
            epif = etl - 1 # epic factor
            wfs_hz = fstrength * wfd_ppm * g_ratio_mhz_t
            ees = wfs / (wfs_hz * etl) / acc

            if not os.path.isfile(finalFile):
                os.system(f'prelude -a {magnitudeFileCopy[:-7]}_brain_ero.nii.gz -p {realFileCopy[:-7]}_rads_reg.nii.gz -u {finalFile}')

        # copy over funcNoEPI scan
        if 'funcNoEPIscan' in experiment['scanInfo'][subject][session].keys():
            FNEdir = os.path.join(sessDir, 'funcNoEPI')
            os.makedirs(FNEdir, exist_ok=True)
            FNEscan = experiment['scanInfo'][subject][session]['funcNoEPIscan']
            inFile = glob.glob(os.path.join(rawDir, f'*{sessID}.{FNEscan:02}*.nii'))[0]
            outFile = os.path.join(FNEdir, 'funcNoEPI.nii')
        if not os.path.exists(outFile):
            shutil.copyfile(inFile, outFile)

        for scan in experiment['scanInfo'][subject][session]['funcScans'].keys():

            # get experiment design from file
            nRuns = len(list(experiment['scanInfo'][subject][session]['funcScans'][scan]))
            if (scan != 'resting_state') & (scan != 'prf'):
                params = Namespace(**experiment['design'][scan]['params'])
                # nBlocks = int(((params.nDynamics * params.TR) - (params.initialFixation + params.finalFixation)) / (
                #     params.blockDuration + params.IBI))

            '''
            # get conditions (only do this for scans with multiple conditions)
            if scan == 'rapid_event_related_object_noise':
                variables = list(experiment['design'][scan]['conditions'].keys())
                levels = [experiment['design'][scan]['conditions'][variable] for variable in variables]
                conds = list(itertools.product(*levels))
                nConds = len(conds)
                nReps = int(nBlocks / nConds)
                condNames = []
                for cond in conds:
                    condNames.append(f'{cond[0]}_{cond[1]}')
            '''
            for r, run in enumerate(experiment['scanInfo'][subject][session]['funcScans'][scan]):

                print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | Subject: {subject} | Session: {session} | Scan: {scan} | Run: {r+1}')
                funcDir = os.path.join(sessDir, 'functional', scan, f'run{r+1:02}')
                os.makedirs(os.path.join(funcDir, 'inputs'), exist_ok=True)

                # move raw data
                inFile = glob.glob(os.path.join(rawDir, f'*{sessID}.{run:02}*.nii'))[0]
                outFile = os.path.join(funcDir, 'inputs/rawData.nii')
                if not os.path.exists(outFile):
                    shutil.copyfile(inFile, outFile)

                # move top up file
                inFile = glob.glob(os.path.join(rawDir, f'*{sessID}.{run+1:02}*.nii'))[0]
                outFile = os.path.join(funcDir, 'inputs/oppPE.nii')
                if not os.path.exists(outFile):
                    shutil.copyfile(inFile, outFile)

                # apply Top Up
                inFile = os.path.join(funcDir, 'inputs/rawData.nii')
                topupFile = os.path.join(funcDir, 'inputs/oppPE.nii')
                outFile = os.path.join(funcDir, 'inputs/rawData_topUp.nii.gz')  # set final filename for top up output
                outFile2 = os.path.join(funcDir, 'inputs/rawData_TUcorrected.nii.gz')  # set final filename for top up output

                if not os.path.isfile(outFile):
                    print('Running Top Up...')
                    # os.system(f'python3 /home/tonglab/Miao/fMRI/masterScripts/apply_topup.py')
                    apply_topup(inFile,topupFile,90,270,outFile2)
                    # os.system(f'python2 /home/tonglab/Miao/fMRI/masterScripts/fsl_TOPUP_call.py {inFile} {topupFile} 90 270')
                    toppedUpFile = glob.glob(f'{inFile[:-4]}*TUcorrected.nii.gz')[0]
                    os.rename(toppedUpFile, outFile)
                extraFiles = glob.glob(os.path.join(funcDir, 'inputs/*topup*'))
                for extrafile in extraFiles:
                    os.remove(extrafile)
                extraFiles = glob.glob('*topup*')
                for extrafile in extraFiles:
                    os.remove(extrafile)

                if 'b0Scan' in experiment['scanInfo'][subject][session].keys():  # only run if b0 map exists

                    for topup, withWithout in zip(['_topUp', ''], ['with', 'without']):

                        inFile = glob.glob(os.path.join(funcDir, f'inputs/rawData{topup}.nii*'))[0]
                        outFile = os.path.join(funcDir, f'inputs/rawData{topup}_b0.nii.gz')  # set final filename for b0 output

                        if os.path.isfile(inFile) and not os.path.isfile(outFile):
                            print(f'b0 correcting scan {withWithout} topup')
                            os.system(f'fugue -i {inFile} --dwell={ees} --loadfmap={realFileCopy[:-7]}_rads_reg_unwrapped.nii.gz --unwarpdir=y- --asym={te} --despike -u {outFile}')

                # generate event files from log files
                if scan == 'object_noise':
                    condNames = experiment['design'][scan]['conditions']['imageID']
                    eventDir = os.path.join(sessDir, 'events', scan, f'3column/run{r+1:02}')
                    os.makedirs(eventDir, exist_ok=True)
                    logFile = glob.glob(os.path.join(sessDir, 'events', scan, f'logFiles/*_rn{r+1}_*'))[0]
                    logData = loadmat(logFile)
                    eventData = logData['onsetTime'][0]
                    for c, cond in enumerate(condNames):
                        eventFile = os.path.join(eventDir, f'{cond}.txt')
                        with open(eventFile, 'w+') as file:
                            start = eventData[c]
                            file.write('%i\t%i\t1' % (start, params.imageDuration))
                            # file.write('\n')
                        file.close()

                if scan == 'object_noise_mseq':
                    condNames = experiment['design'][scan]['conditions']['imageID']
                    eventDir = os.path.join(sessDir, 'events', scan, f'3column/run{r+1:02}')
                    os.makedirs(eventDir, exist_ok=True)
                    logFile = glob.glob(os.path.join(sessDir, 'events', scan, f'logFiles/*_rn{r+1}_*'))[0]
                    logData = loadmat(logFile)
                    eventData = logData['onsetTime'][0]
                    for c, cond in enumerate(condNames):
                        # print(c)
                        eventFile = os.path.join(eventDir, f'{cond}.txt')
                        with open(eventFile, 'w+') as file:
                            start = eventData[c*2]
                            file.write('%i\t%i\t1' % (start, params.imageDuration))
                            file.write('\n')
                            start = eventData[c*2+1]
                            file.write('%i\t%i\t1' % (start, params.imageDuration))
                        file.close()

                if scan == 'checkerboard_loc':
                    eventDir = os.path.join(sessDir, 'events', scan, f'3column')
                    os.makedirs(eventDir, exist_ok=True)
                    start_times = {'left': np.arange(12,300,24).tolist(),
                                   'right': np.arange(24,300,24).tolist()}
                    for cond in ['left','right']:
                        eventFile = os.path.join(eventDir, f'{cond}.txt')
                        thesePositions = start_times[cond]
                        with open(eventFile, 'w+') as file:
                            for p in thesePositions:
                                start = int(p)
                                file.write('%i\t%i\t1' % (start, params.blockDuration))
                                if p != thesePositions[-1]:
                                    file.write('\n')
                        file.close()
                if scan == 'FHO_loc':
                    eventDir = os.path.join(sessDir, 'events', scan, f'3column')
                    os.makedirs(eventDir, exist_ok=True)
                    start_times = {'face': np.arange(12,300,96).tolist(),
                                   'house': np.arange(36, 300, 96).tolist(),
                                   'object': np.arange(60, 300, 96).tolist(),
                                   'scramble': np.arange(84,300,96).tolist()}
                    for cond in ['face','house','object','scramble']:
                        eventFile = os.path.join(eventDir, f'{cond}.txt')
                        thesePositions = start_times[cond]
                        with open(eventFile, 'w+') as file:
                            for p in thesePositions:
                                start = int(p)
                                file.write('%i\t%i\t1' % (start, params.blockDuration))
                                if p != thesePositions[-1]:
                                    file.write('\n')
                        file.close()


                # make output directories for FSL analyses and designs
                for HRFmodel in ['doubleGamma']:

                    if scan != 'restingState':
                        designDir = os.path.join(f'fMRI/data/designs/firstLevel', scan, HRFmodel)
                        os.makedirs(designDir, exist_ok=True)

                        for topup in ['noTopUp', 'topUp']:
                            for b0 in ['noB0', 'b0']:

                                outDir = os.path.join(sessDir, 'functional', scan, f'run{r+1:02}/outputs', topup, b0, HRFmodel)
                                os.makedirs(outDir, exist_ok=True)

        # get reference func images and make registration
        regDir = os.path.join(sessDir, 'reg')
        os.makedirs(regDir, exist_ok=True)
        scanList = []
        for scan in experiment['scanInfo'][subject][session]['funcScans'].keys():
            scanList.append(experiment['scanInfo'][subject][session]['funcScans'][scan])
        scanList = sorted(itertools.chain.from_iterable(scanList))
        refScan = scanList[int(len(scanList) / 2)]
        for scan in experiment['scanInfo'][subject][session]['funcScans'].keys():
            if refScan in experiment['scanInfo'][subject][session]['funcScans'][scan]:
                refScanName = scan
                refScanRun = experiment['scanInfo'][subject][session]['funcScans'][scan].index(refScan) + 1
        for topup, topupString in zip(['topUp', 'noTopUp'],['_topUp','']):
            # for b0, b0String in zip(['b0', 'noB0'],['_b0', '']):
            for b0, b0String in zip([ 'noB0'],[ '']):
                inPath = glob.glob(f'{sessDir}/functional/{refScanName}/run{refScanRun:02}/inputs/rawData{topupString}{b0String}.nii*')[0]
                outPath = os.path.join(os.getcwd(), regDir, f'refFunc{topupString}{b0String}.nii.gz')
                if not os.path.isfile(outPath):
                    os.system(f'fslmaths {inPath} -Tmean {outPath}')
                regPath = os.path.join(os.getcwd(), regDir, f'func2surf{topupString}{b0String}.dat')
                if not os.path.exists(regPath):
                    freesurferCommand = f'bbregister --s {subject} --mov {outPath} --init-fsl --reg {regPath} --bold'
                    os.system(freesurferCommand)

print('Done.')