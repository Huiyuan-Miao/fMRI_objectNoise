import numpy as np
import nibabel as nib
from scipy.stats import sem
import os
import glob
import matplotlib.pyplot as plt
import datetime
import shutil
import pandas as pd
import pickle as pkl

overwrite = False
MRthresh = 0#50000

# get scan info from experiment file
from fMRI.analysis.scripts.experiment import experiment

version = 'v5'

# store collated data in a dictionary of lists
maxSessions = 1
subjects = list(experiment['scanInfo'].keys())[1:]
nSubjects = len(subjects)
dqDir = f'{version}/analysis/results/dataQuality'
os.makedirs(dqDir, exist_ok=True)
regions = ['LGN', 'V1', 'ventral_stream', 'cortex']

# tSNR across all subjects (performed in anatomical space as native space registrations were poor)
print('preprocessing...')
for topup, topupString in zip(['topUp', 'noTopUp'], ['_topUp', '']):
    for b0, b0String in zip(['noB0'], ['']):  # zip(['b0', 'noB0'], ['_b0', '']):
        means = []
        stds = []
        for subject in subjects:
            for session in experiment['scanInfo'][subject].keys():
                sessDir = f'{version}/data/individual/{subject}/{session}'
                for run in ['run01','run02']:
                    dataPath = glob.glob(os.path.join(sessDir, f'functional/resting_state/{run}/inputs/rawData{topupString}{b0String}.nii*'))[0]
                    tSNRdir = os.path.join(os.path.dirname(os.path.dirname(dataPath)), f'outputs/{topup}/{b0}/tSNR')
                    os.makedirs(tSNRdir, exist_ok=True)

                    # run motion correction
                    dataMotCor = os.path.join(tSNRdir, "motionCorrected.nii.gz")
                    if not os.path.exists(dataMotCor):
                        print('motion correction...')
                        os.system(f'mcflirt -in {dataPath} -out {dataMotCor}')

                    # calculate tSNR
                    pathTmean = os.path.join(tSNRdir, "Tmean.nii.gz")
                    pathTstd = os.path.join(tSNRdir, "Tstd.nii.gz")
                    pathTSNR = os.path.join(tSNRdir, "tSNR.nii.gz")
                    if not os.path.exists(pathTmean):
                        print('Creating mean of timeseries...')
                        os.system(f'fslmaths {dataMotCor} -Tmean {pathTmean}')
                    if not os.path.exists(pathTstd):
                        print('Creating std of timeseries...')
                        os.system(f'fslmaths {dataMotCor} -Tstd {pathTstd}')
                    if not os.path.exists(pathTSNR):
                        print('Calculating tSNR map...')
                        os.system(f'fslmaths {pathTmean} -div {pathTstd} {pathTSNR}')

                    # convert cortex mask to func space
                    bilatCortexMask = f'{tSNRdir}/cortex_bi.nii.gz'

                    if not os.path.isfile(bilatCortexMask) or overwrite:

                        # convert freesurfer anat from mgz to nii
                        fsAnatFileMGZ = f'/mnt/HDD12TB/freesurfer/subjects/{subject}/mri/orig.mgz'  # do not confuse with original nifti
                        fsAnatFileNII = f'{tSNRdir}/anatomical_fs.nii'
                        freesurferCommand = f'mri_convert --in_type mgz --out_type nii --out_orientation RAS {fsAnatFileMGZ} {fsAnatFileNII}'
                        os.system(f'bash /mnt/HDD12TB/masterScripts/fMRI/callFreesurferFunction.sh -s "{freesurferCommand}"')

                        # register freesurfer anat to orig anat (copy of anatomical saved by recon-all is not same as original nifti)
                        origAnatFile = f'/mnt/HDD12TB/freesurfer/subjects/{subject}/mri/orig/anatomical.nii'
                        fsAnat2origAnatMat = f'{tSNRdir}/fsAnat2origAnat.mat'
                        os.system(f'flirt -in {fsAnatFileNII} -ref {origAnatFile} -omat {fsAnat2origAnatMat}')

                        # convert cortical mask to func space
                        for hemi in ['lh', 'rh']:
                            print(f'Converting {hemi} cortical mask...')

                            # cortex mgz to nifti
                            inFile = f'/mnt/HDD12TB/freesurfer/subjects/{subject}/mri/{hemi}.ribbon.mgz'
                            outFile = f'{tSNRdir}/cortex_{hemi}_FSanat.nii.gz'
                            if not os.path.exists(outFile):
                                freesurferCommand = f'mri_convert --in_type mgz --out_type nii --out_orientation RAS {inFile} {outFile}'
                                os.system(f'bash /mnt/HDD12TB/masterScripts/fMRI/callFreesurferFunction.sh -s "{freesurferCommand}"')

                            # freesurfer anat space to orig anat space
                            inFile = outFile
                            outFile = f'{tSNRdir}/cortex_{hemi}_origAnat.nii.gz'
                            if not os.path.exists(outFile):
                                os.system(f'flirt -in {inFile} -ref {origAnatFile} -applyxfm -init {fsAnat2origAnatMat} -interp nearestneighbour -out {outFile}')

                            # orig anat space to func space
                            inFile = outFile
                            outFile = f'{tSNRdir}/cortex_{hemi}.nii.gz'
                            origAnat2funcMat = os.path.join(f'{version}/data/individual/{subject}/{session}/functional/localizer/run01/outputs/{topup}/{b0}/'
                                                            f'doubleGamma/firstLevel.feat/reg/highres2example_func.mat')
                            os.system(f'flirt -in {inFile} -ref {pathTmean} -applyxfm -init {origAnat2funcMat} -interp nearestneighbour -out {outFile}')
                        os.system(f'fslmaths {tSNRdir}/cortex_lh.nii.gz -add {tSNRdir}/cortex_rh.nii.gz -bin {bilatCortexMask}')


                    # get xform mats
                    regDir = os.path.join(f'{version}/data/individual/{subject}/{session}/functional/figure_ground/run01/outputs/{topup}/{b0}/'
                                          f'doubleGamma/firstLevel.feat')
                    exampleFunc = f'{regDir}/example_func.nii.gz'
                    std2funcMat = f'{regDir}/reg/standard2example_func.mat'

                    for region in regions:

                        # convert masks from standard to func space
                        if region in ['V1', 'ventral_stream']:

                            # standard space to func space
                            inFile = glob.glob(f'/mnt/HDD12TB/masks/**/*{region}.nii.gz')[0]
                            outFile = os.path.join(tSNRdir, f'{region}_func.nii.gz')

                            if not os.path.exists(outFile) or overwrite:
                                print('Converting standard space masks to native func space...')
                                os.system(f'flirt -in {inFile} -ref {exampleFunc} -out {outFile} -init {std2funcMat} -applyxfm -interp nearestneighbour')

                            # combine cortex mask with ROI mask
                            finalMask = f'{tSNRdir}/{region}_cortex.nii.gz'
                            if not os.path.isfile(finalMask) or overwrite:
                                print('Combining ROI mask with cortical mask...')
                                os.system(f'fslmaths {tSNRdir}/cortex_bi.nii.gz -mul {outFile} {finalMask}')

                        # combine left and right native space masks
                        elif region == 'LGN':
                            finalMask = f'{tSNRdir}/LGN_bi.nii.gz'
                            leftHemi = os.path.join(sessDir, 'masksNative/LGN_lh_00004_voxels.nii.gz')
                            rightHemi = os.path.join(sessDir, 'masksNative/LGN_rh_00004_voxels.nii.gz')
                            if os.path.isfile(leftHemi) and os.path.isfile(rightHemi):
                                os.system(f'fslmaths {leftHemi} -add {rightHemi} {finalMask}')
                            elif os.path.isfile(leftHemi):
                                shutil.copy(leftHemi, finalMask)
                            elif os.path.isfile(rightHemi):
                                shutil.copy(rightHemi, finalMask)

                        # cortical mask already processed
                        elif region == 'cortex':
                            finalMask = f'{tSNRdir}/cortex_bi.nii.gz'


# compare classic and multiX in M015
print('calculating tSNR for each subject and each run...')
for subject in ['M015']:#subjects:
    session = list(experiment['scanInfo'][subject].keys())[0]
    sessDir = f'{version}/data/individual/{subject}/{session}'
    for region in regions:
        if region == 'cortex':
            testMask = os.path.join(sessDir, f'functional/resting_state/run01/outputs/noTopUp/noB0/tSNR/cortex_bi.nii.gz')
        else:
            testMask = os.path.join(sessDir, f'functional/resting_state/run01/outputs/noTopUp/noB0/tSNR/{region}_cortex.nii.gz')
        if os.path.isfile(testMask):
            outDir = f'{dqDir}/tSNR/{subject}/{session}/{region}'
            os.makedirs(outDir, exist_ok=True)
            plotData = {'topUp': {'classic': {'mean': 0,
                                              'std': 0},
                                  'multiX': {'mean': 0,
                                             'std': 0}},
                        'noTopUp': {'classic': {'mean': 0,
                                                'std': 0},
                                    'multiX': {'mean': 0,
                                               'std': 0}}}
            for run, scanType in zip(['run01','run02'], ['classic','multiX']):
                for topup, topupString in zip(['topUp', 'noTopUp'], ['_topUp', '']):
                    for b0, b0String in zip(['noB0'], ['']):  # zip(['b0', 'noB0'], ['_b0', '']):

                        dataPath = glob.glob(os.path.join(sessDir, f'functional/resting_state/{run}/inputs/rawData{topupString}{b0String}.nii*'))[0]
                        tSNRdir = os.path.join(os.getcwd(), os.path.dirname(os.path.dirname(dataPath)), f'outputs/{topup}/{b0}/tSNR')
                        pathTSNR = os.path.join(tSNRdir, 'tSNR.nii.gz')

                        if region in ['cortex']:
                            finalMask = f'{tSNRdir}/{region}_bi.nii.gz'
                        else:
                            finalMask = f'{tSNRdir}/{region}_cortex.nii.gz'

                        # get tSNR values
                        plotData[topup][scanType]['mean'] = float(os.popen(f'fslstats {pathTSNR} -k {finalMask} -m').read())
                        plotData[topup][scanType]['std'] = float(os.popen(f'fslstats {pathTSNR} -k {finalMask} -s').read())

            classic_means = [plotData['topUp']['classic']['mean'], plotData['noTopUp']['classic']['mean']]
            classic_stds = [plotData['topUp']['classic']['std'], plotData['noTopUp']['classic']['std']]
            multiX_means = [plotData['topUp']['multiX']['mean'], plotData['noTopUp']['multiX']['mean']]
            multiX_stds = [plotData['topUp']['multiX']['std'], plotData['noTopUp']['multiX']['std']]

            x_pos = [0, 1]
            fig, ax = plt.subplots()
            width = .33
            horzOffsets = [-.33, .33]
            fig.set_figwidth(4)
            ax.bar([-.167, 1 - .167], classic_means, width, yerr=classic_stds, align='center',
                   ecolor='black', capsize=10, label='classic')
            ax.bar([.167, 1.167], multiX_means, width, yerr=multiX_stds, align='center',
                   ecolor='black', capsize=10, label='multiX')
            ax.set_ylabel('tSNR')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(['topUp', 'noTopUp'])
            ax.set_title(f'resting state tSNR, {region}')
            ax.yaxis.grid(True)
            ax.legend()
            plt.tight_layout()
            plt.savefig(f'{outDir}/tSNR.png')
            plt.show()
            plt.close()

    # make histogram of tSNR values in cortical mask
    for run, scanType in zip(['run01','run02'], ['classic','multiX']):

        for topup, topupString in zip(['topUp', 'noTopUp'], ['_topUp', '']):
            for b0, b0String in zip(['noB0'], ['']):  # zip(['b0', 'noB0'], ['_b0', '']):
                dataPath = glob.glob(os.path.join(sessDir, f'functional/resting_state/{run}/inputs/rawData{topupString}{b0String}.nii*'))[0]
                tSNRdir = os.path.join(os.getcwd(), os.path.dirname(os.path.dirname(dataPath)), f'outputs/{topup}/{b0}/tSNR')
                pathTSNR = os.path.join(tSNRdir, 'tSNR.nii.gz')
                corticalMask = f'{tSNRdir}/cortex_bi.nii.gz'
                outDir = f'{dqDir}/tSNR/{subject}/{session}/cortex/{scanType}/{topup}/{b0}'
                os.makedirs(outDir, exist_ok=True)
                histData = [float(x) for x in os.popen(f'fslstats {pathTSNR} -k {corticalMask} -H 100 0 100').read().split()]
                max = float(os.popen(f'fslstats {pathTSNR} -k {corticalMask} -p 100').read())
                min = float(os.popen(f'fslstats {pathTSNR} -k {corticalMask} -p 0').read())
                x_pos = np.arange(0,100)
                fig, ax = plt.subplots()
                ax.bar(x_pos, histData, width=1)
                ax.set_title(f'{subject}, {scanType}, {topup}, {b0} correction')
                plt.xlabel('tSNR')
                plt.ylabel('voxel count')
                ax.yaxis.grid(True)
                plt.tight_layout()
                plt.savefig(f'{outDir}/tSNR_hist_{scanType}_{topup}_{b0}.png')
                plt.show()
                plt.close()

    # make histogram of mean MR intensity values in cortical mask
    for run, scanType in zip(['run01','run02'], ['classic','multiX']):

        for topup, topupString in zip(['topUp', 'noTopUp'], ['_topUp', '']):
            for b0, b0String in zip(['noB0'], ['']):  # zip(['b0', 'noB0'], ['_b0', '']):
                dataPath = glob.glob(os.path.join(sessDir, f'functional/resting_state/{run}/inputs/rawData{topupString}{b0String}.nii*'))[0]
                tSNRdir = os.path.join(os.getcwd(), os.path.dirname(os.path.dirname(dataPath)), f'outputs/{topup}/{b0}/tSNR')
                pathTmean = os.path.join(tSNRdir, 'Tmean.nii.gz')
                corticalMask = f'{tSNRdir}/cortex_bi.nii.gz'
                outDir = f'{dqDir}/tSNR/{subject}/{session}/cortex/{scanType}/{topup}/{b0}'
                os.makedirs(outDir, exist_ok=True)
                max = float(os.popen(f'fslstats {pathTmean} -k {corticalMask} -p 100').read())
                min = float(os.popen(f'fslstats {pathTmean} -k {corticalMask} -p 0').read())
                histData = [float(x) for x in os.popen(f'fslstats {pathTmean} -k {corticalMask} -H 100 {min} {max}').read().split()]
                x_pos = np.linspace(min, max, len(histData))
                fig, ax = plt.subplots()
                ax.bar(x_pos, histData, width = (max-min)/100)
                ax.set_title(f'{subject}, {scanType}, {topup}, {b0} correction')
                plt.xlabel('MR intensity')
                plt.ylabel('voxel count')
                ax.yaxis.grid(True)
                plt.xlim(-10000, 200000)
                plt.tight_layout()
                plt.savefig(f'{outDir}/MRintensity_hist_{topup}_{b0}.png')
                plt.show()
                plt.close()

    # make scatterplot of tSNR values and signal intensity values in cortical mask
    for run, scanType in zip(['run01', 'run02'], ['classic', 'multiX']):
        for topup, topupString in zip(['topUp', 'noTopUp'], ['_topUp', '']):
            for b0, b0String in zip(['noB0'], ['']):  # zip(['b0', 'noB0'], ['_b0', '']):
                dataPath = glob.glob(os.path.join(sessDir, f'functional/resting_state/{run}/inputs/rawData{topupString}{b0String}.nii*'))[0]

                tSNRdir = os.path.join(os.getcwd(), os.path.dirname(os.path.dirname(dataPath)), f'outputs/{topup}/{b0}/tSNR')
                outDir = f'{dqDir}/tSNR/{subject}/{session}/cortex/{scanType}/{topup}/{b0}'

                pathTSNRanat = os.path.join(tSNRdir, 'tSNR.nii.gz')
                pathTmeanAnat = f'{tSNRdir}/Tmean.nii.gz'
                pathCorticalMask = f'{tSNRdir}/cortex_bi.nii.gz'

                # apply mask
                pathTSNRcortex = os.path.join(tSNRdir, 'tSNR_cortex.nii.gz')
                pathTmeanCortex = f'{tSNRdir}/Tmean_cortex.nii.gz'
                os.system(f'fslmaths {pathTSNRanat} -mul {pathCorticalMask} {pathTSNRcortex}')
                os.system(f'fslmaths {pathTmeanAnat} -mul {pathCorticalMask} {pathTmeanCortex}')

                # load masked data
                dataTmean = nib.load(pathTmeanCortex).get_fdata().flatten()
                dataTmean = np.ma.masked_where(dataTmean < MRthresh, dataTmean)
                dataTSNR = nib.load(pathTSNRcortex).get_fdata().flatten()
                dataTSNR = np.ma.masked_where(dataTmean < MRthresh, dataTSNR)
                rVal = np.corrcoef(dataTSNR, dataTmean)[0,1]

                # plot
                fig, ax = plt.subplots()
                ax.scatter(dataTmean, dataTSNR, s=.01)
                ax.set_title(f'R = {rVal:.2f}, {topup}, {b0} correction')
                plt.xlabel('signal intensity')
                plt.ylabel('tSNR')
                plt.xlim(-10000, 200000)
                plt.ylim(-10, 150)
                plt.tight_layout()
                plt.savefig(f'{outDir}/tSNR_v_sigInt_{topup}_{b0}.png')
                plt.show()
                plt.close()


    # get difference in tSNR and MR intensity and plot on surface
    session = list(experiment['scanInfo'][subject].keys())[0]
    sessDir = f'{version}/data/individual/{subject}/{session}'

    for topup, topupString in zip(['topUp', 'noTopUp'], ['_topUp', '']):
        for b0, b0String in zip(['noB0'], ['']):  # zip(['b0', 'noB0'], ['_b0', '']):

            tSNRdir = os.path.join(sessDir, f'functional/resting_state/run02-run01/{topup}/{b0}')

            if os.path.isdir(f'{sessDir}/functional/resting_state/run01/outputs/{topup}/{b0}/tSNR'):
                os.makedirs(tSNRdir, exist_ok=True)
                func2surf = f'{sessDir}/reg/func2surf{topupString}{b0String}.dat'
                funcPath = f'{sessDir}/reg/refFunc{topupString}{b0String}.nii.gz'

                if not os.path.exists(func2surf):
                    freesurferCommand = f'bbregister --s {subject} --mov {funcPath} --init-fsl --reg {func2surf} --bold'
                    os.system(f'bash /mnt/HDD12TB/masterScripts/fMRI/callFreesurferFunction.sh -s "{freesurferCommand}"')

                for analysis, input, thr, uthr in zip(['tSNR', 'MRsignal'], ['tSNR','Tmean'], [0.1,0.1], [25,80000]):

                    # get difference ratio between classic and multix
                    classic = f'{sessDir}/functional/resting_state/run01/outputs/{topup}/{b0}/tSNR/{input}.nii.gz'
                    multiX = f'{sessDir}/functional/resting_state/run02/outputs/{topup}/{b0}/tSNR/{input}.nii.gz'

                    assert os.path.isfile(classic) and os.path.isfile(multiX)

                    rawDifference = f'{tSNRdir}/{input}_raw.nii.gz'
                    outFile = f'{tSNRdir}/{input}.nii.gz'
                    if not os.path.isfile(outFile):
                        os.system(f'fslmaths {multiX} -sub {classic} {rawDifference}')
                        os.system(f'fslmaths {multiX} -div {classic} -mul 100 -sub 100 {outFile}')
                    inFile = outFile

                    # convert to surface file
                    for hemi in ['lh', 'rh']:

                        # convert volume to surface
                        surfFile = f'{inFile[:-7]}_{hemi}.mgh'
                        if not os.path.isfile(surfFile):
                            freesurferCommand = f'mri_vol2surf --mov {inFile} --out {surfFile} --reg {func2surf} --hemi {hemi} --interp trilinear'
                            os.system(f'bash /mnt/HDD12TB/masterScripts/fMRI/callFreesurferFunction.sh -s "{freesurferCommand}"')

                        for view in ['posterior', 'inferior', 'lateral']:
                            resDir = f'{dqDir}/tSNR/{subject}/{session}/cortex/multiXversusClassic/{topup}/{b0}/{analysis}'
                            os.makedirs(resDir, exist_ok=True)
                            imageFile = os.path.join(resDir, f'{view}_{hemi}.png')

                            inflatedSurface = f'/mnt/HDD12TB/freesurfer/subjects/{subject}/surf/{hemi}.inflated'

                            freesurferCommand = f'freeview -f {inflatedSurface}:curvature_method=binary:overlay={surfFile}:overlay_color=colourwheel:overlay_threshold={0.1},{100} -layout 1 -viewport 3d -view {view} -ss {imageFile} 1 autotrim'
                            #print(freesurferCommand)
                            #os.system(f'bash /mnt/HDD12TB/masterScripts/fMRI/callFreesurferFunction.sh -s "{freesurferCommand}"')

                # scatter plot of improved tSNR v improved MR intensity
                # apply mask
                pathCortex = f'{sessDir}/functional/resting_state/run01/outputs/{topup}/{b0}/tSNR/cortex_bi.nii.gz'
                pathTmean = f'{tSNRdir}/Tmean_raw.nii.gz'
                pathTSNR = f'{tSNRdir}/tSNR_raw.nii.gz'
                pathTmeanCortex = f'{tSNRdir}/Tmean_cortex.nii.gz'
                pathTSNRcortex = f'{tSNRdir}/tSNR_cortex.nii.gz'

                os.system(f'fslmaths {pathTmean} -mul {pathCortex} {pathTmeanCortex}')
                os.system(f'fslmaths {pathTSNR} -mul {pathCortex} {pathTSNRcortex}')

                # load masked data
                dataTmean = nib.load(pathTmeanCortex).get_fdata().flatten()
                dataTmean = np.ma.masked_where(dataTmean == 0, dataTmean)
                dataTSNR = nib.load(pathTSNRcortex).get_fdata().flatten()
                dataTSNR = np.ma.masked_where(dataTmean == 0, dataTSNR)
                rVal = np.corrcoef(dataTSNR, dataTmean)[0, 1]

                # plot
                fig, ax = plt.subplots(figsize=(5,5))
                a, b = np.polyfit(dataTmean, dataTSNR, 1)
                ax.scatter(dataTmean, dataTSNR, s=.01)
                ax.plot(dataTmean, dataTmean * a + b, color='black')
                ax.set_title(f'R = {rVal:.2f}, slope = {a:.2f},\nintercept = {b:.2f}, {topup}')
                plt.xlabel(f'signal intensity')
                plt.ylabel(f'tSNR')
                plt.xlim(-100000, 200000)
                plt.ylim(-40, 80)
                ax.set_aspect(1. / ax.get_data_ratio(), adjustable='box')
                plt.tight_layout()
                resDir = f'{dqDir}/tSNR/{subject}/{session}/cortex/multiXversusClassic/{topup}/{b0}'
                os.makedirs(resDir, exist_ok=True)
                plt.savefig(f'{resDir}/tSNR_v_sigInt_{topup}_{b0}.png')
                plt.show()
                plt.close()

                # scatter plots of classic v multiX for tSNR and MR intensity
                for analysis, input, lims in zip(['tSNR', 'signal intensity'], ['tSNR', 'Tmean'], [[0,150],[0,300000]]):

                    pathClassic = f'{sessDir}/functional/resting_state/run01/outputs/{topup}/{b0}/tSNR/{input}_cortex.nii.gz'
                    pathMultiX = f'{sessDir}/functional/resting_state/run02/outputs/{topup}/{b0}/tSNR/{input}_cortex.nii.gz'

                    # load masked data
                    dataClassic = nib.load(pathClassic).get_fdata().flatten()
                    dataClassic = np.ma.masked_where(dataClassic == 0, dataClassic)
                    dataMultiX = nib.load(pathMultiX).get_fdata().flatten()
                    dataMultix = np.ma.masked_where(dataClassic == 0, dataMultiX)
                    rVal = np.corrcoef(dataClassic, dataMultiX)[0, 1]

                    # plot
                    fig, ax = plt.subplots(figsize=(5,5))
                    a, b = np.polyfit(dataClassic, dataMultiX, 1)
                    ax.scatter(dataClassic, dataMultiX, s=.01)
                    ax.plot(dataClassic, dataClassic*a+b, color='black')
                    ax.set_aspect('equal', adjustable='box')
                    ax.set_title(f'R = {rVal:.2f}, slope = {a:.2f},\nintercept = {b:.2f}, {topup}')
                    plt.ylabel(f'{analysis}, MultiX')
                    plt.xlabel(f'{analysis}, Classic')
                    plt.xlim(lims)
                    plt.ylim(lims)
                    plt.axline((0, 0), (1, 1), linewidth=2, color='red')
                    plt.tight_layout()
                    resDir = f'{dqDir}/tSNR/{subject}/{session}/cortex/multiXversusClassic/{topup}/{b0}'
                    os.makedirs(resDir, exist_ok=True)
                    plt.savefig(f'{resDir}/multiX_v_classic_{analysis}_{topup}_{b0}.png')
                    plt.show()
                    plt.close()


