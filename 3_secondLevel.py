'''
takes second level design files (combining analyses across runs) made prior to this script, edits and submits to feat
'''

import os
import glob
import datetime
import numpy as np
import random
import shutil
import time
# time.sleep(3600)

# get scan info from experiment file
from fMRI.analysis.scripts.experiment import experiment

for topup in ['topUp','noTopUp']:
    # for b0 in ['noB0','b0']:
    for b0 in ['noB0']:
        for HRFmodel in ['doubleGamma']:  # , 'singleGamma']:
            for subject in experiment['scanInfo']:
                for scan in experiment['design']:
                    if (scan != 'resting_state') & (scan != 'FHO_loc') & (scan != 'checkerboard_loc'): #& (scan != 'object_noise_mseq')& (scan != 'object_noise')
                        runDirsAllSessions = []

                        #################################
                        # across runs within each session
                        #################################

                        for session in experiment['scanInfo'][subject].keys():

                            print(f'\n{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | Subject: {subject} | Session: {session} | Scan: {scan} | topup: {topup} | b0: {b0} | HRF model: {HRFmodel}')

                            # get list of runs for first subject (only used to identify string patterns in design file to replace later)
                            # Note: second session is used as first session for M012 contained one run of each scan
                            runDirsTemplate = sorted(glob.glob(os.path.join(f'/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/individual/F019/230516/functional', scan, 'run*/outputs/topUp/noB0/doubleGamma/firstLevel.feat')))
                            nRunsTemplate = len(runDirsTemplate)

                            # compile list of runs for this subject
                            runDirs = sorted(glob.glob(os.path.join(f'/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/individual', subject, session, 'functional', scan, 'run*/outputs', topup, b0, HRFmodel, 'firstLevel.feat')))
                            nRuns = len(runDirs)
                            runDirsAllSessions += runDirs

                            if nRuns > 1:

                                outDir = os.path.join(f'/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/individual', subject, session, 'functional', scan, 'allRuns', topup, b0, HRFmodel, 'secondLevel.gfeat')
                                os.makedirs(os.path.dirname(outDir), exist_ok=True)

                                designFile = os.path.join(f'/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/designs/secondLevel', scan, 'design.fsf')

                                with open(designFile, 'r') as file:
                                    fileData = file.read()

                                # replace subject ID
                                fileData = fileData.replace('M015', subject)

                                # replace session
                                fileData = fileData.replace('230830', session)

                                # replace topup and b0 types for input field and output field
                                fileData = fileData.replace('topUp', topup)
                                fileData = fileData.replace('noB0', b0)

                                # replace HRFmodel
                                fileData = fileData.replace('doubleGamma', HRFmodel)

                                # replace number of runs
                                fileData = fileData.replace(f'set fmri(npts)', '#')
                                fileData = fileData.replace(f'set fmri(multiple)', '#')
                                fileData += f'\nset fmri(npts) {nRuns}'
                                fileData += f'\nset fmri(multiple) {nRuns}'

                                # replace input analysis directories
                                fileData = fileData.replace('set feat_files(', '#')
                                for r, runDir in enumerate(runDirs):
                                    fileData += f'\nset fmri(evg{r + 1}.1) 1.0'
                                    fileData += f'\nset fmri(groupmem.{r + 1}) 1'
                                    fileData += f'\nset feat_files({r + 1}) "{os.path.join(os.getcwd(), runDir)}"'

                                # check out dir in design file is correct
                                lines = fileData.splitlines()
                                for line in lines:
                                    if line.startswith(f'set fmri(outputdir)'):
                                        actualOutDir = line.split()[2][1:-1]
                                        if not actualOutDir == os.path.join(os.getcwd(), outDir):
                                            print(os.path.join(os.getcwd(), outDir))
                                            print(actualOutDir)
                                            raise Exception(f'Destination directory does not match, check feat analysis at {outDir}')

                                # write the file out again
                                designFileTemp = f'{designFile[:-4]}_temp.fsf'
                                with open(designFileTemp, 'w') as file:
                                    file.write(fileData)
                                file.close()

                                # run
                                if not os.path.isfile(os.path.join(outDir, 'done')):
                                    print('Failed to find complete analysis, analysing...')

                                    # remove incomplete analysis, if it exists
                                    try:
                                        shutil.rmtree(outDir)
                                    except:
                                        pass

                                    # run analysis
                                    os.system(f'feat {designFileTemp}')

                                    # check that analysis has completed successfully (see if first cope file poduced)
                                    if not os.path.isfile(os.path.join(outDir, 'cope1.feat/stats/cope1.nii.gz')):
                                        raise Exception(f'No cope files produced, check feat analysis at {outDir}')

                                    # place 'done' file in folder
                                    with open(os.path.join(outDir, 'done'), 'w') as fp:
                                        pass
                                else:
                                    print('Analysis already performed, skipping...')
                            else:
                                print('Only a single run found for this session, skipping...')

                        ##################################
                        # across all runs for this subject
                        ##################################

                        nRunsAllSessions = len(runDirsAllSessions)
                        sessions = list(experiment['scanInfo'][subject].keys())
                        if len(sessions) > 1:

                            print(f'\n{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | Subject: {subject} | Session: all sessions | Scan: {scan} | topup: {topup} | b0: {b0} | HRF model: {HRFmodel}')

                            outDir = os.path.join(f'/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/individual', subject, 'allSessions', 'functional', scan, 'allRuns', topup, b0, HRFmodel, 'secondLevel.gfeat')
                            os.makedirs(os.path.dirname(outDir), exist_ok=True)

                            designFile = os.path.join(f'/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/designs/secondLevel', scan, 'design.fsf')

                            with open(designFile, 'r') as file:
                                fileData = file.read()


                            # replace subject ID
                            fileData = fileData.replace('M012', subject)

                            # replace topup and b0 types for input field and output field
                            fileData = fileData.replace('topUp', topup)
                            fileData = fileData.replace('noB0', b0)

                            # replace HRFmodel
                            fileData = fileData.replace('doubleGamma', HRFmodel)

                            # replace number of runs
                            fileData = fileData.replace(f'\nset fmri(npts)', '#')
                            fileData = fileData.replace(f'\nset fmri(multiple)', '#')
                            fileData += f'\nset fmri(npts) {nRunsAllSessions}'
                            fileData += f'\nset fmri(multiple) {nRunsAllSessions}'

                            # replace input analysis directories
                            fileData = fileData.replace('set feat_files', '#')
                            for r, runDir in enumerate(runDirsAllSessions):
                                fileData += f'\nset fmri(evg{r + 1}.1) 1.0'
                                fileData += f'\nset fmri(groupmem.{r + 1}) 1'
                                fileData += f'\nset feat_files({r + 1}) "{os.path.join(os.getcwd(), runDir)}"'

                            # replace outDir
                            fileData = fileData.replace('set fmri(outputdir)', '#')
                            fileData += f'\nset fmri(outputdir) "{os.path.join(os.getcwd(), outDir)}"'

                            # check out dir in design file is correct
                            lines = fileData.splitlines()
                            for line in lines:
                                if line.startswith(f'set fmri(outputdir)'):
                                    actualOutDir = line.split()[2][1:-1]
                                    if not actualOutDir == os.path.join(os.getcwd(), outDir):
                                        print(os.path.join(os.getcwd(), outDir))
                                        print(actualOutDir)
                                        raise Exception(f'Destination directory does not match, check feat analysis at {outDir}')

                            # write the file out again
                            designFileTemp = f'{designFile[:-4]}_temp.fsf'
                            with open(designFileTemp, 'w') as file:
                                file.write(fileData)
                            file.close()

                            if not os.path.isfile(os.path.join(outDir, 'done')):

                                print('Failed to find complete analysis, analysing...')

                                # remove incomplete analysis, if it exists
                                try:
                                    shutil.rmtree(outDir)
                                except:
                                    pass

                                # run analysis
                                os.system(f'feat {designFileTemp}')

                                # check that analysis has completed successfully (see if first cope file poduced)
                                if not os.path.isfile(os.path.join(outDir, 'cope1.feat/stats/cope1.nii.gz')):
                                    raise Exception(f'No cope files produced, check feat analysis at {outDir}')

                                # place 'done' file in folder
                                with open(os.path.join(outDir, 'done'), 'w') as fp:
                                    pass

                            else:
                                print('Analysis already performed, skipping...')



                        if scan.startswith('occlusion'):

                            '''
                            ##########
                            # run LORO
                            ##########
    
                            for r, runDir in enumerate(runDirsAllSessions):
    
                                print(f'\n{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | Subject: {subject} | Scan: {scan} | LORO: {r+1}/{nRunsAllSessions} | topup: {topup} | b0: {b0} | HRF model: {HRFmodel}')
    
                                if len(sessions) == 1:
                                    outDirLORO = os.path.join(f'fMRI/{version}/data/individual', subject, session, 'functional', scan, f'allRunsBut{r + 1:02}', topup, b0, HRFmodel, 'secondLevel.gfeat')
                                else:
                                    outDirLORO = os.path.join(f'fMRI/{version}/data/individual', subject, 'allSessions', 'functional', scan, f'allRunsBut{r + 1:02}', topup, b0, HRFmodel, 'secondLevel.gfeat')
    
                                # replace outDir
                                LOROfileData = fileData.replace('set fmri(outputdir)', '#')
                                LOROfileData += f'\nset fmri(outputdir) "{os.path.join(os.getcwd(), outDirLORO)}"'
    
                                # check out dir in design file is correct
                                lines = LOROfileData.splitlines()
                                for line in lines:
                                    if line.startswith(f'set fmri(outputdir)'):
                                        actualOutDir = line.split()[2][1:-1]
                                        if not actualOutDir == os.path.join(os.getcwd(), outDirLORO):
                                            print(os.path.join(os.getcwd(), outDirLORO))
                                            print(actualOutDir)
                                            raise Exception(f'Destination directory does not match, check feat analysis at {outDirLORO}')
    
                                # replace number of runs
                                LOROfileData = LOROfileData.replace(f'set fmri(npts)', '#')
                                LOROfileData = LOROfileData.replace(f'set fmri(multiple)', '#')
                                LOROfileData += f'\nset fmri(npts) {nRunsAllSessions-1}'  # set number of runs
                                LOROfileData += f'\nset fmri(multiple) {nRunsAllSessions-1}'  # different nRuns
    
                                # replace run dirs
                                LOROfileData = LOROfileData.replace(f'set feat_files', '#')
                                counter = 1
                                for rAll, runDirAll in enumerate(runDirsAllSessions):
                                    if rAll != r:
                                        LOROfileData += f'\nset fmri(evg{counter}.1) 1.0'
                                        LOROfileData += f'\nset fmri(groupmem.{counter}) 1'
                                        LOROfileData += f'\nset feat_files({counter}) "{os.path.join(os.getcwd(), runDirAll)}"'
                                        counter += 1
    
    
                                # write the file out again
                                designFileTemp = f'{designFile[:-4]}_temp.fsf'
                                with open(designFileTemp, 'w') as file:
                                    file.write(LOROfileData)
                                file.close()
    
                                if not os.path.isfile(os.path.join(outDirLORO, 'done')):
    
                                    print('Failed to find complete analysis, analysing...')
    
                                    # remove incomplete analysis, if it exists
                                    try:
                                        shutil.rmtree(outDirLORO)
                                    except:
                                        pass
    
                                    # run analysis
                                    os.system(f'feat {designFileTemp}')
    
                                    # check that analysis has completed successfully (see if first cope file poduced)
                                    if not os.path.isfile(os.path.join(outDirLORO, 'cope1.feat/stats/cope1.nii.gz')):
                                        raise Exception(f'No cope files produced, check feat analysis at {outDirLORO}')
    
                                    # place 'done' file in folder
                                    with open(os.path.join(outDirLORO, 'done'), 'w') as fp:
                                        pass
    
                                else:
    
                                    print('Analysis already performed, skipping...')
                            
    
                            ################
                            # run split half
                            ################
    
                            nSplits = 10
                            nRunsAllSessions = len(runDirsAllSessions)
                            for split in range(nSplits):
    
                                runSamples = np.array(random.sample(list(np.arange(nRunsAllSessions)), nRunsAllSessions))
                                sampleSplit = [np.arange(int(len(runSamples)/2)), np.arange(int(len(runSamples)/2), len(runSamples))]
    
                                for s, side in enumerate(['A','B']):
    
                                    print(f'\n{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | Subject: {subject} | Scan: {scan} | Split Half: {split}/{side} | topup: {topup} | b0: {b0} | HRF model: {HRFmodel}')
    
                                    inRuns = runSamples[sampleSplit[s]]
    
                                    if len(sessions) == 1:
                                        outDirSplitHalf = os.path.join(f'fMRI/{version}/data/individual', subject, session, 'functional', scan, f'splitHalf{split}{side}', topup, b0, HRFmodel, 'secondLevel.gfeat')
                                    else:
                                        outDirSplitHalf = os.path.join(f'fMRI/{version}/data/individual', subject, 'allSessions', 'functional', scan, f'splitHalf{split}{side}', topup, b0, HRFmodel, 'secondLevel.gfeat')
    
                                    # replace outDir
                                    splitHalfFileData = fileData.replace('set fmri(outputdir)', '#')
                                    splitHalfFileData += f'\nset fmri(outputdir) "{os.path.join(os.getcwd(), outDirSplitHalf)}"'
    
                                    # replace number of runs
                                    splitHalfFileData = splitHalfFileData.replace(f'set fmri(npts)', '#')
                                    splitHalfFileData = splitHalfFileData.replace(f'set fmri(multiple)', '#')
                                    splitHalfFileData += f'\nset fmri(npts) {len(inRuns)}'
                                    splitHalfFileData += f'\nset fmri(multiple) {len(inRuns)}'
    
                                    # replace run dirs
                                    splitHalfFileData = splitHalfFileData.replace(f'set feat_files', '#')
                                    for r, run in enumerate(inRuns):
                                        splitHalfFileData += f'\nset fmri(evg{r + 1}.1) 1.0'
                                        splitHalfFileData += f'\nset fmri(groupmem.{r + 1}) 1'
                                        splitHalfFileData += f'\nset feat_files({r + 1}) "{os.path.join(os.getcwd(), runDirsAllSessions[run])}"'
    
                                    # check out dir in design file is correct
                                    lines = splitHalfFileData.splitlines()
                                    for line in lines:
                                        if line.startswith(f'set fmri(outputdir)'):
                                            actualOutDir = line.split()[2][1:-1]
                                            if not actualOutDir == os.path.join(os.getcwd(), outDirSplitHalf):
                                                print(os.path.join(os.getcwd(), outDirSplitHalf))
                                                print(actualOutDir)
                                                raise Exception(f'Destination directory does not match, check feat analysis at {outDir}')
    
                                    # write the file out again
                                    designFileTemp = f'{designFile[:-4]}_temp.fsf'
                                    with open(designFileTemp, 'w') as file:
                                        file.write(splitHalfFileData)
                                    file.close()
    
                                    if not os.path.isfile(os.path.join(outDirSplitHalf, 'done')):
    
                                        print('Failed to find complete analysis, analysing...')
    
                                        # remove incomplete analysis, if it exists
                                        try:
                                            shutil.rmtree(outDirSplitHalf)
                                        except:
                                            pass
    
                                        # run analysis
                                        os.system(f'feat {designFileTemp}')
    
                                        # check that analysis has completed successfully (see if first cope file poduced)
                                        if not os.path.isfile(os.path.join(outDirSplitHalf, 'cope1.feat/stats/cope1.nii.gz')):
                                            raise Exception(f'No cope files produced, check feat analysis at {outDirSplitHalf}')
    
                                        # place 'done' file in folder
                                        with open(os.path.join(outDirSplitHalf, 'done'), 'w') as fp:
                                            pass
    
                                    else:
    
                                        print('Analysis already performed, skipping...')
                            '''
print('Done')
