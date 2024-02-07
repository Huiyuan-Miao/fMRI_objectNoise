#!/usr/bin/python
'''
takes first level design files (each run) made prior to this script, edits and submits to feat
'''

import os
import datetime
import time
#time.sleep(10000)

# get scan info from experiment file
from fMRI.analysis.scripts.experiment import experiment

# first level analysis
for topup, topupString in zip(['topUp', 'noTopUp'],['_topUp','']):
    # for b0, b0String in zip(['noB0', 'b0'],['', '_b0']):
    for b0, b0String in zip(['noB0'],['']):
        for HRFmodel in ['doubleGamma']:#,'singleGamma']:
            for subject in experiment['scanInfo'].keys():
                for session in experiment['scanInfo'][subject].keys():
                    for scan in experiment['design'].keys():
                        if (scan == 'checkerboard_loc')|(scan == 'object_noise_mseq')|(scan == 'object_noise'):#(scan != 'resting_state') & (scan != 'FHO_loc'): #& (scan != 'object_noise_mseq')& (scan != 'object_noise')
                            for r, run in enumerate(experiment['scanInfo'][subject][session]['funcScans'][scan]):

                                print(f'\n{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | Subject: {subject} | Session: {session} | Scan: {scan} | Run: {r+1} | Top Up: {topup} |  B0: {b0} | HRF model: {HRFmodel}')

                                outDir = os.path.join(f'/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/individual', subject, session, 'functional', scan, f'run{r+1:02}/outputs', topup, b0, HRFmodel, 'firstLevel.feat')

                                if not os.path.isdir(outDir):
                                    print('Analysis not found, analysing...')

                                    # replace relevant settings in design file. The original should be set up for the first run of the first subject,
                                    # and the files should be set up so that all that needs changing is the subject ID and the run number
                                    designFile = os.path.join(f'/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/designs/firstLevel', scan, HRFmodel, 'design.fsf')
                                    with open(designFile, 'r') as file:
                                        fileData = file.read()

                                    # replace topup and b0 types for input field and output field
                                    fileData = fileData.replace('rawData_topUp', f'rawData{topupString}{b0String}')
                                    fileData = fileData.replace('refFunc_topUp', f'refFunc{topupString}{b0String}')
                                    fileData = fileData.replace('topUp', topup)
                                    fileData = fileData.replace('noB0', b0)

                                    # replace subject ID
                                    fileData = fileData.replace('debug', subject)

                                    # replace session
                                    fileData = fileData.replace('230829', session) # this date is from v1 but its ok

                                    # replace run number
                                    fileData = fileData.replace('run01', f'run{r+1:02}')

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

                                    # run analysis
                                    os.system(f'feat {designFileTemp}')

                                    # check that analysis has completed successfully
                                    if not os.path.isfile(os.path.join(outDir, 'stats/cope1.nii.gz')):
                                        raise Exception(f'No cope files produced, check feat analysis at {outDir}')

                                else:
                                    if time.time() - os.stat(outDir).st_ctime > 1800: # if created over 30 mins ago

                                        # check that analysis has completed successfully
                                        if not os.path.isfile(os.path.join(outDir, 'stats/cope1.nii.gz')):
                                            raise Exception(f'No cope files found in previously completed feat dir, check feat analysis at {outDir}')
                                        else:
                                            print('Analysis with cope files found, skipping...')
                                    else:
                                        print('Analysis started < 30 mins ago, assumed still performing, skipping...')

print('Done')
