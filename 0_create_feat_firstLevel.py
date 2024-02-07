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
import numpy as np
# first level analysis
nCond = 32
scanName = 'object_noise_mseq'
designFile = os.path.join(f'/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/designs/firstLevel/object_noise_mseq/doubleGamma/design_base.fsf')
with open(designFile, 'r') as file:
    fileData = file.read()

fileData = fileData.replace('set fmri(evs_orig) 48', 'set fmri(evs_orig) ' + str(nCond))
fileData = fileData.replace('set fmri(evs_real) 96', 'set fmri(evs_real) ' + str(nCond*2))
fileData = fileData.replace('set fmri(ncon_orig) 49', 'set fmri(ncon_orig) ' + str(nCond+1))
fileData = fileData.replace('set fmri(ncon_real) 49', 'set fmri(ncon_real) ' + str(nCond+1))
fileData = fileData.replace('/object_noise/', '/'+scanName+'/')
fileData = fileData.replace('126', '141')
fileData = fileData.replace('252', '282')
# write the file out again
designFileTemp = f'{designFile[:-9]}.fsf'
with open(designFileTemp, 'w') as file:
    file.write(fileData)
    for i in range(nCond):
        file.write('# EV %i title' % (i+1))
        file.write('\n')
        file.write('set fmri(evtitle%i) "%i"' % (i + 1,i))
        file.write('\n')
        file.write('\n')
        file.write('# Basic waveform shape (EV %i)' % (i+1))
        file.write('\n')
        file.write('# 0 : Square')
        file.write('\n')
        file.write('# 1 : Sinusoid')
        file.write('\n')
        file.write('# 2 : Custom (1 entry per volume)')
        file.write('\n')
        file.write('# 3 : Custom (3 column format)')
        file.write('\n')
        file.write('# 4 : Interaction')
        file.write('\n')
        file.write('# 10 : Empty (all zeros)')
        file.write('\n')
        file.write('set fmri(shape%i) 3' % (i+1))
        file.write('\n')
        file.write('\n')
        file.write('# Convolution (EV %i)' % (i + 1))
        file.write('\n')
        file.write('# 0 : None')
        file.write('\n')
        file.write('# 1 : Gaussian')
        file.write('\n')
        file.write('# 2 : Gamma')
        file.write('\n')
        file.write('# 3 : Double-Gamma HRF')
        file.write('\n')
        file.write('# 4 : Gamma basis functions')
        file.write('\n')
        file.write('# 5 : Sine basis functions')
        file.write('\n')
        file.write('# 6 : FIR basis functions')
        file.write('\n')
        file.write('# 8 : Alternate Double-Gamma')
        file.write('\n')
        file.write('set fmri(convolve%i) 3' % (i + 1))
        file.write('\n')
        file.write('\n')
        file.write('# Convolve phase (EV %i)' % (i + 1))
        file.write('\n')
        file.write('set fmri(convolve_phase%i) 0' % (i + 1))
        file.write('\n')
        file.write('\n')
        file.write('# Apply temporal filtering (EV %i)' % (i + 1))
        file.write('\n')
        file.write('set fmri(tempfilt_yn%i) 1' % (i + 1))
        file.write('\n')
        file.write('\n')
        file.write('# Add temporal derivative (EV %i)' % (i + 1))
        file.write('\n')
        file.write('set fmri(deriv_yn%i) 1' % (i + 1))
        file.write('\n')
        file.write('\n')
        file.write('# Custom EV file (EV %i)' % (i + 1))
        file.write('\n')
        file.write('set fmri(custom%i) "/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/individual/debug/230829/events/%s/3column/run01/%i.txt"' % (i+1,scanName,i))
        file.write('\n')
        file.write('\n')
        for j in range(nCond+1):
            file.write('# Orthogonalise EV %i wrt EV %i' % (i + 1,j))
            file.write('\n')
            file.write('set fmri(ortho%i.%i) 0' % (i + 1, j))
            file.write('\n')
            file.write('\n')
    file.write('# Contrast & F-tests mode')
    file.write('\n')
    file.write('# real : control real EVs')
    file.write('\n')
    file.write('# orig : control original EVs')
    file.write('\n')
    file.write('set fmri(con_mode_old) orig')
    file.write('\n')
    file.write('set fmri(con_mode) orig')
    file.write('\n')
    file.write('\n')
    for i in range(nCond):
        file.write('# Display images for contrast_real %i' % (i + 1))
        file.write('\n')
        file.write('set fmri(conpic_real.%i) 1' % (i + 1))
        file.write('\n')
        file.write('\n')
        file.write('# Title for contrast_real %i' % (i + 1))
        file.write('\n')
        file.write('set fmri(conname_real.%i) "%i"' % (i + 1,i))
        file.write('\n')
        file.write('\n')
        for j in range(nCond*2):
            file.write('# Real contrast_real vector %i element %i' % (i + 1,j+1))
            file.write('\n')
            if (j+1) == (i+1)*2-1:
                file.write('set fmri(con_real%i.%i) 1.0' % (i + 1, j+1))
            else:
                file.write('set fmri(con_real%i.%i) 0' % (i + 1, j + 1))
            file.write('\n')
            file.write('\n')
    file.write('# Display images for contrast_real %i' % (nCond + 1))
    file.write('\n')
    file.write('set fmri(conpic_real.%i) 1' % (nCond + 1))
    file.write('\n')
    file.write('\n')
    file.write('# Title for contrast_real %i' % (nCond + 1))
    file.write('\n')
    file.write('set fmri(conname_real.%i) "allConds"' % (nCond + 1))
    file.write('\n')
    file.write('\n')
    for j in range(nCond * 2):
        file.write('# Real contrast_real vector %i element %i' % (nCond + 1, j + 1))
        file.write('\n')
        if np.mod(j,2) ==0:
            file.write('set fmri(con_real%i.%i) 1.0' % (nCond + 1, j + 1))
        else:
            file.write('set fmri(con_real%i.%i) 0' % (nCond + 1, j + 1))
        file.write('\n')
        file.write('\n')
    for i in range(nCond):
        file.write('# Display images for contrast_orig %i' % (i + 1))
        file.write('\n')
        file.write('set fmri(conpic_orig.%i) 1' % (i + 1))
        file.write('\n')
        file.write('\n')
        file.write('# Title for contrast_orig %i' % (i + 1))
        file.write('\n')
        file.write('set fmri(conname_orig.%i) "%i"' % (i + 1,i))
        file.write('\n')
        file.write('\n')
        for j in range(nCond):
            file.write('# Real contrast_orig vector %i element %i' % (i + 1,j+1))
            file.write('\n')
            if (j+1) == (i+1):
                file.write('set fmri(con_orig%i.%i) 1.0' % (i + 1, j+1))
            else:
                file.write('set fmri(con_orig%i.%i) 0' % (i + 1, j + 1))
            file.write('\n')
            file.write('\n')
    file.write('# Display images for contrast_orig %i' % (nCond + 1))
    file.write('\n')
    file.write('set fmri(conpic_orig.%i) 1' % (nCond + 1))
    file.write('\n')
    file.write('\n')
    file.write('# Title for contrast_orig %i' % (nCond + 1))
    file.write('\n')
    file.write('set fmri(conname_orig.%i) "allConds"' % (nCond + 1))
    file.write('\n')
    file.write('\n')
    for j in range(nCond):
        file.write('# Real contrast_orig vector %i element %i' % (nCond + 1, j + 1))
        file.write('\n')
        file.write('set fmri(con_orig%i.%i) 1.0' % (nCond + 1, j + 1))
        file.write('\n')
        file.write('\n')
    file.write('# Contrast masking - use >0 instead of thresholding?')
    file.write('\n')
    file.write('set fmri(conmask_zerothresh_yn) 0')
    file.write('\n')
    file.write('\n')
    for i in range(nCond+1):
        for j in range(nCond+1):
            if i != j:
                file.write('# Mask real contrast/F-test %i with real contrast/F-test %i?' % (i+1, j+1))
                file.write('\n')
                file.write('set fmri(conmask%i_%i) 0' % (i+1, j+1))
                file.write('\n')
                file.write('\n')
    file.write('# Do contrast masking at all?')
    file.write('\n')
    file.write('set fmri(conmask1_1) 0')
    file.write('\n')
    file.write('\n')
    file.write('##########################################################')
    file.write('\n')
    file.write('# Now options that don\'t appear in the GUI')
    file.write('\n')
    file.write('\n')
    file.write('# Alternative (to BETting) mask image')
    file.write('\n')
    file.write('set fmri(alternative_mask) ""')
    file.write('\n')
    file.write('\n')
    file.write('# Initial structural space registration initialisation transform')
    file.write('\n')
    file.write('set fmri(init_initial_highres) ""')
    file.write('\n')
    file.write('\n')
    file.write('# Structural space registration initialisation transform')
    file.write('\n')
    file.write('set fmri(init_highres) ""')
    file.write('\n')
    file.write('\n')
    file.write('# Standard space registration initialisation transform')
    file.write('\n')
    file.write('set fmri(init_standard) ""')
    file.write('\n')
    file.write('\n')
    file.write('# For full FEAT analysis: overwrite existing .feat output dir?')
    file.write('\n')
    file.write('set fmri(overwrite_yn) 0')
    file.write('\n')
file.close()

print('Done')