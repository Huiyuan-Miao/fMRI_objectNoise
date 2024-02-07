import numpy as np
import os
import os.path as op
# 20230620
# in this version, the preprocessing is more and done with FSL
subject = 'F019'
session = '231020'
topupString = ''
topup = 'noTopUp'
reg_dir = '/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/individual/'+subject+'/temp_tsnr_restingState/'+session+'/black_background'
os.makedirs(reg_dir, exist_ok=True)

ref_func_Preprocessed = reg_dir+'/filtered_func_data'
# ref_func_Preprocessed = '/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/individual/M001/230823/functional/resting_state/run01/inputs/'+topup+'/noB0/doubleGamma/preprocessing.feat/filtered_func_data'
ref_func = reg_dir+'/rawData'

### remove the skull
overwrite = 0
#### std_anatomical = '/home/tonglab/fsl/data/standard/MNI152_T1_2mm_brain'
fs_dir = '/home/tonglab/freesurfer/subjects/'+subject
ref_anat = '/home/tonglab/freesurfer/subjects/'+subject+'/mri/orig/anatomical.nii'
ref_anat_brain = '/home/tonglab/freesurfer/subjects/'+subject+'/mri/orig/anatomical_brain.nii.gz'
if not op.isfile(ref_anat_brain):
    os.system(f'bet {ref_anat} {ref_anat_brain}')

### do preprocessing and average resting state in fsl and then preceed
designFile = os.path.join(f'/home/tonglab/Miao/fMRI/rapid_event_related_object_noise/fMRI/data/designs/preprocessing/resting_state/design.fsf')
with open(designFile, 'r') as file:
    fileData = file.read()
fileData = fileData.replace('rawData_topUp', f'rawData{topupString}')
fileData = fileData.replace('topUp', topup)

# replace subject ID
fileData = fileData.replace('M012', subject)

# replace session
fileData = fileData.replace('230823', session) # this date is from v1 but its ok

# write the file out again
designFileTemp = f'{designFile[:-4]}_temp.fsf'
with open(designFileTemp, 'w') as file:
    file.write(fileData)
file.close()

# run analysis
if not os.path.isfile(ref_func_Preprocessed+'.nii.gz'):
    os.system(f'feat {designFileTemp}')

os.system(f'fslmaths {ref_func_Preprocessed} -Tmean {ref_func}')

### from function to native anatomical
xform_func_to_anat = f'{reg_dir}/func_to_anat.mat'
if not op.isfile(xform_func_to_anat):
    os.system(
        f'epi_reg --epi={ref_func} --t1={ref_anat} --t1brain={ref_anat_brain} --out={xform_func_to_anat[:-4]}')  # BBR
xform_anat_to_func = f'{reg_dir}/anat_to_func.mat'
if not op.isfile(xform_anat_to_func):
    os.system(f'convert_xfm -omat {xform_anat_to_func} -inverse {xform_func_to_anat}')

### from native anatomical to standard anatomical
xform_std_to_anat = f'{fs_dir}/mri/transforms/reg.mni152.2mm.lta'
if not op.isfile(xform_std_to_anat):
    os.system(f'mni152reg --s {subject}')

### TSNR
tSNRdir = reg_dir +'/tsnr'
os.makedirs(tSNRdir,exist_ok = True)
pathTmean = os.path.join(tSNRdir, "Tmean.nii.gz")
pathTstd = os.path.join(tSNRdir, "Tstd.nii.gz")
pathTSNR = os.path.join(tSNRdir, "tSNR.nii.gz")
if not os.path.exists(pathTmean) or overwrite:
    print('Creating mean of timeseries...')
    os.system(f'fslmaths {ref_func_Preprocessed} -Tmean {pathTmean}')
if not os.path.exists(pathTstd) or overwrite:
    print('Creating std of timeseries...')
    os.system(f'fslmaths {ref_func_Preprocessed} -Tstd {pathTstd}')
if not os.path.exists(pathTSNR) or overwrite:
    print('Calculating tSNR map...')
    os.system(f'fslmaths {pathTmean} -div {pathTstd} {pathTSNR}')

resultsFile_name = tSNRdir + '/results_name_nonVisualRegion.csv'
resultsFile_mean = tSNRdir + '/results_mean_nonVisualRegion.csv'
resultsFile_std = tSNRdir + '/results_std_nonVisualRegion.csv'
resultsFile_number = tSNRdir + '/results_number_nonVisualRegion.csv'

### convert cortex mask
os.makedirs(reg_dir +'/anatomicalMask/',exist_ok=True)
os.makedirs(reg_dir +'/functionalMask/',exist_ok=True)
os.makedirs(reg_dir +'/functionalMaskInCortex/',exist_ok=True)

bilatCortexMask = f'{reg_dir}/functionalMask/cortex_bi.nii.gz'

if not os.path.isfile(bilatCortexMask) or overwrite:

    # convert freesurfer anat from mgz to nii
    fsAnatFileMGZ = f'/home/tonglab/freesurfer/subjects/{subject}/mri/orig.mgz'  # do not confuse with original nifti
    fsAnatFileNII = f'{reg_dir}/anatomical_fs.nii'
    freesurferCommand = f'mri_convert --in_type mgz --out_type nii --out_orientation RAS {fsAnatFileMGZ} {fsAnatFileNII}'
    # os.system(f'bash /home/tonglab/masterScripts/fMRI/callFreesurferFunction.sh -s "{freesurferCommand}"')
    os.system(freesurferCommand)
    # register freesurfer anat to orig anat (copy of anatomical saved by recon-all is not same as original nifti)
    origAnatFile = f'/home/tonglab/freesurfer/subjects/{subject}/mri/orig/anatomical.nii'
    fsAnat2origAnatMat = f'{reg_dir}/fsAnat2origAnat.mat'
    os.system(f'flirt -in {fsAnatFileNII} -ref {origAnatFile} -omat {fsAnat2origAnatMat}')

    # convert cortical mask to func space
    for hemi in ['lh', 'rh']:
        print(f'Converting {hemi} cortical mask...')

        # cortex mgz to nifti
        inFile = f'/home/tonglab/freesurfer/subjects/{subject}/mri/{hemi}.ribbon.mgz'
        outFile = f'{reg_dir}/cortex_{hemi}_FSanat.nii.gz'
        if not os.path.exists(outFile):
            freesurferCommand = f'mri_convert --in_type mgz --out_type nii --out_orientation RAS {inFile} {outFile}'
            # os.system(f'bash /home/tonglab/masterScripts/fMRI/callFreesurferFunction.sh -s "{freesurferCommand}"')
            os.system(freesurferCommand)
        # freesurfer anat space to orig anat space
        inFile = outFile
        outFile = f'{reg_dir}/anatomicalMask/cortex_{hemi}_origAnat.nii.gz'
        if not os.path.exists(outFile):
            os.system(f'flirt -in {inFile} -ref {origAnatFile} -applyxfm -init {fsAnat2origAnatMat} -interp nearestneighbour -out {outFile}')

        # orig anat space to func space
        # inFile = outFile
        # outFile = f'{reg_dir}/anatomicalMask/cortex_{hemi}.nii.gz'
        # os.system(
        #     f'mri_vol2vol --mov {inFile} --targ {ref_anat}.nii.gz --lta {xform_std_to_anat} --o {outFile} --nearest')
        inFile = outFile
        outFile = f'{reg_dir}/functionalMask/cortex_{hemi}.nii.gz'
        os.system(
            f'flirt -in {inFile} -ref {ref_func}.nii.gz -applyxfm -init {xform_anat_to_func} -out {outFile} -interp nearestneighbour')

    os.system(f'fslmaths {reg_dir}/functionalMask/cortex_lh.nii.gz -add {reg_dir}/functionalMask/cortex_rh.nii.gz -bin {bilatCortexMask}')

### convert region mask
maskName =['Superior_Parietal_lobule_thr50','Postcentral_Gyrus_thr50']
results = open(resultsFile_name, 'a+')
for i in range(len(maskName)):
    results.write(f'{maskName[i]}\n')
    mask_path_stds = '/home/tonglab/Miao/fMRI/masks/nonVisualRegion'
    mask_path_std = mask_path_stds +'/'+ maskName[i]+'.nii.gz'
    mask_path_anat = reg_dir +'/anatomicalMask/'+ maskName[i]+'.nii.gz'
    mask_path_func  = reg_dir +'/functionalMask/'+ maskName[i]+'.nii.gz'
    finalMask = reg_dir +'/functionalMaskInCortex/'+ maskName[i]+'.nii.gz'

    os.system(
        f'mri_vol2vol --mov {mask_path_std} --targ {ref_anat} --lta {xform_std_to_anat} --o {mask_path_anat} --nearest')

    os.system(
        f'flirt -in {mask_path_anat} -ref {ref_func}.nii.gz -applyxfm -init {xform_anat_to_func} -out {mask_path_func} -interp nearestneighbour')

    # combine cortex mask with ROI mask
    # if not os.path.isfile(finalMask) or overwrite:
    print('Combining ROI mask with cortical mask...')
    os.system(f'fslmaths {reg_dir}/functionalMask/cortex_bi.nii.gz -mul {mask_path_func} {finalMask}')
    ### save by region

    os.system(f'fslstats {pathTSNR} -k {finalMask} -M >> {resultsFile_mean}')
    os.system(f'fslstats {pathTSNR} -k {finalMask} -S >> {resultsFile_std}')
    os.system(f'fslstats {pathTSNR} -k {finalMask} -V | tr \' \' \',\' >> {resultsFile_number}')

results.close()

