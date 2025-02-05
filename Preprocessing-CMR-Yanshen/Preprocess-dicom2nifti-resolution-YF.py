import pydicom
import numpy as np
import SimpleITK as sitk
import os
from tqdm import tqdm
import shutil
def BiobankBatchProcess(datapath='..\\data',NII_path='..\\NII_data',Resampled_path='..\\Resampled'):
    '''
    ## Process all patients' data into what we need
    ### Args:
        datapath: The root path of all patients' dicom data file folders,such as datapath/202101_20208,datapath/202102_20209...
        NII_path: The root path to store all patients' nifiti data
        Resampled_path: The root path to store all patients' resampled nifti data
    '''
    # Get all 20209-short axis dicom data files' path
    alldatapath=os.listdir(datapath)
    usefuldata=[]
    for i in range(len(alldatapath)):
        if '20209' in alldatapath[i]:
            if os.path.isdir(os.path.join(datapath,alldatapath[i])):
                usefuldata.append(alldatapath[i])
    # Convert dicom files to nifti data file and resample it
    bar=tqdm(range(len(usefuldata)),mininterval=1)
    bar.set_description('Extracting and Resampling Dicom Data Into Nifti Files')
    for i in bar:
        series,slicemap=Series(dicom_folder=os.path.join(datapath,usefuldata[i]))
        temppath=os.path.join(datapath,'temp',usefuldata[i])
        niipath=os.path.join(NII_path,usefuldata[i])
        try:
            os.makedirs(temppath)
        except:
            pass
        # DO NOT COMBINE THESE TWO MAKEDIR TOGETHER!!!!
        try:
            os.makedirs(niipath)
        except:
            pass
        # Extract 25 frames
        frame_dcm=extract_frames(slicemap=slicemap,dicom_folder=os.path.join(datapath,usefuldata[i]))
        sliceID=list(frame_dcm.keys())
        # Copy extracted 25 frames into temporary folder and convert them into useful nifti files
        for j in range(len(sliceID)):
            slice_path=os.path.join(temppath,f'{sliceID[j]}')
            try:
                os.mkdir(slice_path)
            except:
                pass
            for k in range(len(frame_dcm[sliceID[j]])):
                shutil.copyfile(os.path.join(datapath,usefuldata[i],frame_dcm[sliceID[j]][k]),os.path.join(slice_path,frame_dcm[sliceID[j]][k]))
            # Convert extracted frames into a single nifti file
            dcm2nii(path_read=slice_path,path_save=os.path.join(niipath,f'sliceID_{sliceID[j]}.nii.gz'))
            # Resample extracted frames to specified spacing
            resample_volume(Origin=os.path.join(niipath,f'sliceID_{sliceID[j]}.nii.gz'),output=os.path.join(niipath,f'sliceID_{sliceID[j]}.nii.gz'))
        # Remove temporary files
        shutil.rmtree(temppath)
    # Remove temporary file folder
    shutil.rmtree(os.path.join(datapath,'temp'))
    return usefuldata

def extract_frames(slicemap,dicom_folder='..\\data\\4395042_20209_2_0'):
    '''
    ## Extract 25 frames of dicom data from 50 frames
    ### Args:
        slicemap: The map of slices got by Series()
        dicom_folder: the path of the folder containing dicom series
    ### Return
        slicemapnew: The dictionary maps sliceID to the list of their relative path of extracted frames
    '''
    series = list(slicemap.keys())
    slicemapnew = {}
    for i in range(len(series)):
        ins_num=[]
        relpos={}
        if len(slicemap[series[i]])==50:
            for j in range(len(slicemap[series[i]])):
                frameID=int(str(pydicom.read_file(os.path.join(dicom_folder,slicemap[series[i]][j]))[0x20,0x13].value))
                ins_num.append(frameID)
                relpos[frameID]=slicemap[series[i]][j]
            ins_num=sorted(ins_num)
            slicemapnew[series[i]]=[]
            for j in range(1,50,2):
                slicemapnew[series[i]].append(relpos[j])
    return slicemapnew


def Series(dicom_folder='..\\data\\4395042_20209_2_0'):
    '''
    ## Get Series Numbers in a Dicom FileFolder and Their Corresponding Relative Paths of Dicom Files(You don't have to delete manifest.cvs, we can automatically ignore it)
    ### Args:
        dicom_folder: the path of the folder containing dicom series
    ### Return(Tuple):
        SeriesNumber: A list containing all series number of dicom files
        slicemap: A dictionary maps series number to the list of corresponding relative paths of dicom files
    '''
    dcms=os.listdir(dicom_folder)
    SeriesNumber=[]
    slicemap={}
    # Get all slices' series number in SeriesNumber
    # Map all slices' class into slicemap, such as {10:[xx.dcm,xy.dcm],11:[yy.dcm,yz.dcm]}
    for i in range(len(dcms)):
        if dcms[i].endswith('.dcm'):
            dcmpath=os.path.join(dicom_folder,dcms[i])
            num=pydicom.read_file(dcmpath)[0x20,0x11].value
            if num not in SeriesNumber:
                SeriesNumber.append(num)
            if num not in list(slicemap.keys()):
                slicemap[num]=[dcms[i]]
            else:
                slicemap[num].append(dcms[i])
        else:
            dcms.pop(i)
    return SeriesNumber,slicemap

def dcm2nii(path_read, path_save):
    '''
    ## Convert Dicom Series Files to a Single NII File
    ### Args:
        path_read: The file folder containing dicom series files(No other files exits)
        path_save: The path you save the .nii/.nii.gz data file
    '''
    # GetGDCMSeriesIDs读取序列号相同的dcm文件
    series_id = sitk.ImageSeriesReader.GetGDCMSeriesIDs(path_read)
    # GetGDCMSeriesFileNames读取序列号相同dcm文件的路径，series[0]代表第一个序列号对应的文件
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path_read, series_id[0])
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    image3d = series_reader.Execute()
    sitk.WriteImage(image3d, path_save)


def resample_volume(Origin='NII.nii.gz', interpolator = sitk.sitkLinear, new_spacing = [1.7708333730698,1.7708333730698,1],output='Resampled.nii.gz'):
    '''
    ## Resample MRI files to specified spacing, here we define spacing=[1.7708333730698,1.7708333730698,1]
    ### Args:
        Origin: The Original MRI file, MUST NOT BE DICOM!!!!!
        interpolater: The method of resampling
        new_spacing: The spacing we want to set
        output: The Output path of resampled MRI data
    ### Return:
        resample_image: The resampled MRI data
    '''
    volume = sitk.ReadImage(Origin)
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(original_size, original_spacing, new_spacing)]
    resampled_image = sitk.Resample(volume, 
                                    new_size, 
                                    sitk.Transform(), 
                                    interpolator,
                                    volume.GetOrigin(), 
                                    new_spacing, volume.GetDirection(), 
                                    0,
                                    volume.GetPixelID())
    sitk.WriteImage(resampled_image,output)
    return resampled_image

def FindMiddle(slicemap,dicom_folder='..\\data\\4395042_20209_2_0'):
    '''
    ## Find three middle slices
    ### Args:
        slicemap: The map of slices got by Series()
        dicom_folder: the path of the folder containing dicom series
    ### Return:
        middlemap: The dictionary maps three center slices' series number to their physical position, such as {'15': 124.8532579303, '14': 131.08738116028, '13': 137.32150439026}
    '''
    series=list(slicemap.keys())
    posmap={}
    pos=[]
    for i in range(len(series)):
        if len(slicemap[series[i]])==50:
            pos.append(position(os.path.join(dicom_folder,slicemap[series[i]][0])))
            posmap[pos[i]]=series[i]
            pos[i]=float(str(pos[i]))
    pos=sorted(pos)
    start=int((len(pos)-3)/2)
    middlemap={}
    for i in range(1,4):
        middlemap[posmap[pos[start+i]]]=pos[start+i]
    return middlemap


def position(dicom_dir):
    # Get single dcm's physical position
    file = pydicom.filereader.dcmread(dicom_dir)
    cosines = file[0x20, 0x37].value # ImageOrientationPatient [0020,0037]
    ipp = file[0x20, 0x32].value #ImagePositionPatient tag [0020, 0032]
    a1 = np.zeros((3,))
    a2 = np.zeros((3,))
    for i in range(3):
        a1[i] = cosines[i]
        a2[i] = cosines[i+3]
    index = np.abs(np.cross(a1, a2))
    return ipp[np.argmax(index)]
