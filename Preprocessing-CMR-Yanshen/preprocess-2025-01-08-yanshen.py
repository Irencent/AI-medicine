import os
import shutil
import SimpleITK as sitk
import numpy as np
import pydicom
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle as pkl
import pandas as pd
from PIL import Image

def rename_dicom(dataroot='./DICOMs'):
    '''
    This function is:
    1. to remove 'DICOM.dcm', which serves as an index for the dataset but is not typically needed for most DICOM processing tasks
    2. to append the suffix '.dcm'
    '''
    for root, dirs, files in os.walk(dataroot):
        # if len(dirs)==0:
        if True:
            for i in range(len(files)):
                if files[i] == 'DICOMDIR.dcm':
                    os.remove(os.path.join(root, files[i]))
                    print(f'{os.path.join(root, files[i])} deleted')
                if not files[i].endswith('.dcm'):
                    os.rename(os.path.join(root, files[i]), os.path.join(root, f'{files[i]}.dcm'))

def resample_dataset(data_root, output):
    '''
    This function is:
    1. to extract the patient_id, patient_name, mod, date, and UnqiueID of a dcm file 
    2. to save the orignial dcm into 'output/<patient_id>_<patient_name>/<date>/<mod>/<UniqueID>.dcm'
    '''
    error=[]
    count=0
    for paths, dirs, files in os.walk(data_root):
        if len(dirs)==0:
        # if True:
            for i in range(len(files)):
                try:
                    info=pydicom.dcmread(os.path.join(paths,files[i]))
                    id=info['0010','0020'].value
                    name=str(info['0010','0010'].value).replace(' ','_').replace(':','')
                    mod=str(info['0008','103e'].value).replace(':','_')
                    date=str(info['0008','0020'].value)
                    ui=str(info['0008','0018'].value)
                    # print(id, name, mod)
                except:
                    print(f'Basic info missed: {os.path.join(paths,files[i])}')
                    continue
                if name:
                    
                    savepath=os.path.join(output,f'{id}_{name}',date,mod).replace('*','_').replace('?','_').replace('>','_').replace('<','_').replace('^','_')
                    print(os.path.join(savepath,f'{ui}.dcm'))
                    if not os.path.exists(savepath):
                        os.makedirs(savepath)
                    count+=1
                    try:
                        shutil.copyfile(os.path.join(paths,files[i]),os.path.join(savepath,f'{ui}.dcm'))
                    except:
                        print(f"Error occured when copying {os.path.join(paths,files[i])} to {os.path.join(savepath,f'{ui}.dcm')}")
                    # os.remove(os.path.join(paths,files[i]))
                else:
                    error.append(os.path.join(paths,files[i]))
        # for i in range(len(error)):
        #     print(error[i])
    return error

def fix_seriesnum(root='./New_dicom/',threshold=100,min_fps=25,expand_mod=False):
    """Fixes series number according to the positions of dicom files

    Args:
        root (str, optional): The root path of resampled dataset. Defaults to './New_dicom/'.
        threshold (int, optional): If number of files oversize the threshold, this folder will be checked whether if there exists error of dicom SNs. Defaults to 100.
        min_fps (int, optional): The fps of a single slice. Defaults to 25.
    """
    error=[]
    info_error=[]
    for paths,dirs,files in os.walk(root):
        if len(dirs)==0 and len(files)>=threshold:
            print(paths)
            try:
                series_num=[]
                pos_z=[]
                for i in range(len(files)):
                    try:
                        info=pydicom.dcmread(os.path.join(paths,files[i]))
                        pz=float(list(info['0020','0032'].value)[-1])
                        sn=info['0020','0011'].value
                        if pz not in pos_z:
                            pos_z.append(pz)
                        if sn not in series_num:
                            series_num.append(sn)
                    except:
                        print(f'error with {os.path.join(paths,files[i])}')
                        os.remove(os.path.join(paths,files[i]))
                        info_error.append(os.path.join(paths,files[i]))
                pos_z.sort(reverse=True)
                # if len(series_num)<len(files)/min_fps:
                if len(series_num)<len(pos_z):
                    print(f'{paths} own {len(files)} dicom files,\
                            \nsingle slice fps set as {min_fps},\
                            \nbut only {len(series_num)} series given by heads of dicom files,\
                            \nthey are {series_num}\
                            \ntheir physical z-axis position: {pos_z}')
                    for i in range(len(files)):
                        dicom_path=os.path.join(paths,files[i])
                        if dicom_path in info_error:
                            continue
                        info=pydicom.dcmread(dicom_path)
                        if expand_mod==False:
                            try:
                                info.SeriesNumber=int(info['0020','9057'].value)
                            except KeyError:
                                try:
                                    info.SeriesNumber=int(1+pos_z.index(float(list(info['0020','0032'].value)[-1])))
                                except:
                                    # print('set error')
                                    pass
                                error.append(os.path.join(paths,files[i]))
                                # print(info.SeriesNumber)
                        else:
                            # print(f'Expand mode: {expand_mod}')
                            try:
                                info.SeriesNumber=int(1+pos_z.index(float(list(info['0020','0032'].value)[-1])))
                            except:
                                # print('set error')
                                pass
                            error.append(os.path.join(paths,files[i]))
                        info.save_as(os.path.join(paths,files[i]))
                        # info['0020','0011']=instack_id
            except:
                print(f'unknow error occured with {paths}')
    return error,info_error


if __name__ == "__main__":
    root_path = 'I:/AMI'
    new_root =  'I:/AMI_new_root_hys/raw_data'
    save_root = 'I:/AMI_new_root_hys/nifti_data'
    
    # 1. process raw data
    rename_dicom(root_path)

    ## reorganize the dcm files based on <patient_id>_<patient_name>/<date>/<mod>/<UniqueID>.dcm'
    resample_dataset(root_path, new_root)

    ## fix some potential errors of series numbers
    fix_seriesnum(new_root)
