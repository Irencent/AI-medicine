'''
Author: airscker
Date: 2023-02-02 08:46:30
LastEditors: airscker
LastEditTime: 2023-02-02 09:10:42
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''
import os
import shutil
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle as pkl

'''Preperation'''


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
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
        path_read, series_id[0])
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    image3d = series_reader.Execute()
    sitk.WriteImage(image3d, path_save)


def rename_dicom(dataroot='./DICOMs'):
    """
    The rename_dicom function renames all files in the DICOMs directory to have a .dcm extension.
    This is necessary for some of the functions used later on.

    :param dataroot: Specify the root of the directory containing all dicom files
    :return: 0
    :doc-author: airscker-AI
    """
    for root, dirs, files in os.walk(dataroot):
        # if len(dirs)==0:
        if True:
            for i in range(len(files)):
                if files[i] == 'DICOMDIR.dcm':
                    os.remove(os.path.join(root, files[i]))
                    print(f'{os.path.join(root,files[i])} deleted')
                if not files[i].endswith('.dcm'):
                    os.rename(os.path.join(root, files[i]), os.path.join(
                        root, f'{files[i]}.dcm'))
    return 0


def Rec_crop2D(mask, index):
    """
    The Rec_crop2D function takes a 3D mask and an index of the slice to be cropped.
    It returns a 2D croped mask and the coordinates of the croped region.

    :param mask: Crop the image
    :param index: Select the mask that is going to be cropped
    :return: The cropped mask and the coordinates of the bounding box
    :doc-author: airscker-AI
    """
    x_l, x_r, y_d, y_u = 0, 0, 0, 0
    for i in range(len(mask[index])):
        if np.count_nonzero(mask[index][i] == 1) > 0:
            y_u = i-1
            break
    for i in range(len(mask[index])-1, 0, -1):
        if np.count_nonzero(mask[index][i] == 1) > 0:
            y_d = i+1
            break
    mask_t = np.transpose(mask[index])
    for i in range(len(mask_t)-1, 0, -1):
        if np.count_nonzero(mask_t[i] == 1) > 0:
            x_r = i+1
            break
    for i in range(len(mask_t)):
        if np.count_nonzero(mask_t[i] == 1) > 0:
            x_l = i-1
            break
    # print(x_l,x_r,y_u,y_d)
    croped_mask = mask[index][y_u:y_d, x_l:x_r]
    return croped_mask, [x_l, x_r, y_u, y_d]


def crop(mask):
    x_l, x_r, y_d, y_u = 0, 0, 0, 0
    for i in range(len(mask)):
        if np.count_nonzero(mask[i] > 0) > 0:
            index = i
    for i in range(len(mask[index])):
        if np.count_nonzero(mask[index][i] == 1) > 0:
            y_u = i-1
            break
    for i in range(len(mask[index])-1, 0, -1):
        if np.count_nonzero(mask[index][i] == 1) > 0:
            y_d = i+1
            break
    mask_t = np.transpose(mask[index])
    for i in range(len(mask_t)-1, 0, -1):
        if np.count_nonzero(mask_t[i] == 1) > 0:
            x_r = i+1
            break
    for i in range(len(mask_t)):
        if np.count_nonzero(mask_t[i] == 1) > 0:
            x_l = i-1
            break
    # print(x_l,x_r,y_u,y_d)
    # croped_mask=mask[index][y_u:y_d,x_l:x_r]
    return x_l, x_r, y_u, y_d


def sort_dict(dictionary):
    """
    The sort_dict function takes a dictionary and sorts it by key.
    It returns a new dictionary with the same keys but sorted values.

    :param dictionary: Pass in the dictionary that is to be sorted
    :return: A dictionary with the keys sorted in ascending order
    :doc-author: airscker-AI
    """
    sorted_keys = sorted(dictionary)
    new_map = {}
    for i in range(len(sorted_keys)):
        new_map[sorted_keys[i]] = dictionary[sorted_keys[i]]
    return new_map


def del_empty(root):
    """
    The del_empty function deletes all empty folders in the root directory.


    :param root: Specify the root directory to start from
    :return: The directory that was deleted
    :doc-author: airscker-AI
    """
    for roots, dirs, files in os.walk(root):
        if len(dirs) == 0:
            if len(files) == 0:
                shutil.rmtree(roots)
                print(f'{roots}')


def mac_bug(root):
    for roots, dirs, files in os.walk(root):
        for i in range(len(files)):
            if files[i].startswith('._'):
                os.remove(os.path.join(roots, files[i]))


def norm_img(img):
    return np.uint8(255.0*(img-np.min(img))/(np.max(img)-np.min(img)))


def exclude_seg_files(file_list=[], segmentations=['Segmentation.nii', 'Segmentation.nii.gz', '.DS_Store']):
    for seg_file in segmentations:
        if seg_file in file_list:
            file_list.remove(seg_file)
    return file_list


def ds_store(root):
    for roots, dirs, files in os.walk(root):
        for i in range(len(files)):
            if '.DS_Store' == files[i]:
                os.remove(os.path.join(roots, files[i]))
                print(os.path.join(root, files[i]))


def resample_volume(Origin='NII.nii.gz', volume=None, interpolator=sitk.sitkLinear, new_spacing=[1.7708333730698, 1.7708333730698, 1], output='Resampled.nii'):
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
    if volume is None:
        volume = sitk.ReadImage(Origin)
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for osz, ospc,
                nspc in zip(original_size, original_spacing, new_spacing)]
    resampled_image = sitk.Resample(volume,
                                    new_size,
                                    sitk.Transform(),
                                    interpolator,
                                    volume.GetOrigin(),
                                    new_spacing, volume.GetDirection(),
                                    0,
                                    volume.GetPixelID())
    if output != '':
        sitk.WriteImage(resampled_image, output)
    return resampled_image


'''resample dataset'''


def resample_dataset(dataroot='./DICOMs', output='./New_dicom/'):
    """Resample the dataset to a new directory .

    Args:
        dataroot (str, optional):The root path of all patients' dicom files. Defaults to './DICOMs'.
        output (str, optional):The output path of resampled dicom dataset. Defaults to './New_dicom/'.
    """
    error = []
    count = 0
    for paths, dirs, files in os.walk(dataroot):
        # if len(dirs)==0:
        if True:
            for i in range(len(files)):
                try:
                    info = pydicom.dcmread(os.path.join(paths, files[i]))
                    id = info['0010', '0020'].value
                    name = str(info['0010', '0010'].value).replace(' ', '_')
                    mod = str(info['0008', '103e'].value).replace(':', '_')
                    date = str(info['0008', '0020'].value)
                    ui = str(info['0008', '0018'].value)
                except:
                    print(f'Basic info missed: {os.path.join(paths,files[i])}')
                    continue
                if name != '':
                    savepath = os.path.join(output, f'{id}_{name}', date, mod).replace(
                        '*', '_').replace('?', '_').replace('>', '_').replace('<', '_')
                    if not os.path.exists(savepath):
                        os.makedirs(savepath)
                    count += 1
                    try:
                        shutil.copyfile(os.path.join(
                            paths, files[i]), os.path.join(savepath, f'{ui}.dcm'))
                    except:
                        print(
                            f"Error occured when copying {os.path.join(paths,files[i])} to {os.path.join(savepath,f'{ui}.dcm')}")
                    # os.remove(os.path.join(paths,files[i]))
                else:
                    error.append(os.path.join(paths, files[i]))
        # for i in range(len(error)):
        #     print(error[i])
    return error


'''fix series numbers'''


def fix_seriesnum(root='./New_dicom/', threshold=100, min_fps=25, expand_mod=False):
    """Fixes series number according to the positions of dicom files

    Args:
        root (str, optional): The root path of resampled dataset. Defaults to './New_dicom/'.
        threshold (int, optional): If number of files oversize the threshold, this folder will be checked whether if there exists error of dicom SNs. Defaults to 100.
        min_fps (int, optional): The fps of a single slice. Defaults to 25.
    """
    error = []
    info_error = []
    for paths, dirs, files in os.walk(root):
        if len(dirs) == 0 and len(files) >= threshold:
            try:
                series_num = []
                pos_z = []
                for i in range(len(files)):
                    try:
                        info = pydicom.dcmread(os.path.join(paths, files[i]))
                        pz = float(list(info['0020', '0032'].value)[-1])
                        sn = info['0020', '0011'].value
                        if pz not in pos_z:
                            pos_z.append(pz)
                        if sn not in series_num:
                            series_num.append(sn)
                    except:
                        print(f'error with {os.path.join(paths,files[i])}')
                        os.remove(os.path.join(paths, files[i]))
                        info_error.append(os.path.join(paths, files[i]))
                pos_z.sort(reverse=True)
                # if len(series_num)<len(files)/min_fps:
                if len(series_num) < len(pos_z):
                    print(f'{paths} own {len(files)} dicom files,\
                            \nsingle slice fps set as {min_fps},\
                            \nbut only {len(series_num)} series given by heads of dicom files,\
                            \nthey are {series_num}\
                            \ntheir physical z-axis position: {pos_z}')
                    for i in range(len(files)):
                        dicom_path = os.path.join(paths, files[i])
                        if dicom_path in info_error:
                            continue
                        info = pydicom.dcmread(dicom_path)
                        if expand_mod == False:
                            try:
                                info.SeriesNumber = int(
                                    info['0020', '9057'].value)
                            except KeyError:
                                try:
                                    info.SeriesNumber = int(
                                        1+pos_z.index(float(list(info['0020', '0032'].value)[-1])))
                                except:
                                    # print('set error')
                                    pass
                                error.append(os.path.join(paths, files[i]))
                                # print(info.SeriesNumber)
                        else:
                            # print(f'Expand mode: {expand_mod}')
                            try:
                                info.SeriesNumber = int(
                                    1+pos_z.index(float(list(info['0020', '0032'].value)[-1])))
                            except:
                                # print('set error')
                                pass
                            error.append(os.path.join(paths, files[i]))
                        info.save_as(os.path.join(paths, files[i]))
                        # info['0020','0011']=instack_id/
            except:
                print(f'unknow error occured with {paths}')
    return error, info_error


'''split 4ch data from sax'''


def fix_4ch(path=r'E:\BaiduNetdiskDownload\ARVC_1\11235202_HE_HUI', fps=25, thres=30):
    """
    The fix_4ch function is used to fix the 4 channel images.
    The function will find all the folders with more than 30 dicom files, and then move all of them into a new folder named '4CH'.


    :param path: Specify the root path of the data
    :param fps: Set the frame rate of the video
    :param thres: Determine the minimum number of files in a folder
    :return: The path of the 4ch folder
    :doc-author: airscker-AI
    """
    for roots, dirs, files in os.walk(path):
        if len(dirs) == 0:
            size = {}
            if len(files) > thres:
                for i in range(len(files)):
                    sd = pydicom.dcmread(os.path.join(
                        roots, files[i])).pixel_array.shape
                    if sd in size.keys():
                        size[sd].append(os.path.join(roots, files[i]))
                    else:
                        size[sd] = [os.path.join(roots, files[i])]
                # print(size)
                if len(size.keys()) == 2:
                    savepath = roots.replace(roots.split('/')[-1], '4CH')
                    print(savepath)
                    try:
                        os.makedirs(savepath)
                    except:
                        pass
                    for key in size.keys():
                        if len(size[key]) < 30:
                            print(len(size[key]))
                            for num in range(len(size[key])):
                                shutil.move(size[key][num], savepath)
                                print(f'{size[key][num]} moved to {savepath}')


'''split 4chlge data from saxlge'''


def fix_4ch_lge(path=r'E:\BaiduNetdiskDownload\ARVC_1\11235202_HE_HUI', fps=25, thres=4):
    """
    The fix_4ch_lge function is a helper function that fixes the 4ch_lge dataset.
    The problem with this dataset is that it has multiple frames for each slice, but
    the number of frames varies from slice to slice. This function finds all the slices
    that have an incorrect number of frames and moves them into a new folder called 
    4CH_LGE in their original location.

    :param path: Specify the root path of the data
    :param fps: Determine how many frames are used to calculate the average pixel value of a slice
    :param thres: Determine whether to delete the file
    :return: The path of the files that have been moved
    :doc-author: airscker-AI
    """
    for roots, dirs, files in os.walk(path):
        if len(dirs) == 0:
            size = {}
            if len(files) % fps != 0 and len(files) < 50:
                for i in range(len(files)):
                    sd = pydicom.dcmread(os.path.join(
                        roots, files[i])).pixel_array.shape
                    if sd in size.keys():
                        size[sd].append(os.path.join(roots, files[i]))
                    else:
                        size[sd] = [os.path.join(roots, files[i])]
                # print(size)
                # if len(size.keys())>1:
                #     print(roots)
                if len(size.keys()) == 2:
                    savepath = roots.replace(roots.split('/')[-1], '4CH_LGE')
                    # print(savepath)
                    try:
                        os.makedirs(savepath)
                    except:
                        pass
                    f_key = None
                    num = len(files)
                    for key in size.keys():
                        if len(size[key]) < num:
                            f_key = key
                            num = len(size[key])
                    for i in range(len(size[f_key])):
                        shutil.move(size[key][i], savepath)
                        print(f'{size[key][i]} moved to {savepath}')


'''save lge data'''


def save_lge(dataroot='./New_dicom/', output='./', fps=25, thres=100):
    """Convert dicom data files into nifti format, and resample its resolution
    Args:
        dataroot (str, optional): The root path of resampled data. Defaults to './New_dicom/'.
        output (str, optional): The output path of generated nfiti data. Defaults to './'.
        fps (int, optional): The fps of a single slice(SAX/LAX4CH). Defaults to 25.
    """
    # try:
    #     shutil.rmtree(os.path.join(output,'SAX_LGE_data',dataroot.split('/')[-1]))
    # except:
    #     pass
    error = []
    for roots, dirs, files in os.walk(dataroot):
        if len(dirs) == 0:
            slice_map = {}
            file_info = {}
            files = exclude_seg_files(files)
            try:
                for i in range(len(files)):
                    info = pydicom.dcmread(os.path.join(roots, files[i]))
                    file_info[files[i]] = info
                    sn = info.SeriesNumber
                    if sn not in list(slice_map.keys()):
                        slice_map[sn] = [files[i]]
                    else:
                        slice_map[sn].append(files[i])
                slices = list(slice_map.keys())
                for i in range(len(slices)):
                    dcms = slice_map[slices[i]]
                    pats_name = str(
                        file_info[dcms[0]]['0010', '0010'].value).replace(' ', '_')
                    pats_id = file_info[dcms[0]]['0010', '0020'].value
                    filetag = str(file_info[dcms[0]]['0008', '103e'].value)
                    print(filetag)
                    # if ('BH' in filetag or '4CH' in filetag or '4 CH' in filetag or 'TFE' in filetag or 'SecondaryCapture' in filetag or '1'==filetag) and ('SA' not in filetag and 'sa' not in filetag and 'shot' not in filetag and '8slices' not in filetag and 'LVOT' not in filetag):
                    if len(files) < thres:
                        print(f'fps: {len(dcms)}')
                        # if len(dcms)%fps==0:
                        if len(dcms) >= 10:
                            # temp_folder=os.path.join(output,'4ch_temp')
                            # savepath=os.path.join(output,'4CH_data',f'{pats_id}_{pats_name}')
                            # filename=f'{pats_id}_{pats_name}_{slices[i]}.nii.gz'
                            continue
                        else:
                            temp_folder = os.path.join(output, '4ch_lge_temp')
                            savepath = os.path.join(
                                output, '4CH_LGE_data', f'{pats_id}_{pats_name}')
                            filename = f'{pats_id}_{pats_name}_{slices[i]}.nii.gz'
                            # continue
                        for j in range(len(dcms)):
                            if not os.path.exists(temp_folder):
                                os.makedirs(temp_folder)
                            shutil.copyfile(os.path.join(
                                roots, dcms[j]), os.path.join(temp_folder, dcms[j]))
                        if not os.path.exists(savepath):
                            os.makedirs(savepath)
                        try:
                            # if True:
                            dcm2nii(temp_folder, os.path.join(
                                savepath, filename))
                            # resample_volume(Origin=os.path.join(savepath,filename),output=os.path.join(savepath,filename))
                            print(f'{filename} saved into {savepath}')
                        except:
                            pass
                        shutil.rmtree(temp_folder)
                    # elif 'SA' in filetag or 'sa' in filetag or '2'==filetag or 'shot' in filetag or 'PSIR_TFE 8slices'==filetag:
                    elif len(files) >= thres:
                        # continue
                        print(f'fps: {len(dcms)}')
                        """
                        Bigger slice ID, lower z-axis position
                        """
                        # if len(dcms)%fps==0:
                        if len(dcms) >= 25:
                            # if True:
                            # temp_folder=os.path.join(output,'sax_temp')
                            # savepath=os.path.join(output,'SAX_data',f'{pats_id}_{pats_name}')
                            # filename=f'slice_{slices[i]}.nii.gz'
                            continue
                        else:
                            temp_folder = os.path.join(output, 'sax_lge_temp')
                            savepath = os.path.join(
                                output, 'SAX_LGE_data', f'{pats_id}_{pats_name}')
                            filename = f'{pats_id}_{pats_name}_{slices[i]}.nii.gz'
                            # continue
                        for j in range(len(dcms)):
                            if not os.path.exists(temp_folder):
                                os.makedirs(temp_folder)
                            shutil.copyfile(os.path.join(
                                roots, dcms[j]), os.path.join(temp_folder, dcms[j]))
                        if not os.path.exists(savepath):
                            os.makedirs(savepath)
                        try:
                            dcm2nii(temp_folder, os.path.join(
                                savepath, filename))
                            # resample_volume(Origin=os.path.join(savepath,filename),output=os.path.join(savepath,filename))
                            print(f'{filename} saved into {savepath}')
                        except:
                            pass
                        shutil.rmtree(temp_folder)
            except:
                print(f'unknow error occured with {roots}')
                error.append(roots)
    temps = ['4ch_temp', '4ch_lge_temp', 'sax_temp', 'sax_lge_temp']
    for i in range(len(temps)):
        try:
            shutil.rmtree(os.path.join(output, temps[i]))
        except:
            pass
    return error


'''save cine data'''


def save_cine(dataroot='./New_dicom/', output='./', fps=25, thres=60):
    """Convert dicom data files into nifti format, and resample its resolution
    Args:
        dataroot (str, optional): The root path of resampled data. Defaults to './New_dicom/'.
        output (str, optional): The output path of generated nfiti data. Defaults to './'.
        fps (int, optional): The fps of a single slice(SAX/LAX4CH). Defaults to 25.
    """
    error = []
    for roots, dirs, files in os.walk(dataroot):
        if len(dirs) == 0:
            try:
                slice_map = {}
                file_info = {}
                files = exclude_seg_files(files)
                for i in range(len(files)):
                    info = pydicom.dcmread(os.path.join(roots, files[i]))
                    file_info[files[i]] = info
                    sn = info.SeriesNumber
                    if sn not in list(slice_map.keys()):
                        slice_map[sn] = [files[i]]
                    else:
                        slice_map[sn].append(files[i])
                slices = list(slice_map.keys())
                for i in range(len(slices)):
                    dcms = slice_map[slices[i]]
                    pats_name = str(
                        file_info[dcms[0]]['0010', '0010'].value).replace(' ', '_')
                    pats_id = file_info[dcms[0]]['0010', '0020'].value
                    filetag = str(file_info[dcms[0]]['0008', '103e'].value)
                    print(filetag)
                    # if ('BH' in filetag or '4CH' in filetag or '4 CH' in filetag or 'TFE' in filetag or 'SecondaryCapture' in filetag or '1'==filetag) and ('SA' not in filetag and 'sa' not in filetag and 'shot' not in filetag and '8slices' not in filetag and 'LVOT' not in filetag):
                    if len(files) < thres and 'LVOT' not in filetag and 'SA' not in filetag and '3Ch' not in filetag:
                        print(f'fps: {len(dcms)}')
                        # if len(dcms)%fps==0:
                        if len(dcms) >= 10:
                            temp_folder = os.path.join(output, '4ch_temp')
                            savepath = os.path.join(
                                output, '4CH_data', f'{pats_id}_{pats_name}')
                            filename = f'{pats_id}_{pats_name}_{slices[i]}.nii.gz'
                            # continue
                        else:
                            # temp_folder=os.path.join(output,'4ch_lge_temp')
                            # savepath=os.path.join(output,'4CH_LGE_data',f'{pats_id}_{pats_name}')
                            # filename=f'{pats_id}_{pats_name}_{slices[i]}.nii.gz'
                            continue
                        for j in range(len(dcms)):
                            if not os.path.exists(temp_folder):
                                os.makedirs(temp_folder)
                            shutil.copyfile(os.path.join(
                                roots, dcms[j]), os.path.join(temp_folder, dcms[j]))
                        if not os.path.exists(savepath):
                            os.makedirs(savepath)
                        try:
                            # if True:
                            dcm2nii(temp_folder, os.path.join(
                                savepath, filename))
                            # resample_volume(Origin=os.path.join(savepath,filename),output=os.path.join(savepath,filename))
                            print(f'{filename} saved into {savepath}')
                        except:
                            pass
                        shutil.rmtree(temp_folder)
                        # pass
                    # elif 'SA' in filetag or 'sa' in filetag or '2'==filetag or 'shot' in filetag or 'PSIR_TFE 8slices'==filetag:
                    elif len(files) >= thres and '4CH' not in filetag or 'SAX' in filetag:
                        print(f'fps: {len(dcms)}')
                        """
                        Bigger slice ID, lower z-axis position
                        """
                        # if len(dcms)%fps==0:
                        # if len(dcms)>=25:
                        if True:
                            temp_folder = os.path.join(output, 'sax_temp')
                            savepath = os.path.join(
                                output, 'SAX_data', f'{pats_id}_{pats_name}')
                            filename = f'slice_{slices[i]}.nii.gz'
                            # continue
                        else:
                            # temp_folder=os.path.join(output,'sax_lge_temp')
                            # savepath=os.path.join(output,'SAX_LGE_data',f'{pats_id}_{pats_name}')
                            # filename=f'{pats_id}_{pats_name}_{slices[i]}.nii.gz'
                            continue
                        for j in range(len(dcms)):
                            if not os.path.exists(temp_folder):
                                os.makedirs(temp_folder)
                            shutil.copyfile(os.path.join(
                                roots, dcms[j]), os.path.join(temp_folder, dcms[j]))
                        if not os.path.exists(savepath):
                            os.makedirs(savepath)
                        try:
                            dcm2nii(temp_folder, os.path.join(
                                savepath, filename))
                            # resample_volume(Origin=os.path.join(savepath,filename),output=os.path.join(savepath,filename))
                            print(f'{filename} saved into {savepath}')
                        except:
                            pass
                        shutil.rmtree(temp_folder)
            except:
                print(f'unknow error occured with {roots}')
                error.append(roots)
    temps = ['4ch_temp', '4ch_lge_temp', 'sax_temp', 'sax_lge_temp']
    for i in range(len(temps)):
        try:
            shutil.rmtree(os.path.join(output, temps[i]))
        except:
            pass
    return error


'''reslice fps as 25'''


def fix_single_file_fps(path, fps=25):
    """
    The fix_single_file_fps function fixes the fps of a single file.

    :param path: Specify the path to the file that needs to be fixed
    :param fps: Specify the number of frames in a single second
    :return: The number of frames in the file
    :doc-author: airscker-AI
    """
    data = sitk.ReadImage(path)
    spacing = list(data.GetSpacing())
    array = sitk.GetArrayFromImage(data)
    frames = len(array)
    if frames != fps:
        new_space = spacing.copy()
        new_space[2] = frames/float(fps)
        resampled_data = resample_volume(
            Origin=path, output='', new_spacing=new_space)
        if sitk.GetArrayFromImage(resampled_data).shape[0] != 25:
            print(
                f'resample failure: {path}/{sitk.GetArrayFromImage(resampled_data).shape[0]} {spacing}/{new_space}')
        else:
            sitk.WriteImage(resampled_data, path)


def fix_fps(fps=25, nifti_path='E:/VST_fusion/ARVC_1_nifti/', fix=True):
    """
    The fix_fps function fixes the fps of a nifti file by resampling it to match
    the desired fps. The function takes in an optional argument fix which is set
    to True by default. If this argument is set to False, the function will print out
    a list of all files that need fixing and their current frame count but will not 
    actually fix them.

    :param fps: Set the fps of the output video
    :param nifti_path: Specify the path to the nifti files
    :param fix: Decide whether to fix the fps or not
    :return: The number of frames in each 
    :doc-author: airscker-AI
    """
    mods = ['SAX_data', '4CH_data']
    for i in range(len(mods)):
        for roots, dirs, files in os.walk(os.path.join(nifti_path, mods[i])):
            if len(dirs) == 0:
                frame_list = []
                files = exclude_seg_files(os.listdir(roots))
                for j in range(len(files)):
                    data = sitk.ReadImage(os.path.join(roots, files[j]))
                    spacing = list(data.GetSpacing())
                    array = sitk.GetArrayFromImage(data)
                    frames = len(array)
                    # print(frames)
                    frame_list.append(frames)
                    if frames != fps:
                        print(f'{roots} {files[j]} {frames}')
                        if fix:
                            '''FIX FPS HERE'''
                            new_space = spacing.copy()
                            new_space[2] = frames/float(fps)
                            resampled_data = resample_volume(Origin=os.path.join(
                                roots, files[j]), output='', new_spacing=new_space)
                            if sitk.GetArrayFromImage(resampled_data).shape[0] != 25:
                                print(
                                    f'resample failure: {roots} {files[j]} {frames}/{sitk.GetArrayFromImage(resampled_data).shape[0]} {spacing}/{new_space}')
                            else:
                                sitk.WriteImage(
                                    resampled_data, os.path.join(roots, files[j]))
                if np.std(frame_list) != 0:
                    print(f'frame_error {roots} {frame_list}')


'''reslice lge as 9 views'''


def reslice_lge(root, refer: sitk.Image = None, fps=9):
    """
    The reslice_lge function takes in a root directory and returns an image with the correct number of slices.
    The function will first walk through the root directory to find all .nii files, excluding segmentations.
    It will then sort these files based on their z-coordinate and create a new array that has only one slice from each scan.
    The function will then return this new array as well as the spacing information for each file.

    :param root: Specify the root directory of the data
    :param refer: sitk.Image: Specify the reference image
    :param fps: Control the frame rate of the output video
    :return: A new image with the same origin, spacing and direction as the refer image
    :doc-author: airscker-AI
    """
    for roots, dirs, files in os.walk(root):
        if len(dirs) == 0:
            files = exclude_seg_files(files)
            data_map = {}
            for i in range(len(files)):
                data = sitk.ReadImage(os.path.join(roots, files[i]))
                data_map[data.GetOrigin()[-1]] = data
            data_map = sort_dict(data_map)
            new_array = []
            for key in data_map.keys():
                array = sitk.GetArrayFromImage(data_map[key])
                if array.shape[0] > 1:
                    array = np.array([array[array.shape[0]//2]])
                new_array.append(array)
            new_array = np.array(new_array).squeeze()

            if refer is None:
                refer = data_map[key]
            # print(new_array.shape,sitk.GetArrayFromImage(refer).shape)
            origin = refer.GetOrigin()
            spacing = refer.GetSpacing()
            new_spacing = list(spacing)
            new_spacing[2] = spacing[2]*len(files)/float(fps)
            direction = refer.GetDirection()
            new_data = sitk.GetImageFromArray(new_array)
            new_data.SetDirection(direction)
            new_data.SetOrigin(origin)
            new_data.SetSpacing(spacing)
            # new_size = [new_array.shape[1],new_array.shape[2],fps]
            # resampled_image = sitk.Resample(new_data,new_size,sitk.Transform(),sitk.sitkLinear,origin,new_spacing,direction,0,refer.GetPixelID())
            return new_data


def batch_reslice(root, fps=9, delete=True, mod='SAX_LGE_data'):
    """
    The batch_reslice function takes in a root directory and reslices all the LGE images within that directory.
    The function will also delete the original files after it has created a new file with the same name but with an _fps_## suffix.
    This is done to save space on disk, as well as speed up future processing.

    :param root: Specify the root directory of the data
    :param fps: Specify the number of frames to be extracted from each slice
    :param delete: Delete the original data after reslicing
    :param mod: Specify the folder name of the data
    :return: A list of folders that failed to reslice
    :doc-author: airscker-AI
    """
    error = []
    available_data = []
    for roots, dirs, files in os.walk(root):
        if len(dirs) == 0 and mod in roots:
            pat_name = roots.split('/')[-1]
            # if f'{pat_name}_fps_{fps}.nii.gz' not in files:
            available_data.append(roots)
    bar = tqdm(range(len(available_data)))
    for i in bar:
        roots = available_data[i]

        pat_name = roots.split('/')[-1]
        if os.path.exists(os.path.join(roots, f'{pat_name}_fps_{fps}.nii.gz')):
            continue
            # os.remove(os.path.join(roots, f'{pat_name}_fps_{fps}.nii.gz'))
        original_files = exclude_seg_files(os.listdir(roots))
        try:
            # new_data = reslice_lge(root=roots, refer=sitk.ReadImage(os.path.join(roots,'Segmentation.nii.gz')),fps=fps)
            new_data = reslice_lge(root=roots, fps=fps)
        except:
            print(f'Reslice failure with {roots}')
            error.append(roots)
            continue
        temp_file = os.path.join(roots, 'temp.nii.gz')
        sitk.WriteImage(new_data, temp_file)


        new_spacing = list(new_data.GetSpacing())
        new_spacing[2] = new_spacing[2] * \
            sitk.GetArrayFromImage(new_data).shape[0]/fps
        new_data = resample_volume(Origin=temp_file, new_spacing=new_spacing)
        # print(sitk.GetArrayFromImage(new_data).shape)
        if sitk.GetArrayFromImage(new_data).shape[0] != fps:
            print(
                f'ERROR occured with {roots}: {fps}/{sitk.GetArrayFromImage(new_data).shape[0]}')
            error.append(roots)
        else:
            # if not os.path.exists(roots):
            #     os.makedirs(roots)
            sitk.WriteImage(new_data, os.path.join(
                roots, f'{pat_name}_fps_{fps}.nii.gz'))
            if delete:
                # shutil.rmtree(roots)
                for file in original_files:
                    os.remove(os.path.join(roots, file))
        os.remove(temp_file)
        # break
    return error


'''check sizes of data'''


def check_size(dataroot='/Users/airskcer/Downloads/DCM_nifti-20230107/'):
    """
    The check_size function checks the size of all the files in a given folder.
    It returns a dictionary with keys as folders and values as numpy arrays containing
    the shape of each file in that folder. If there is any inconsistency in the sizes, 
    it will return an error dictionary with keys as folders and values as numpy arrays 
    containing shapes of each file.

    :param dataroot: Specify the root folder of the dataset
    :return: A dictionary of the folders containing files with different sizes
    :doc-author: airscker-AI
    """
    error = {}
    folders = []
    for roots, dirs, files in os.walk(dataroot):
        if len(dirs) == 0:
            folders.append(roots)
    bar = tqdm(range(len(folders)))
    for i in bar:
        files = exclude_seg_files(os.listdir(folders[i]))
        if len(files) == 1:
            continue
        sizes = []
        for j in range(len(files)):
            data = sitk.ReadImage(os.path.join(folders[i], files[j]))
            sizes.append(sitk.GetArrayFromImage(data).shape)
        sizes = np.array(sizes)
        if np.sum(np.std(sizes, axis=0)) != 0:
            error[folders[i]] = sizes
    for key in error:
        print(key, error[key])
    return error


def error_process(error_map: dict):
    """
    The error_process function is used to find the files that have different shapes and move them into a new folder.


    :param error_map: dict: Store the error information
    :return: The file names and the shape of the files that have an error
    :doc-author: airscker-AI
    """
    for path in error_map.keys():
        map = {}
        files = exclude_seg_files(os.listdir(path))
        for i in range(len(files)):
            map[files[i]] = np.array(list(sitk.GetArrayFromImage(
                sitk.ReadImage(os.path.join(path, files[i]))).shape))
        shapes = np.array(list(map.values()))
        new_shape = np.unique(shapes, axis=0)
        error_shape = None
        if len(files) > 2:
            for i in range(len(new_shape)):
                count = 0
                for j in range(len(shapes)):
                    if np.all(shapes[j] == new_shape[i]):
                        count += 1
                if count == 1:
                    error_shape = new_shape[i]
        else:
            continue
        if error_shape is not None:
            for key in map:
                if np.all(error_shape == map[key]):
                    print(path, key, error_shape)
                    # if 'SAX_LGE_data' in path:
                    #     target = path.replace('SAX_LGE_data', '4CH_LGE_data')
                    # try:
                    #     os.makedirs(target)
                    # except:
                    #     pass
                    # try:
                    #     shutil.move(os.path.join(path,key),target)
                    # except:
                    #     print(f'{target} already exists')
                    os.remove(os.path.join(path, key))


'''remove mixed data according to their fps'''


def remove_mixed(root='/Users/airskcer/Downloads/PAH_nifti-20230201/'):
    """
    The remove_mixed function removes the mixed series from the dataset.
    The mixed series are those with more than 3 frames, which is not possible for a single cardiac cycle.


    :param root: Specify the root directory of the data
    :return: The files that have more than 3 frames and the folders that have no files in them
    :doc-author: airscker-AI
    """
    for roots, dirs, files in os.walk(root):
        if len(dirs) == 0:
            file = exclude_seg_files(os.listdir(roots))
            if len(files) >= 1 and 'LGE' in roots:
                for i in range(len(files)):
                    fps = sitk.GetArrayFromImage(sitk.ReadImage(
                        os.path.join(roots, files[i]))).shape[0]
                    if fps > 3:
                        print(roots, files[i], fps)
                        if fps > 5:
                            os.remove(os.path.join(roots, files[i]))
            if 'SAX_data' in roots:
                for i in range(len(files)):
                    fps = sitk.GetArrayFromImage(sitk.ReadImage(
                        os.path.join(roots, files[i]))).shape[0]
                    if fps < 5:
                        print(roots, files[i], fps)
                        os.remove(os.path.join(roots, files[i]))
            if len(files) == 0:
                print(roots)
                shutil.rmtree(roots)


'''rename sax data file names'''


def rename_mid(root, show=True):
    """
    The rename_mid function renames the slice files in a directory so that they are named 'slice_mid.nii.gz', 'slice_up.nii.gz', and
    'slice_down.nii.gz'. This is done by first finding all of the slice files, sorting them by their z-coordinate, and then renaming them
    so that they are ordered from top to bottom.

    :param root: Specify the root directory where the data is located
    :param show: Print the processing information
    :return: The number of patients processed
    :doc-author: airscker-AI
    """
    for roots, dirs, files in os.walk(root):
        if len(dirs) != 0:
            print(roots)
        if len(dirs) == 0 and 'SAX_data' in roots:
            files = exclude_seg_files(files)
            if len(files) >= 5 and ('slice_up.nii.gz' not in files or 'slice_mid.nii.gz' not in files or 'slice_down.nii.gz' not in files):
                slice_map = {}
                for i in range(len(files)):
                    data = sitk.ReadImage(os.path.join(roots, files[i]))
                    slice_map[data.GetOrigin()[-1]] = files[i]
                slice_pos = sorted(list(slice_map.keys()))
                # for i in range(len(slice_pos)):
                #     os.rename(os.path.join(roots,slice_map[slice_pos[i]]),os.path.join(roots,f'slice_{i}.nii.gz'))
                # os.rename(os.path.join(roots,f'slice_{len(slice_pos)//2}.nii.gz'), os.path.join(roots, 'slice_mid.nii.gz'))
                # os.rename(os.path.join(roots,f'slice_{len(slice_pos)//2+2}.nii.gz'), os.path.join(roots, 'slice_down.nii.gz'))
                # os.rename(os.path.join(roots,f'slice_{len(slice_pos)//2-2}.nii.gz'), os.path.join(roots, 'slice_up.nii.gz'))
                # if show:
                #     print(f'{roots} processed')
                os.rename(os.path.join(roots, slice_map[slice_pos[len(
                    slice_pos)//2]]), os.path.join(roots, 'slice_mid.nii.gz'))
                os.rename(os.path.join(roots, slice_map[slice_pos[len(
                    slice_pos)//2-2]]), os.path.join(roots, 'slice_down.nii.gz'))
                os.rename(os.path.join(roots, slice_map[slice_pos[len(
                    slice_pos)//2+2]]), os.path.join(roots, 'slice_up.nii.gz'))
            elif len(files) < 5:
                print(roots, len(files))


'''find out error of data'''


def find_error(nifti_path='E:/BaiduNetdiskDownload/PAH_nifti', niftis=[], cine_fps=25, max_slices_sax=13, max_slices_lax=1):
    '''
    ## Find out: 
        - The lost data
        - Multi slices of 4CH CINE/LGE
        - Error with cine_fps

    Automatically exclude the segmentation files
    '''
    chart = []
    for i in range(len(niftis)):
        try:
            note = {}
            info = niftis[i]
            note['NAME'] = niftis[i]
            note['MOD'] = nifti_path.split('/')[-1]
            # print(niftis[i])
            try:
                sax_file = os.path.join(nifti_path, 'SAX_data', niftis[i])
                '''Remove segmentation files'''
                slice_files = exclude_seg_files(os.listdir(sax_file))

                if len(slice_files) >= max_slices_sax:
                    info += f' Multiple_sax_cine_slice_{len(slice_files)}'
                    note['SAX_CINE'] = f'{len(slice_files)} slices'
                sax_file = os.path.join(sax_file, slice_files[0])
                sax_data = sitk.GetArrayFromImage(sitk.ReadImage(sax_file))
                if len(sax_data) != cine_fps:
                    info += f" {len(sax_data)}fps-sax_cine"
                    note['SAX_CINE'] = f'{len(sax_data)}fps'
            except FileNotFoundError:
                info += ' No_sax_cine'
                note['SAX_CINE'] = 'No_data'
            try:
                lax_file = os.path.join(nifti_path, '4CH_data', niftis[i])
                '''Remove segmentation files'''
                slice_files = exclude_seg_files(os.listdir(lax_file))

                if len(slice_files) > max_slices_lax:
                    info += f' Multiple_4ch_cine_slice_{len(slice_files)}'
                    note['LAX4CH_CINE'] = f'{len(slice_files)} slices'
                lax_file = os.path.join(lax_file, slice_files[0])
                lax_data = sitk.GetArrayFromImage(sitk.ReadImage(lax_file))
                if len(lax_data) != cine_fps:
                    info += f' {len(lax_data)}fps-4ch_cine'
                    note['LAX4CH_CINE'] = f'{len(lax_data)}fps'
            except FileNotFoundError:
                info += ' No_4ch_cine'
                note['LAX4CH_CINE'] = 'No_data'
            try:
                sax_file = os.path.join(nifti_path, 'SAX_LGE_data', niftis[i])
                '''Remove segmentation files'''
                slice_files = exclude_seg_files(os.listdir(sax_file))

                if len(slice_files) >= max_slices_sax:
                    info += f' Multiple_sax_lge_slice_{len(slice_files)}'
                    note['SAX_LGE'] = f'{len(slice_files)} slices'
                sax_file = os.path.join(sax_file, slice_files[0])
                sax_data = sitk.GetArrayFromImage(sitk.ReadImage(sax_file))
            except FileNotFoundError:
                info += ' No_sax_lge'
                note['SAX_LGE'] = 'No_data'
            try:
                lax_file = os.path.join(nifti_path, '4CH_LGE_data', niftis[i])
                '''Remove segmentation files'''
                slice_files = exclude_seg_files(os.listdir(lax_file))

                if len(slice_files) > max_slices_lax:
                    info += f' Multiple_4ch_lge_slice_{len(slice_files)}'
                    note['LAX4CH_LGE'] = f'{len(slice_files)} slices'
                lax_file = os.path.join(lax_file, slice_files[0])
                lax_data = sitk.GetArrayFromImage(sitk.ReadImage(lax_file))
            except FileNotFoundError:
                info += ' No_4ch_lge'
                note['LAX4CH_LGE'] = 'No_data'
            if info != niftis[i]:
                print(f"{info}")
                chart.append(note)
        except:
            print(f'error occured with {nifti_path} {niftis[i]}')
    return chart


'''get spacing info of data'''


def get_data(path='./test2/HCM_new/'):
    """
    The get_data function returns a list of dictionaries, where each dictionary
    contains the patient's name and ID number. The function also contains two 
    variables that are used to determine whether or not the patient has 4-chamber 
    or cine images. If there is no image data for a given patient, then their name 
    and ID will be listed in an empty dictionary.

    :param path: Specify the path of the data
    :return: A list of dictionaries
    :doc-author: airscker-AI
    """
    pats = os.listdir(path)
    all_data = []
    for i in range(len(pats)):
        pat_path = os.path.join(path, pats[i])
        pat_data = {}
        for roots, dirs, files in os.walk(pat_path):
            if len(dirs) == 0:
                for i in range(len(files)):
                    info = pydicom.dcmread(os.path.join(roots, files[i]))
                    pat_data['Name'] = str(info['0010', '0010'].value)
                    pat_data['ID'] = info['0010', '0020'].value
                    if len(files) % 25 == 0:
                        pat_data['Vendor_cine'] = str(
                            info['0008', '0070'].value)
                        if len(files) > 100:
                            try:
                                pat_data['SAX_cine_z_spacing'] = float(
                                    info['0018', '0088'].value)
                            except:
                                pass
                            try:
                                pat_data['SAX_cine_xy_spacing'] = list(
                                    info['0028', '0030'].value)
                            except:
                                pass
                        else:
                            try:
                                pat_data['4CH_cine_z_spacing'] = float(
                                    info['0018', '0088'].value)
                            except:
                                pass
                            try:
                                pat_data['4CH_cine_xy_spacing'] = list(
                                    info['0028', '0030'].value)
                            except:
                                pass
                    # elif len(files):
                    else:
                        pat_data['Vendor_LGE'] = info['0008', '0070'].value
                        if len(files) > 5:
                            try:
                                pat_data['SAX_LGE_z_spacing'] = float(
                                    info['0018', '0088'].value)
                            except:
                                pass
                            try:
                                pat_data['SAX_LGE_xy_spacing'] = list(
                                    info['0028', '0030'].value)
                            except:
                                pass
                        else:
                            try:
                                pat_data['4CH_LGE_z_spacing'] = float(
                                    info['0018', '0088'].value)
                            except:
                                pass
                            try:
                                pat_data['4CH_LGE_xy_spacing'] = list(
                                    info['0028', '0030'].value)
                            except:
                                pass
        print(pat_data)
        all_data.append(pat_data)
    return all_data


def get_info_from_nifti(nifti_path='E:/BaiduNetdiskDownload/DCM_nifti/', check_mode=True):
    """
    The get_info_from_nifti function takes in a path to the nifti files and returns a list of dictionaries.
    Each dictionary contains information about one patient, including their name, spacing for each modality (4CH, 4CH_LGE, SAX and SAX_LGE),
    and whether or not there is an error with the data. The function also checks if all patients have the same resolution for each modality.

    :param nifti_path: Specify the path to the folder containing all nifti files
    :param check_mode: Check whether the spacing of all files in a patient folder are the same
    :return: A list of dictionaries, each dictionary contains the patient name and xyz spacing for all four nifti types
    :doc-author: airscker-AI
    """
    niftis = ['4CH_data', '4CH_LGE_data', 'SAX_data', 'SAX_LGE_data']
    avail_pats = []
    for i in range(len(niftis)):
        try:
            pats = os.listdir(os.path.join(nifti_path, niftis[i]))
            if len(pats) > len(avail_pats):
                avail_pats = pats
        except:
            pass
    print(f'{len(avail_pats)} patients available')
    all_info = []

    bar = tqdm(range(len(avail_pats)), mininterval=1)
    for i in bar:
        pat_info = dict(name=avail_pats[i])
        for j in range(len(niftis)):
            sizes = []
            try:
                pat_folder = os.path.join(nifti_path, niftis[j], avail_pats[i])
                pat_files = exclude_seg_files(os.listdir(pat_folder))
                if check_mode:
                    for k in range(len(pat_files)):
                        data = sitk.ReadImage(
                            os.path.join(pat_folder, pat_files[k]))
                        sizes.append(list(data.GetSpacing()))
                else:
                    data = sitk.ReadImage(
                        os.path.join(pat_folder, pat_files[0]))
                spacing = data.GetSpacing()
                pat_info[f'{niftis[j]}_xyz'] = spacing
            except:
                pass
            if check_mode:
                sizes = np.array(sizes)
                if np.all(np.std(sizes, axis=0)) != 0:
                    print(pat_folder, sizes)
        all_info.append(pat_info)
        # break
    return all_info


'''get the file number of every modality'''


def slice_stats(root_path, save=''):
    """
    The slice_stats function takes a root path to the data and returns a list of dictionaries.
    Each dictionary contains information about the patient, including their name, class (i.e., 'HGG' or 'LGG'), 
    and number of slices for each modality (4CH_data, SAX_data, 4CH_LGE_data and SAX_LGE_data). The function also 
    returns an error if there is more than one slice per modality in any given folder.

    :param root_path: Specify the path to the root folder of your data
    :param save: Save the results of the slice_stats function to a csv file
    :return: A list of dictionaries, where each dictionary contains the information about a patient
    :doc-author: airscker-AI
    """
    assert save == '' or save.endswith('.csv')
    mods = ['4CH_data', 'SAX_data', '4CH_LGE_data', 'SAX_LGE_data']
    pats = []
    info = []
    for i in range(len(mods)):
        try:
            pats_mod = os.listdir(os.path.join(root_path, mods[i]))
            for j in range(len(pats_mod)):
                if pats_mod[j] not in pats:
                    pats.append(pats_mod[j])
        except:
            pass
    # print(root_path,len(pats))
    pats = exclude_seg_files(pats)
    for i in range(len(pats)):
        pat_info = {'Class': root_path.split('/')[-1], 'Name': pats[i]}
        for j in range(len(mods)):
            try:
                pat_folder = os.path.join(root_path, mods[j], pats[i])
                pat_files = exclude_seg_files(os.listdir(pat_folder))
                pat_info[mods[j]] = len(pat_files)
                if '4CH_LGE' in mods[j] and len(pat_files) > 1:
                    print(pat_folder, len(pat_files))
            except:
                pass
        info.append(pat_info)
    if save != '':
        chart = pd.DataFrame(info, columns=[
                             'Class', 'Name', '4CH_data', 'SAX_data', '4CH_LGE_data', 'SAX_LGE_data'])
        chart.to_csv(save)
    return info


'''plot mid/up/down view of sax data'''


def plot_mid_up_down(root, output):
    """
    The plot_mid_up_down function plots the mid, up and down slices of a patient's scan.
    It takes as input the root directory where all of the SAX_data folders are located, and an output directory to save images in.


    :param root: Specify the folder where the images are stored
    :param output: Specify where the images will be saved
    :return: Nothing, but it creates a jpg file in the output directory
    :doc-author: airscker-AI
    """
    try:
        os.makedirs(output)
    except:
        pass
    for roots, dirs, files in os.walk(root):
        if len(dirs) == 0 and 'SAX_data' in roots:
            pat_name = roots.split('/')[-1]
            if 'slice_mid.nii.gz' in files and 'slice_up.nii.gz' in files and 'slice_down.nii.gz' in files:
                mid = sitk.GetArrayFromImage(sitk.ReadImage(
                    os.path.join(roots, 'slice_mid.nii.gz')))
                up = sitk.GetArrayFromImage(sitk.ReadImage(
                    os.path.join(roots, 'slice_up.nii.gz')))
                down = sitk.GetArrayFromImage(sitk.ReadImage(
                    os.path.join(roots, 'slice_down.nii.gz')))
                plt.figure(figsize=(15, 15))
                plt.subplot(131)
                plt.imshow(up[0], cmap='gray')
                plt.subplot(132)
                plt.imshow(mid[0], cmap='gray')
                plt.subplot(133)
                plt.imshow(down[0], cmap='gray')
                plt.savefig(os.path.join(output, f'{pat_name}.jpg'))
