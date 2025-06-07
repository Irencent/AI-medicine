import os
import shutil
import nibabel as nib
import SimpleITK as sitk
import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle as pkl
import pandas as pd
from PIL import Image

def sort_dict(dictionary):
    sorted_keys = sorted(dictionary)
    new_map = {}
    for i in range(len(sorted_keys)):
        new_map[sorted_keys[i]] = dictionary[sorted_keys[i]]
    return new_map

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
    image3d.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    sitk.WriteImage(image3d, path_save)

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

def resample_dataset(dataroot='./DICOMs',output='./New_dicom/'):
    """Resample the dataset to a new directory .

    Args:
        dataroot (str, optional):The root path of all patients' dicom files. Defaults to './DICOMs'.
        output (str, optional):The output path of resampled dicom dataset. Defaults to './New_dicom/'.
    """
    error=[]
    count=0
    for paths,dirs,files in os.walk(dataroot):
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
    new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]
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

def resample_dataset2(spacing=[0.994, 1.826], datapath='E:/VST_fusion/dataset'):
    error = []
    for space in spacing:
        for roots, dirs, files in os.walk(datapath):
            if '1144739' not in roots:
                continue
            if len(dirs) == 0:
                # print(roots)
                # files = exclude_seg_files(os.listdir(roots))
                for i in range(len(files)):
                    data_path = os.path.join(roots, files[i])
                    # print(data_path)
                    pre_data = sitk.ReadImage(data_path)
                    data_spacing = list(pre_data.GetSpacing())
                    data_spacing[0] = space
                    data_spacing[1] = space
                    new_data = resample_volume(Origin=data_path, new_spacing=data_spacing, output='')
                    if len(sitk.GetArrayFromImage(new_data)) != 50 and 'LGE' not in data_path:
                        print(f'frame resample error: {data_path} {sitk.GetArrayFromImage(new_data).shape}')
                        error.append(data_path)
                    else:
                        save_path = roots.replace('nifti_data', f'nifti_spacing_{space}_data')
                        # save_path = save_path.replace('dataset', f'dataset_spacing_{space}')
                        # save_path=save_path.replace('E:','F:')
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        save_path = os.path.join(save_path, files[i])
                        sitk.WriteImage(new_data, save_path)
                        print(f'{save_path} saved')
    return error

def fix_single_file_fps(path,fps=30):
    """Resample frame to 25
        Args:
            path (str, optional): The path of resampled data. Defaults to './New_dicom/xx.nii.gz'.
            fps : target resample frames
    """
    data=sitk.ReadImage(path)
    spacing=list(data.GetSpacing())
    array=sitk.GetArrayFromImage(data)
    frames=len(array)
    if frames!=fps:
        new_space=spacing.copy()
        new_space[2]=new_space[2]*frames/float(fps)
        resampled_data=resample_volume(Origin=path,output='',new_spacing=new_space)
        if sitk.GetArrayFromImage(resampled_data).shape[0]!=fps:
            print(f'resample failure: {path}/{sitk.GetArrayFromImage(resampled_data).shape[0]} {spacing}/{new_space}')
        else:
            sitk.WriteImage(resampled_data,path)

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

def fix_fps(fps=30,nifti_path='E:/VST_fusion/ARVC_1_nifti/',fix=True):
    mods=['SAX_data','4CH_data']
    # mods = ['4CH_data']
    for i in range(len(mods)):
        for roots,dirs,files in os.walk(os.path.join(nifti_path,mods[i])):
            if '1707640' not in roots:
                continue
            if len(dirs)==0:
                print(roots)
                frame_list=[]
                files=exclude_seg_files(os.listdir(roots))
                for j in range(len(files)):
                    data=sitk.ReadImage(os.path.join(roots,files[j]))
                    spacing=list(data.GetSpacing())
                    array=sitk.GetArrayFromImage(data)
                    frames=len(array)
                    # print(frames)
                    frame_list.append(frames)
                    # print(f'{roots} {files[j]} {frames}')
                    if frames!=fps:
                        print(f'{roots} {files[j]} {frames}')
                        if frames < 100:
                            fix_single_file_fps(os.path.join(roots,files[j]))
                        # else:
                        #     fix_seriesnum(roots)
                        # if fix:
                        #     '''FIX FPS HERE'''
                        #     new_space=spacing.copy()
                        #     new_space[2]=frames/float(fps)
                        #     resampled_data=resample_volume(Origin=os.path.join(roots,files[j]),output='',new_spacing=new_space)
                        #     if sitk.GetArrayFromImage(resampled_data).shape[0]!=25:
                        #         print(f'resample failure: {roots} {files[j]} {frames}/{sitk.GetArrayFromImage(resampled_data).shape[0]} {spacing}/{new_space}')
                        #     else:
                        #         sitk.WriteImage(resampled_data,os.path.join(roots,files[j]))
                if np.std(frame_list)!=0:
                    print(f'frame_error {roots} {frame_list}')

def exclude_seg_files(file_list=[],segmentations=['Segmentation.nii','Segmentation.nii.gz','.DS_Store']):
    for seg_file in segmentations:
        if seg_file in file_list:
            file_list.remove(seg_file)
    return file_list

def find_error(root_path='E:/BaiduNetdiskDownload/PAH_nifti', out_name='', cine_fps=25, max_slices_sax=13,
               max_slices_lax=1):
    '''
    ## Find out:
        - The lost data
        - Multi slices of 4CH CINE/LGE
        - Error with cine_fps

    Automatically exclude the segmentation files
    '''
    nifti_path = ['']
    chart = []
    mods = ['4CH_data', 'SAX_data', '4CH_LGE_data', 'SAX_LGE_data']

    for cls in nifti_path:
        niftis = []
        for i in range(len(mods)):
            try:
                pats_mod = os.listdir(os.path.join(root_path, cls, mods[i]))
                for j in range(len(pats_mod)):
                    if pats_mod[j] not in niftis:
                        niftis.append(pats_mod[j])
            except:
                pass

        for i in range(len(niftis)):
            try:
                note = {}
                info = niftis[i]
                note['NAME'] = niftis[i].split('.')[0]
                note['MOD'] = cls
                # print(niftis[i])
                try:
                    sax_file = os.path.join(root_path, cls, 'SAX_data', niftis[i])
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
                    lax_file = os.path.join(root_path, cls, '4CH_data', niftis[i])
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
                    sax_file = os.path.join(root_path, cls, 'SAX_LGE_data', niftis[i])
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
                    lax_file = os.path.join(root_path, cls, '4CH_LGE_data', niftis[i])
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
                print(f'error occured with {cls} {niftis[i]}')

    chart = pd.DataFrame(chart)
    chart.to_csv(out_name, index=False)
    return chart

def slice_stats(root_path):
    mods=['4CH_data','SAX_data','SAX_LGE_data']
    info=[]

    nifti_path = ['']
    for cls in nifti_path:
        pats=[]
        for i in range(len(mods)):
            try:
                pats_mod = os.listdir(os.path.join(root_path, cls, mods[i]))
                for j in range(len(pats_mod)):
                    if pats_mod[j] not in pats:
                        pats.append(pats_mod[j])
            except:
                pass

        for i in range(len(pats)):
            pat_info = {'Class': cls, 'Name': pats[i]}
            for j in range(len(mods)):
                try:
                    pat_folder = os.path.join(root_path, cls, mods[j], pats[i])
                    pat_files = exclude_seg_files(os.listdir(pat_folder))
                    pat_info[mods[j]] = len(pat_files)
                    if '4CH_LGE' in mods[j] and len(pat_files) > 1:
                        print(pat_folder, len(pat_files))
                except:
                    pass
            info.append(pat_info)

    return info

def rename_mid(root,show=True):
    for roots, dirs, files in os.walk(root):
        # if len(dirs) != 0:
        #     print(roots)
        if len(dirs) == 0 and 'SAX_data' in roots:
            print(roots)
            files = exclude_seg_files(files)
            if len(files) > 2 and ('slice_up.nii.gz' not in files or 'slice_mid.nii.gz' not in files or 'slice_down.nii.gz' not in files):
                slice_map = {}
                for i in range(len(files)):
                    # slice_num = int(files[i].split('.')[0].split('_')[-1])
                    data = sitk.ReadImage(os.path.join(roots, files[i]))
                    # print(data.GetOrigin()[-1])
                    slice_map[data.GetOrigin()[-1]] = files[i]
                    # slice_map[slice_num] = files[i]
                slice_pos = sorted(list(slice_map.keys()))
                if len(files) >= 5:
                    os.rename(os.path.join(roots, slice_map[slice_pos[len(
                        slice_pos) // 2]]), os.path.join(roots, 'slice_mid.nii.gz'))
                    os.rename(os.path.join(roots, slice_map[slice_pos[len(
                        slice_pos) // 2 - 2]]), os.path.join(roots, 'slice_down.nii.gz'))
                    os.rename(os.path.join(roots, slice_map[slice_pos[len(
                        slice_pos) // 2 + 2]]), os.path.join(roots, 'slice_up.nii.gz'))
                elif len(files)<5:
                    print(roots, len(files))
                    os.rename(os.path.join(roots, slice_map[slice_pos[len(
                        slice_pos) // 2]]), os.path.join(roots, 'slice_mid.nii.gz'))
                    os.rename(os.path.join(roots, slice_map[slice_pos[len(
                        slice_pos) // 2 - 1]]), os.path.join(roots, 'slice_down.nii.gz'))
                    os.rename(os.path.join(roots, slice_map[slice_pos[len(
                        slice_pos) // 2 + 1]]), os.path.join(roots, 'slice_up.nii.gz'))

def mask2nii_filelist(path_read, dcm_files, save_full_path):
    order = {}
    # dcm_files = list(map(lambda x:root + x, files))

    info = pydicom.dcmread(os.path.join(path_read, dcm_files[0]))
    spacing = info.PixelSpacing[0]
    origin = info.ImagePositionPatient

    for dcm_file in dcm_files:
        dcm = pydicom.dcmread(os.path.join(path_read, dcm_file))
        col = dcm.Columns
        row = dcm.Rows
        instancenum = dcm[0x00200013].value
        order[instancenum] = os.path.join(path_read, dcm_file)

    orderlist = list(order)
    orderlist.sort()
    data = np.zeros((len(dcm_files), row, col))

    for i in range(len(orderlist)):
        dcm = pydicom.dcmread(order[orderlist[i]])

        overlay_data = dcm[0x60003000].value
        overlay_frames = dcm[0x60000015].value
        bits_allocated = dcm[0x60000100].value

        np_dtype = np.dtype('uint8')
        length_of_pixel_array = len(overlay_data)
        expected_length = row * col
        if bits_allocated == 1:
            expected_bit_length = expected_length
            bit = 0
            arr = np.ndarray(shape=(length_of_pixel_array * 8), dtype=np_dtype)
            for byte in overlay_data:
                for bit in range(bit, bit + 8):
                    arr[bit] = byte & 1
                    byte >>= 1
                bit += 1
            arr = arr[:expected_bit_length]
        if overlay_frames == 1:
            arr = arr.reshape(row, col)
        arr = arr * 255

        data[i, :, :] = arr

    out = sitk.GetImageFromArray(data)
    out.SetOrigin(origin)
    out.SetSpacing([spacing, spacing, 1])
    out.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    sitk.WriteImage(out, save_full_path)

def save_cine_with_overlay(dataroot, output):
    roots = os.path.normpath(dataroot)
    files = os.listdir(roots)

    # print(roots, len(files))
    id = roots.split('\\')[-1]

    slice_map = {}
    new_slice_map = {}
    error = []
    for i in range(len(files)):
        try:
            info = pydicom.dcmread(os.path.join(roots, files[i]))
            overlay_data = info[0x60003000].value
            position = info[0x00200032].value
            col = info.Columns
            row = info.Rows
            if str(col) + '_' + str(row) not in list(slice_map.keys()):
                slice_map[str(col) + '_' + str(row)] = [files[i]]
            else:
                slice_map[str(col) + '_' + str(row)].append(files[i])
        except:
            pass

    for key in slice_map.keys():
        filenames = slice_map[key]
        if len(filenames) >= 150:
            # sax
            for filename in filenames:
                info = pydicom.dcmread(os.path.join(roots, filename))
                position = info[0x00200032].value
                pos = int(position[0])
                if pos not in list(new_slice_map.keys()):
                    new_slice_map[pos] = [filename]
                else:
                    new_slice_map[pos].append(filename)
        # elif len(filenames) == 100:
        #     if 'lax' in new_slice_map.keys():
        #         continue
        #     # 4ch and others
        #     for filename in filenames:
        #         info = pydicom.dcmread(os.path.join(roots, filename))
        #         position = info[0x00200032].value
        #         orientation = info[0x00200037].value
        #         if is_4ch(orientation, position):
        #             if 'lax' not in list(new_slice_map.keys()):
        #                 new_slice_map['lax'] = [filename]
        #             else:
        #                 new_slice_map['lax'].append(filename)
        # elif len(filenames) == 50:
        #     # 4ch or else
        #     info = pydicom.dcmread(os.path.join(roots, filenames[0]))
        #     position = info[0x00200032].value
        #     orientation = info[0x00200037].value
        #     if is_4ch(orientation, position):
        #         new_slice_map['lax'] = filenames

    slices = list(new_slice_map.keys())
    print(slices)
    # if 'lax' not in slices:
    #     print('error: no lax!')
    #     error.append(['no lax', roots, id])
    # elif len(slices) < 5:
    #     print('error: slice num error!')
    #     error.append(['slice num error', roots, id])

    for i in range(len(slices)):
        slice_name = slices[i]

        if slice_name == 'lax':
            save_label_path = os.path.join(output, 'LAX_label', id)
            filename = f'_LAX.nii.gz'
        else:
            temp_path = os.path.join('D:/data_temp/SAX_temp', str(slice_name))
            save_data_path = os.path.join(output, 'SAX_data', id)
            save_label_path = os.path.join(output, 'SAX_label', id)
            # if not os.path.exists(save_label_path):
            #     os.makedirs(save_label_path)
            filename = f'/slice_{i + 1}.nii.gz'

        dcm_files = new_slice_map[slices[i]]
        for file in dcm_files:
            if not os.path.exists(temp_path):
                os.makedirs(temp_path)
            shutil.copyfile(os.path.join(roots, file), os.path.join(temp_path, file))

        try:
            # save image data to nii
            if not os.path.exists(save_data_path):
                os.makedirs(save_data_path)
            dcm2nii(temp_path, save_data_path + filename)
            print(save_data_path + filename, ' saved!')
        except:
            error.append(['data save error',roots, id])
            pass

        try:
            # save overlay data to nii
            # mask2nii_path(temp_path, save_label_path + filename)
            mask2nii_filelist(roots, dcm_files, save_label_path + filename)
            print(filename, ' saved!')
        except:
            error.append(['mask save error', roots, id])
            pass

        shutil.rmtree(temp_path)

    return

def save_sax_cine(dataroot='./New_dicom/',output='./',fps=25,thres=100):
    error=[]
    for roots,dirs,files in os.walk(dataroot):
        if len(dirs)==0 and 'sa' in roots.lower():
            print(roots)
            try:
                slice_map={}
                file_info={}

                for i in range(len(files)):
                    try:
                        info=pydicom.dcmread(os.path.join(roots,files[i]))
                        file_info[files[i]]=info
                        sn=int(info['0020','0032'].value[0])
                        if sn not in list(slice_map.keys()):
                            slice_map[sn]=[files[i]]
                        else:
                            slice_map[sn].append(files[i])
                    except:
                        continue
                slices=sorted(list(slice_map.keys()))
                slices.reverse()
                print(slices)
                idx = 0
                for i in range(len(slices)):
                    idx += 1
                    dcms=slice_map[slices[i]]
                    pats_name=str(file_info[dcms[0]]['0010','0010'].value).replace(' ','_').replace('^', '_').replace('·', '').replace('*','')
                    pats_id=file_info[dcms[0]]['0010','0020'].value
                    filetag=str(file_info[dcms[0]]['0008','103e'].value)

                    # print(filetag)


                    if len(dcms) >= 10:
                        temp_folder=os.path.join(output,'sax_temp')
                        savepath=os.path.join(output,'SAX_data',f'{pats_id}_{pats_name}_AZ')
                        filename=f'slice_{idx}.nii.gz'

                        for j in range(len(dcms)):
                            if not os.path.exists(temp_folder):
                                os.makedirs(temp_folder)
                            shutil.copyfile(os.path.join(roots,dcms[j]),os.path.join(temp_folder,dcms[j]))

                        if not os.path.exists(savepath):
                            os.makedirs(savepath)

                        try:
                            dcm2nii(temp_folder,os.path.join(savepath,filename))
                            print(f'{filename} saved into {savepath}', len(dcms))
                        except:
                            pass
                        shutil.rmtree(temp_folder)
            except Exception as e:
                print(e)
                print(f'unknow error occured with {roots}')
                error.append(roots)
    try:
        shutil.rmtree(os.path.join(output,'sax_temp'))
    except:
        pass
    return error

def save_lge(dataroot='./New_dicom/',output='./',fps=25,thres=5):

    error=[]
    for roots,dirs,files in os.walk(dataroot):
        if len(dirs)==0 and 'sa' not in roots.lower() and 'psir' in roots.lower() and '4c' not in roots.lower():
            slice_map={}
            file_info={}
            bmp_files = []
            try:
                for i in range(len(files)):
                    if not files[i].endswith('dcm'):
                        if files[i].endswith('bmp') and 'combine' not in files[i] and 'dir' not in files[i] and 'mask' not in files[i] and 'p.bmp' not in files[i]:
                            bmp_files.append(files[i])
                            # print(bmp_files)
                    else:
                        try:
                            info=pydicom.dcmread(os.path.join(roots,files[i]))
                            file_info[files[i]]=info
                            sn=info.SeriesNumber
                            if sn not in list(slice_map.keys()):
                                slice_map[sn]=[files[i]]
                            else:
                                slice_map[sn].append(files[i])
                        except:
                            pass
                slices=list(slice_map.keys())   # [series num1, series num2, ...]
                for i in range(len(slices)):
                    dcms=slice_map[slices[i]]
                    pats_name=str(file_info[dcms[0]]['0010','0010'].value).replace(' ','_').replace('^', '_').replace('*','')
                    pats_id=file_info[dcms[0]]['0010','0020'].value
                    filetag=str(file_info[dcms[0]]['0008','103e'].value)
                    # if 'bh' in filetag.lower():
                    #     continue
                    print(filetag, 'files: ', len(files), '  fps: ', len(dcms))
                    if len(files) >= 3:
                        temp_folder = os.path.join(output, 'sax_lge_temp')
                        savepath = os.path.join(output, 'SAX_LGE_data', f'{pats_id}_{pats_name}_AZ')
                        savepath_lbl = os.path.join(output, 'SAX_LGE_label', f'{pats_id}_{pats_name}_AZ')
                        filename = f'{pats_id}_{pats_name}_{slices[i]}.nii.gz'
                        filename_lbl = f'{pats_id}_{pats_name}.nii.gz'

                        for j in range(len(dcms)):
                            if not os.path.exists(temp_folder):
                                os.makedirs(temp_folder)
                            shutil.copyfile(os.path.join(roots,dcms[j]),os.path.join(temp_folder,dcms[j]))
                        if not os.path.exists(savepath):
                            os.makedirs(savepath)
                        try:
                            dcm2nii(temp_folder,os.path.join(savepath,filename))
                            print(f'{filename} saved into {savepath}')
                        except Exception as e:
                            print(e)
                            print(f'unknow error occured with {filename} and {savepath}')
                        shutil.rmtree(temp_folder)
            except Exception as e:
                print(e)
                print(f'unknow error occured with {roots}')
                error.append(roots)

            # bmp_files.sort()
            # arr_list = []
            # for bmp in bmp_files:
            #     image = Image.open(os.path.join(roots,bmp))
            #     pixels = list(image.getdata())
            #     width, height = image.size
            #     data = [[pixel for pixel in pixels[i * width: (i + 1) * width]] for i in range(height)]
            #     arr = np.array(data, dtype=np.float32)
            #     try:
            #         arr[np.where(arr == 1.)] = 80
            #         arr[np.where(arr == 2.)] = 170
            #     except:
            #         pass
            #     # print(np.unique(arr), os.path.join(savepath_lbl,filename_lbl))
            #     arr_list.append(arr)
            #
            # nifti_arr = np.array(arr_list)
            # if not os.path.exists(savepath_lbl):
            #     os.makedirs(savepath_lbl)
            # save_data = sitk.GetImageFromArray(nifti_arr)
            # save_data.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
            # sitk.WriteImage(save_data, os.path.join(savepath_lbl,filename_lbl))
            #
            # print(f'{filename_lbl} saved into {savepath_lbl}')

    try:
        shutil.rmtree(os.path.join(output,'sax_lge_temp'))
    except:
        pass
    return error

def save_4ch_cine(dataroot='./New_dicom/',output='./', mod='CINE-SA', fps=25,thres=100):
    error=[]
    for roots,dirs,files in os.walk(dataroot):
        if len(dirs)==0 and 'sa' not in roots.lower() and 'psir' not in roots.lower():
            print(roots)
            try:
                slice_map={}
                file_info={}
                for i in range(len(files)):
                    try:
                        info=pydicom.dcmread(os.path.join(roots,files[i]))
                        file_info[files[i]]=info
                        # sn=info.SeriesNumber
                        sn=int(info['0020','0032'].value[0])
                        if sn not in list(slice_map.keys()):
                            slice_map[sn]=[files[i]]
                        else:
                            slice_map[sn].append(files[i])
                    except:
                        continue
                slices=sorted(list(slice_map.keys()))
                print(slices)
                idx = 0
                for i in range(len(slices)):
                    idx += 1
                    dcms=slice_map[slices[i]]
                    pats_name=str(file_info[dcms[0]]['0010','0010'].value).replace(' ','_').replace('^', '_').replace('·', '').replace('*','')
                    pats_id=file_info[dcms[0]]['0010','0020'].value
                    filetag=str(file_info[dcms[0]]['0008','103e'].value)

                    # print(pats_name, pats_id, filetag, len(files), len(dcms))
                    if ('4ch' in filetag.lower() or '4 ch' in filetag.lower()) or ('sa' not in filetag.lower() and 'psir' not in filetag.lower()):
                        # print(filetag, len(files), len(dcms))
                    # if len(files):
                        temp_folder = os.path.join(output, '4ch_temp')
                        savepath = os.path.join(output, '4CH_data', f'{pats_id}_{pats_name}_AZ')
                        filename = f'{pats_id}_{pats_name}_{idx}.nii.gz'

                        for j in range(len(dcms)):
                            if not os.path.exists(temp_folder):
                                os.makedirs(temp_folder)
                            shutil.copyfile(os.path.join(roots,dcms[j]),os.path.join(temp_folder,dcms[j]))

                        if not os.path.exists(savepath):
                            os.makedirs(savepath)
                        try:
                            dcm2nii(temp_folder,os.path.join(savepath,filename))
                            print(f'{filename} saved into {savepath}', len(dcms))
                        except:
                            pass
                        shutil.rmtree(temp_folder)

            except Exception as e:
                print(e)
                print(f'unknow error occured with {roots}')
                error.append(roots)

    try:
        shutil.rmtree(os.path.join(output,'4ch_temp'))
    except:
        pass
    return error

def rename_dicom(dataroot='./DICOMs'):
    for root,dirs,files in os.walk(dataroot):
        if len(dirs)==0:
            print(root)
        # if True:
            for i in range(len(files)):
                if files[i]=='DICOMDIR.dcm':
                    os.remove(os.path.join(root,files[i]))
                    print(f'{os.path.join(root,files[i])} deleted')
                # if files[i].endswith('.dcm'):
                #     os.rename(os.path.join(root, files[i]), os.path.join(root, files[i].replace('.dcm','')))
                elif not files[i].endswith('.dcm'):
                    os.rename(os.path.join(root,files[i]),os.path.join(root,f'{files[i]}.dcm'))

def check_size(dataroot='/Users/airskcer/Downloads/DCM_nifti-20230107/'):
    error={}
    folders=[]
    for roots,dirs,files in os.walk(dataroot):
        if len(dirs)==0 and 'SAX_data' in roots and 'SAX_data_label' not in roots:
            folders.append(roots)
    bar=tqdm(range(len(folders)))
    for i in bar:
        files=exclude_seg_files(os.listdir(folders[i]))
        if len(files)==1:
            continue
        sizes=[]
        for j in range(len(files)):
            data=sitk.ReadImage(os.path.join(folders[i],files[j]))
            sizes.append(sitk.GetArrayFromImage(data).shape)
        sizes=np.array(sizes)
        if np.sum(np.std(sizes,axis=0))!=0:
            error[folders[i]]=sizes
    for key in error:
        print(key,error[key])
    return error

def error_process(error_map: dict):
    """
    The error_process function is used to find the files that have different shapes and move them into a new folder.


    :param error_map: dict: Store the error information
    :return: The file names and the shape of the files that have an error
    :doc-author: Trelent
    """
    for path in error_map.keys():
        map = {}
        files = exclude_seg_files(os.listdir(path))
        for i in range(len(files)):
            map[files[i]] = np.array(list(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(path, files[i]))).shape))
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
                    target = path.replace('nii_data', 'error_shape_data')
                    # print(target)
                    # if 'SAX_LGE_data' in path:
                    #     target = path.replace('SAX_LGE_data', '4CH_LGE_data')
                    try:
                        os.makedirs(target)
                    except:
                        pass
                    try:
                        shutil.move(os.path.join(path,key),target)
                    except:
                        print(f'{target} already exists')
                    # os.remove(os.path.join(path,key))

def reslice_lge(root, fps=9):
    for roots, dirs, files in os.walk(root):
        if len(dirs) == 0:
            files = exclude_seg_files(files)
            # data_map = {}
            data = sitk.ReadImage(os.path.join(roots, files[0]))
            # if data.GetSize()[2] == fps:
            #     continue
            # new_array = sitk.GetArrayFromImage(data)
            # print(new_array.shape)
            # print(data.GetSize())

            target_Size = (data.GetSize()[0], data.GetSize()[1], fps)  # 目标图像大小  [x,y,z]
            target_Spacing = data.GetSpacing()  # 目标的体素块尺寸    [x,y,z]
            target_origin = data.GetOrigin()  # 目标的起点 [x,y,z]
            target_direction = data.GetDirection()  # 目标的方向 [冠,矢,横]=[z,y,x]
            # # itk的方法进行resample
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(data)  # 需要重新采样的目标图像
            # # 设置目标图像的信息
            resampler.SetSize(target_Size)  # 目标图像大小
            resampler.SetOutputOrigin(target_origin)
            resampler.SetOutputDirection(target_direction)
            resampler.SetOutputSpacing(target_Spacing)
            # # 根据需要重采样图像的情况设置不同的dype
            resampler.SetOutputPixelType(sitk.sitkFloat32)  # 线性插值用于PET/CT/MRI之类的，保存float32
            resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            itk_img_resampled = resampler.Execute(data)  # 得到重新采样后的图像
            # print(itk_img_resampled.GetSize())

            # origin = data.GetOrigin()
            # spacing = data.GetSpacing()
            # new_spacing = list(spacing)
            # new_spacing[2] = spacing[2] * len(files) / float(fps)
            # direction = data.GetDirection()
            # # pixelid = data.GetPixelID()
            # # size = data.GetSize()
            # new_data = sitk.GetImageFromArray(new_array)
            # new_data.SetDirection(direction)
            # new_data.SetOrigin(origin)
            # new_data.SetSpacing(spacing)
            # new_size = [new_array.shape[1], new_array.shape[2], fps]
            #
            # resampled_image = sitk.Resample(new_data, new_size, sitk.Transform(), sitk.sitkLinear, origin, new_spacing,
            #                                 direction, 0, new_data.GetPixelID())

            #
            # for i in range(len(files)):
            #     # print(os.path.join(roots,files[i]))
            #     data = sitk.ReadImage(os.path.join(roots, files[i]))
            #     # print(sitk.GetArrayFromImage(data).shape)
            #     data_map[data.GetOrigin()[-1]] = data
            #     # print(data.GetOrigin())
            # data_map = sort_dict(data_map)
            # new_array = []
            # print(data_map)
            # for key in data_map.keys():
            #     array = sitk.GetArrayFromImage(data_map[key])
            #     print(array.shape)
            #     if array.shape[0] > 1:
            #         array = np.array([array[array.shape[0] // 2]])
            #         print(array.shape)
            #     new_array.append(array)
            # # new_array = np.array(new_array)
            # new_array = np.array(new_array).squeeze()
            # new_array = np.expand_dims(new_array, axis=-1)
            # print(new_array.shape)
            # origin = data_map[key].GetOrigin()
            # spacing = data_map[key].GetSpacing()
            # new_spacing = list(spacing)
            # new_spacing[2] = spacing[2] * len(files) / float(fps)
            # direction = data_map[key].GetDirection()
            # pixelid = data_map[key].GetPixelID()
            # size = data_map[key].GetSize()
            # new_data = sitk.GetImageFromArray(new_array)
            # new_data.SetDirection(direction)
            # new_data.SetOrigin(origin)
            # new_data.SetSpacing(spacing)
            # new_size = [new_array.shape[1], new_array.shape[2], fps]
            # resampled_image = sitk.Resample(new_data, new_size, sitk.Transform(), sitk.sitkLinear, origin, new_spacing,
            #                                 direction, 0, new_data.GetPixelID())
            #
            # resampled_array = sitk.GetArrayFromImage(resampled_image)
            # print(resampled_array.shape)
            # resampled_image = sitk.GetImageFromArray(resampled_image)
            return itk_img_resampled

def batch_reslice(root, fps=9, delete=True, mod='SAX_LGE_data'):
    error = []
    available_data = []
    for roots, dirs, files in os.walk(root):
        if len(dirs) == 0 and mod in roots and len(files) > 0:
            available_data.append(roots)
    bar = tqdm(range(len(available_data)))
    for i in bar:
        roots = available_data[i].replace('\\', '/')
        original_files = os.listdir(roots)
        pat_name = roots.split('/')[-1]
        if os.path.exists(os.path.join(roots, f'{pat_name}_fps_{fps}.nii.gz')):
            continue
        # new_data = reslice_lge(root=roots, fps=fps)
        try:
            new_data = reslice_lge(root=roots, fps=fps)
        except:
            print(f'Reslice failure with {roots}')
            error.append(roots)
            continue
        if sitk.GetArrayFromImage(new_data).shape[0] != fps:
            print(f'ERROR occured with {roots}: {fps}/{sitk.GetArrayFromImage(new_data).shape[0]}')
            error.append(roots)
        else:
            if not os.path.exists(roots):
                os.makedirs(roots)
            sitk.WriteImage(new_data, os.path.join(
                roots, f'{pat_name}_fps_{fps}.nii.gz'))
            if delete:
                # shutil.rmtree(roots)
                for file in original_files:
                    os.remove(os.path.join(roots, file))
    return error

def check_data_loss(nifti_path, dcm_path):
    total_id_dcm = {}
    for batch in os.listdir(dcm_path):
        for roots, dirs, files in os.walk(os.path.join(dcm_path, batch)):
            if len(dirs) == 0 and ('CINE-SA' in roots or 'CINE-4CH' in roots) and len(files):
                # print(os.path.join(roots, files[0]))
                info = pydicom.dcmread(os.path.join(roots, files[0]))
                pa_name = str(info['0010','0010'].value).replace(' ','_').replace('^', '_').replace('·', '').replace('*', '')
                pa_id = str(info['0010','0020'].value)
                total_id_dcm[pa_id] = [pa_id, pa_name, '', '','', '']

    mods = ['SAX_data', '4CH_data', 'SAX_LGE_data', 'SAX_LGE_label']
    for i in range(len(mods)):
        mod = mods[i]
        print(f'Checking mod {mod} ...')
        mod_path = os.path.join(nifti_path, mod)
        for id_name in tqdm(os.listdir(mod_path)):
            # print(id_name)
            id = id_name.split('_')[0]
            if id in total_id_dcm.keys():
                file_num = len(os.listdir(os.path.join(mod_path, id_name)))
                total_id_dcm[id][i + 2] = str(file_num)
                # print(os.path.join(mod_path, id_name))
                if file_num:
                    file_path = os.listdir(os.path.join(mod_path, id_name))[0]
                    # print(file_path)
                    fps = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(mod_path, id_name, file_path))).shape[0]
                    total_id_dcm[id][i + 2] = str(file_num)+'_'+str(fps)

    info = []
    for key in total_id_dcm.keys():
        info.append(total_id_dcm[key])


    return info

def split_data(path, fps, mode='sax'):
    file = os.listdir(path)
    data = sitk.ReadImage(os.path.join(path, file[0]))
    filename = file[0].split('.')[0]
    array = sitk.GetArrayFromImage(data)
    # print(array.shape)
    slices = int(array.shape[0] / fps)

    target_Spacing = data.GetSpacing()  # 目标的体素块尺寸    [x,y,z]
    target_origin = data.GetOrigin()  # 目标的起点 [x,y,z]
    target_direction = data.GetDirection()  # 目标的方向 [冠,矢,横]=[z,y,x]

    for slice in range(slices):
        new_array = array[slice * fps:slice * fps + fps, :, :]
        new_data = sitk.GetImageFromArray(new_array)
        new_data.SetOrigin(target_origin)
        new_data.SetDirection(target_direction)
        new_data.SetSpacing(target_Spacing)
        if mode == 'sax':
            sitk.WriteImage(new_data, os.path.join(path, f'slice_{slice + 1}.nii.gz'))
        if mode == 'lge':
            sitk.WriteImage(new_data, os.path.join(path, f'{filename}_{slice + 1}.nii.gz'))
    os.remove(os.path.join(path, file[0]))

def split_cine(path, slices, keep1slice=False):
    file = os.listdir(path)
    data = sitk.ReadImage(os.path.join(path, file[0]))
    filename = file[0].split('.')[0]
    array = sitk.GetArrayFromImage(data)
    print(array.shape)
    fps = int(array.shape[0] / slices)
    for slice in range(slices):
        if keep1slice and slice != 1:
            continue
        new_array = array[slice * fps:slice * fps + fps, :, :]
        new_data = sitk.GetImageFromArray(new_array)
        sitk.WriteImage(new_data, os.path.join(path, f'{filename}_{slices - slice}.nii.gz'))

    os.remove(os.path.join(path, file[0]))

def check_abnormal_lge(root):
    # root = 'H:/CMR-external-cohort/nii_data/'
    for roots, dirs, files in os.walk(root):
        if len(dirs) == 0:
            # file = exclude_seg_files(os.listdir(roots))
            if 'SAX_LGE' in roots:
                if len(files) != 1:
                    print(roots, len(files))
                    for file in files:
                        fps = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(roots, file))).shape[0]
                        print(file, fps)
                        if fps <= 3:
                            os.remove(os.path.join(roots, file))
                    # merge_lge_frame(roots)
                    # ser_num1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(roots, files[0]))).shape[0]
                    # ser_num2 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(roots, files[1]))).shape[0]
                    # ser_num1 = int(files[0].split('.')[0].split('_')[-1])
                    # ser_num2 = int(files[1].split('.')[0].split('_')[-1])
                    # print(ser_num1, ser_num2)
                    # roots = os.path.normpath(roots)
                    # if ser_num1 < 5:
                    #     movefile(roots, files[0])
                    # else:
                    #     movefile(roots, files[1])

                # if len(files) == 1:
                #     fps = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(roots, files[0]))).shape[0]
                #     # print(roots, files[0], fps)
                #     if fps == 24 or fps == 30:
                #         print(roots, files[0], fps)
                #         split_cine(roots, slices=3, keep1slice=True)
                #     if fps < 8:
                #         print(roots, files[0], fps)
                #         # os.remove(os.path.join(roots, files[i]))
                #     if fps > 10 and fps % 3 == 0:
                #         print(roots, files[0], fps)
                #         split_data(roots, fps=int(fps/3), mode='lge')
                        # os.remove(os.path.join(roots, files[i]))
                if len(files) == 0:
                    print(roots)
                    shutil.rmtree(roots)

def merge_lge_frame(path):
    files = os.listdir(path)
    def number(string):
        num = int(string.split('.')[0].split('_')[-1])
        return num
    files.sort(key=number)
    filename = files[0].split('.')[0]

    data = sitk.ReadImage(os.path.join(path, files[0]))
    target_Spacing = data.GetSpacing()  # 目标的体素块尺寸    [x,y,z]
    target_origin = data.GetOrigin()  # 目标的起点 [x,y,z]
    target_direction = data.GetDirection()  # 目标的方向 [冠,矢,横]=[z,y,x]

    sample_data = sitk.GetArrayFromImage(data)
    merge_data = np.zeros((len(files), sample_data.shape[1], sample_data.shape[2]))

    for i in range(len(files)):
        data = sitk.ReadImage(os.path.join(path, files[i]))
        array = sitk.GetArrayFromImage(data)
        try:
            merge_data[i,:,:] = array
        except:
            print("shape error!")

    new_data = sitk.GetImageFromArray(merge_data)

    new_data.SetOrigin(target_origin)
    new_data.SetDirection(target_direction)
    new_data.SetSpacing(target_Spacing)

    sitk.WriteImage(new_data, os.path.join(path, f'{filename}_{len(files)}.nii.gz'))
    for file in files:
        os.remove(path + '/' + file)

def movefile(path, filename):
    # print(path, filename)
    tar_path = path.replace('SAX_LGE_data', '4CH_LGE_data')
    if not os.path.exists(tar_path):
        os.makedirs(tar_path)
    # newname = path.split('\\')[-1].replace('_HEB', '.nii.gz')
    shutil.move(path + '\\' + filename, tar_path + '\\' + filename)

def check_abnormal_4ch(root):
    for roots, dirs, files in os.walk(root):
        if len(dirs) == 0 and '4CH_data' in roots:
            print(roots, len(files), files)
            if len(files) != 1:
                print(roots)
                # for file in files:
                #     data = sitk.ReadImage(os.path.join(roots, file))
                #     fps = data.GetSize()[2]
                #     print(file, fps)
                    # slice_num = int(file.split('.')[0].split('_')[-1])
                    # if slice_num > 100:
                    #     print(roots, file)
                    #     tar_path = roots.replace('SAX_LGE_data', '4CH_LGE_data')
                    #     if not os.path.exists(tar_path):
                    #         os.makedirs(tar_path)
                    #     filename = roots.split('\\')[-1].replace('GD', str(slice_num)) + '.nii.gz'
                    #     print(tar_path, filename)
                    #     shutil.move(roots + '/' + file, tar_path + '/' + filename)
            # if len(files) == 0:
            #     print(roots, len(files))
            #     shutil.rmtree(roots)
                # data = sitk.ReadImage(os.path.join(roots, files[0]))
                # fps = data.GetSize()[2]
                # if fps != 25:
                #     print(roots, len(files), fps)
                # if fps%25 == 0:
                #     split_data(roots, fps=25, mode='sax')
                # if (fps > 25):
                #     split_data(roots, fps=25)
            # if len(files) == 2:
            #     print(roots, len(files))
            #     roots = os.path.normpath(roots)
            #     ser_num1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(roots, files[0]))).shape[0]
                # ser_num2 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(roots, files[1]))).shape[0]
                # ser_num1 = int(files[0].split('.')[0].split('_')[-1])
                # ser_num2 = int(files[1].split('.')[0].split('_')[-1])
                # print(ser_num1, ser_num2)
                # if ser_num1 == 25:
                #     movefile(roots, files[0])
                    # os.remove(os.path.join(roots, files[0]))
                # else:
                #     movefile(roots, files[1])
                    # os.remove(os.path.join(roots, files[1]))



if __name__ == '__main__':
    root_path = "I:/CMR-China-new/RJ_data/raw_dcmdata_old"
    sub_paths = ['标签第三批(175)-全', '第二批149-全', '第一批-179']
    new_path = "I:/CMR-China-new/RJ_data/raw_dcmdata_rj/"
    save_path = "I:/CMR-China-new/RJ_data/nifti_data"

    ## 1. process raw dicom
    # rename_dicom(root_path)

    resample_dataset(root_path, new_path)

    # fix_seriesnum(new_path)

    '''## 2. save dicom to nifti

    # depends on overlay type
    # save_cine_with_overlay(new_path, save_path)
    save_4ch_cine(new_path, save_path)
    save_lge(new_path, save_path)

    # 3. check and fix (manual)
    check_abnormal_lge(root=save_path)

    check_abnormal_4ch(root=save_path)

    error = check_size(dataroot=save_path)
    error_process(error_map=error)

    # 4. fix frames, rename SAX cine, normalize LGE frames
    fix_fps(nifti_path = save_path)

    rename_mid(save_path)

    batch_reslice(root=save_path)

    # 5. final step, resample, only when all problems were fixed
    error = resample_dataset2(datapath=save_path, spacing=[1.826])

    # **. check data and print info csv
    info = check_data_loss(nifti_path=save_path, dcm_path=new_path)
    # print(info)
    chart = pd.DataFrame(info, columns=['ID', 'Name', 'SAX_data', '4CH_data', 'SAX_LGE_data', 'SAX_LGE_label'])
    chart.to_csv('I:/CMR-China-new/RJ_data/nifti_data/check_loss.csv', index=False)
    chart.head(10)
    # error chart
    chart = find_error(root_path=save_path, out_name='G:/NM2023-FW-additional-testing/1-data/error.csv')
    # info chart
    all_info=slice_stats(save_path)
    print(len(all_info))
    chart = pd.DataFrame(all_info, columns=['Class', 'Name', '4CH_data', 'SAX_data', 'SAX_LGE_data'])
    chart.to_csv('G:/NM2023-FW-additional-testing/2-5-data/info.csv', index=False)
    chart.head(10)
'''
