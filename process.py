import pydicom
import numpy as np
import os
import shutil
import SimpleITK as sitk
import csv
from zipfile import ZipFile
from tqdm import tqdm
import cv2
import pandas as pd

def is_4ch(orientation, position):
    list_x = [0, 0, 0]
    list_y = [0, 0, 0]
    list_x[orientation.index(max(orientation[0:3]))] = 1
    list_y[orientation.index(max(orientation[3:6])) - 3] = 1
    return list_x == [1, 0, 0] and list_y[0] == 0 and position[0] < 0 and position[1] < 0

def sax_or_lax(orientation, position):
    list_x = [0,0,0]
    list_y = [0,0,0]
    list_x[orientation.index(max(orientation[0:3]))] = 1
    list_y[orientation.index(max(orientation[3:6])) - 3] = 1
    # print(list_x, list_y)
    if list_x == [1,0,0] and list_y[0] == 0:
        return '4ch'
    elif list_x[0] == 0 and list_y[1] == 0:   # 2ch 3ch 1ch
        if int(position[0]) > 0 and int(position[2]) > 0:
            return '2ch'
    else:
        return 0

def save_cine_with_overlay(dataroot, output):
    error = []
    idx = 0
    for roots, dirs, files in os.walk(dataroot):
        if len(dirs) == 0:
            idx += 1
            roots = os.path.normpath(roots)
            print(idx, roots, len(files))
            id = roots.split('\\')[-1]
            # if id in os.listdir(output+'/SAX_label') and id+'_LAX.nii.gz' in os.listdir(output+'/LAX_label'):
            #     continue
            slice_map = {}
            for i in range(len(files)):
                try:
                    info = pydicom.dcmread(os.path.join(roots, files[i]))
                    overlay_data = info[0x60003000].value
                    position = info[0x00200032].value
                    col = info.Columns
                    row = info.Rows
                    if str(col)+'_'+str(row) not in list(slice_map.keys()):
                        slice_map[str(col)+'_'+str(row)] = [files[i]]
                    else:
                        slice_map[str(col)+'_'+str(row)].append(files[i])
                except:
                    pass
            new_slice_map = {}
            # slices = list(slice_map.keys())
            # print(slices)
            # for i in range(len(slices)):
            #     print(slices[i], len(slice_map[slices[i]]))

            for key in slice_map.keys():
                filenames = slice_map[key]
                if len(filenames) < 200:
                    for filename in filenames:
                        info = pydicom.dcmread(os.path.join(roots, filename))
                        position = info[0x00200032].value
                        pos = int(position[0])
                        if pos not in list(new_slice_map.keys()):
                            new_slice_map[pos] = [filename]
                        else:
                            new_slice_map[pos].append(filename)
                # if len(filenames) >= 200:
                #     # sax
                #     for filename in filenames:
                #         info = pydicom.dcmread(os.path.join(roots, filename))
                #         position = info[0x00200032].value
                #         pos = int(position[0])
                #         if pos not in list(new_slice_map.keys()):
                #             new_slice_map[pos] = [filename]
                #         else:
                #             new_slice_map[pos].append(filename)
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

            slice_map.clear()
            del slice_map

            slices = list(new_slice_map.keys())
            print(slices)
            # for i in range(len(slices)):
            #     print(slices[i], len(new_slice_map[slices[i]]))
            # if 'lax' not in slices:
            #     print('error: no lax!')
            #     error.append(['no lax', roots, id])
            # elif len(slices) < 5:
            #     print('error: slice num error!')
            #     error.append(['slice num error', roots, id])

            for i in range(len(slices)):
                slice_name = slices[i]     # position[0]
                # print(slice_name, len(slice_map[slice_name]))
                if len(new_slice_map[slice_name]) != 50:
                    print('error: frame num error!')
                    error.append(['frame num error', roots, slice_name])
                    # continue

                if slice_name != 'lax':
                    temp_path = os.path.join(output, 'LAX_temp')
                    save_data_path = os.path.join(output, 'LAX_data', id)
                    save_label_path = os.path.join(output, 'LAX_label', id)
                    filename = f'/{id}_LAX_{slice_name}.nii.gz'
                # elif slice_name != 'lax':
                #     temp_path = os.path.join(output, 'SAX_temp', str(slice_name))
                #     save_data_path = os.path.join(output, 'SAX_data', id)
                #     save_label_path = os.path.join(output, 'SAX_label', id)
                #     filename = f'/slice_{i + 1}.nii.gz'
                    if not os.path.exists(save_label_path):
                        os.makedirs(save_label_path)
                else:
                    continue

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
                    print('data save error!',roots, id)
                    error.append(['data save error',roots, id])
                    pass

                try:
                    # save overlay data to nii
                    # mask2nii_path(temp_path, save_label_path + filename)
                    mask2nii_filelist(roots, dcm_files, save_label_path + filename)
                    print(filename, ' saved!')
                except:
                    error.append(['mask save error',roots, id])
                    pass

                shutil.rmtree(temp_path)
            new_slice_map.clear()
            slices.clear()
            del new_slice_map
            del slices
            # try:
            #     shutil.rmtree(os.path.join(output, 'LAX_temp'))
            #     shutil.rmtree(os.path.join(output, 'SAX_temp'))
            # except:
            #     pass

    # out = open('H:/biobank-dicom-SAX-LAX-5w/nifti_data/error.csv', 'a', newline='')
    # csv_write = csv.writer(out, dialect='excel')
    # for i in range(len(error)):
    #     idx_info = error[i]
    #     csv_write.writerow([idx_info[0], idx_info[1], idx_info[2]])
    # out.close()
    return

def dcm2nii(path_read, path_save):
    '''
    ## Convert Dicom Series Files to a Single NII File
    ### Args:
        path_read: The file folder containing dicom series files(No other files exits)
        path_save: The path you save the .nii/.nii.gz data file
    '''
    # GetGDCMSeriesIDs读取序列号相同的dcm文件
    series_id = sitk.ImageSeriesReader.GetGDCMSeriesIDs(path_read)
    # print(series_id)
    # GetGDCMSeriesFileNames读取序列号相同dcm文件的路径，series[0]代表第一个序列号对应的文件
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path_read, series_id[0])
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    image3d = series_reader.Execute()
    image3d.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    sitk.WriteImage(image3d, path_save)

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

def mask2nii_path(path_read, save_full_path):
    order = {}
    dcm_files = os.listdir(path_read)

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

def fix_single_file_fps(path,fps=25):
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
        new_space[2]=frames/float(fps)
        resampled_data=resample_volume(Origin=path,output='',new_spacing=new_space)
        if sitk.GetArrayFromImage(resampled_data).shape[0]!=25:
            print(f'resample failure: {path}/{sitk.GetArrayFromImage(resampled_data).shape[0]} {spacing}/{new_space}')
        else:
            sitk.WriteImage(resampled_data,path)

def getlistfromcsv(input = 'H:/biobank-dicom-SAX-LAX-5w/nifti_data/error_nolax.csv'):
    files = []
    with open(input) as f:
        cs = csv.reader(f)
        for row in cs:
            id = row[0].strip()
            files.append(id)
    return files

def getalldatadict():
    ":return data dict in id format"

    zip_dict = {}
    zip_path = "H:/biobank-nifti-SAX-LAX_normal_all_16_20"

    for dir in os.listdir(zip_path):
        if dir in ['data', 'nifti_data'] or dir.endswith('.zip'):
            continue
        file_path = os.path.join(zip_path, dir)

        for zip_file in os.listdir(file_path):
            if not zip_file.endswith('.zip'):
                continue
            if '9_2_0.zip' in zip_file or '9_3_0.zip' in zip_file:
                filename = zip_file.split('.')[0]
                zip_dict[filename] = os.path.join(file_path, zip_file)

    return zip_dict

def check_abnormal_4ch(root):
    for roots, dirs, files in os.walk(root):
        if len(dirs) == 0 and 'LAX_data' in roots:
            print(roots, len(files))
            for file in files:
                pos = int(file.split('.')[0].split('_')[-1])
                data = sitk.ReadImage(os.path.join(roots, file))
                position = data.GetOrigin()
                direction = data.GetDirection()
                fps = data.GetSize()[2]
                # print(file, fps, position)
                if not(position[0]<0 and position[1]<0):
                    print(file, fps, position)
            #     if slice_num > 100:
            #         print(roots, file)
            #         tar_path = roots.replace('SAX_LGE_data', '4CH_LGE_data')
            #         if not os.path.exists(tar_path):
            #             os.makedirs(tar_path)
            #         filename = roots.split('\\')[-1].replace('GD', str(slice_num)) + '.nii.gz'
            #         print(tar_path, filename)
            #         shutil.move(roots + '/' + file, tar_path + '/' + filename)
            # if len(files) > 1:
            #     print(roots, len(files))
            #     data = sitk.ReadImage(os.path.join(roots, files[0]))
            #     fps = data.GetSize()[2]
            #     if fps:
            #         print(roots, len(files), fps)
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


# file_path = 'G:\CMR-external-china\WHTJ\data_dicom_tj\ARVC/119121105650_SU_CAI_NENG/20191213/4ch-cine'
# save_path = 'I:/CMR-external-cohort/out_check/119121105650_SU_CAI_NENG.nii.gz'
# save_cine_with_overlay(file_path, save_path)

miss_list = getlistfromcsv('G:/NM2023-FW-additional-testing/2-5-data/SAX_LGE_data_loss.csv')
data_path = "G:/NM2023-FW-additional-testing/2-5-data/dicom_data"

info = []
for id in miss_list:
    id_info = [id]
    roots = os.path.join(data_path, id)
    for root, dir, files in os.walk(roots):
        if len(dir) == 0 and '4ch' not in root.lower() and 'cine' not in root.lower() and 'lvef' not in root.lower() and 'lv ef' not in root.lower() and 'cas' not in root.lower():
            id_info.append(root.split('\\')[-1])
    if len(id_info) > 1:
        info.append(id_info)
        print(id_info)

chart = pd.DataFrame(info)
chart.to_csv('G:/NM2023-FW-additional-testing/2-5-data//SAX_LGE_data_loss_dir.csv', index=False)

# # miss_list.sort()
# zip_dict = getalldatadict()
#
# nii_save_path = 'H:/biobank-dicom-SAX-LAX-5w/nifti_data/LAX_label'
# # temp_path = 'D:/data_temp'
# file_list = os.listdir(nii_save_path)
#
# csv_list = []
# for file in file_list:
#     if '20209' in file:
#         newname = file.replace('20209', '20208')
#         try:
#             os.remove(os.path.join(nii_save_path, newname))
#         except:
#             pass
#         os.rename(os.path.join(nii_save_path, file), os.path.join(nii_save_path, newname))


# exist_list = os.listdir("H:/biobank-dicom-SAX-LAX-5w/nifti_data\LAX_data")
#
# error = []
# for i in tqdm(range(len(miss_list))):
#     filename = miss_list[i].replace('20208', '20209')
#     # if filename in exist_list:
#     #     continue
#     try:
#         file_path = zip_dict[filename]
#     except:
#         continue
#
#     zipfile = ZipFile(file_path)
#     try:
#         print(i, file_path)
#
#         zip_out_path = os.path.join(temp_path, filename)
#         if not os.path.exists(zip_out_path):
#             os.makedirs(zip_out_path)
#
#         print('Zip file Extracting...')
#         zipfile.extractall(zip_out_path)
#
#         print('Read and processing...')
#         save_cine_with_overlay(zip_out_path, nii_save_path)
#
#         shutil.rmtree(zip_out_path)
#     except:
#         error.append(filename)
#         pass



# dcm2nii(file_path, save_path)
# fix_single_file_fps(save_path)