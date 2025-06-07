import os
import shutil
import csv
import pydicom
import SimpleITK as sitk
import numpy as np
from PIL import Image
import pandas as pd

def position(file):
    # Get single dcm's physical position
    cosines = file[0x20, 0x37].value
    ipp = file[0x20, 0x32].value
    a1 = np.zeros((3,))
    a2 = np.zeros((3,))
    for i in range(3):
        a1[i] = cosines[i]
        a2[i] = cosines[i+3]
    index = np.abs(np.cross(a1, a2))
    return ipp[np.argmax(index)]

def dcm2nii(path_read, path_save):
    # Convert Dicom Series Files to a Single NII File
    series_id = sitk.ImageSeriesReader.GetGDCMSeriesIDs(path_read)
    if not series_id:
        raise RuntimeError(f"No DICOM series found in {path_read}")
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path_read, series_id[0])
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    image3d = series_reader.Execute()
    image3d.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    sitk.WriteImage(image3d, path_save)

def nii_to_png(nii_path, png_path):
    # 读取 NIfTI 文件
    img = sitk.ReadImage(nii_path)
    arr = sitk.GetArrayFromImage(img)  #(slices, H, W)
    num_slices = arr.shape[0]
    if num_slices >= 3:
        indices = np.linspace(0, num_slices-1, 3, dtype=int)
        selected_slices = [arr[i] for i in indices]
    else:
        mid = num_slices // 2
        selected_slices = [arr[mid]]

    processed = []
    for slice_img in selected_slices:
        slice_min = np.min(slice_img)
        slice_max = np.max(slice_img)
        slice_norm = (slice_img - slice_min) / (slice_max - slice_min + 1e-5) * 255
        processed.append(slice_norm.astype(np.uint8))

    if len(processed) > 1:
        combined = np.hstack(processed)  # (H, W*3)
    else:
        combined = processed[0]         # (H, W)

    # save
    im = Image.fromarray(combined)
    im.save(png_path)


def process_dicom2nii(dataroot,output,png_root,fps=25, thres=100):
    error = []
    temp_folders = {'4ch cine': '4ch_temp', 'sax cine': 'sax_temp', 'sax lge': 'lge_temp'}
    four_ch_slice_map = {}
    four_ch_file_info = {}
    sax_slice_map = {}
    sax_file_info = {}
    sax_lge_slice_map = {}
    sax_lge_file_info = {}
    t1_slice_map = {}
    t1_file_info = {}
    t2_slice_map = {}
    t2_file_info = {}
    for roots, dirs, files in os.walk(dataroot):
        try:
            for file in files:
                if file.endswith('.dcm') or file.endswith('.DCM') or file.startswith('I'):
                    try:
                        info = pydicom.dcmread(os.path.join(roots, file))
                        mod = info.SeriesDescription

                        if mod =='B-TFE_BH' or 'B-TFE_4CH' in mod or 'CINE_segmented_LAX_4Ch' in mod or 'cine_4ch' in mod or '4ch_function' in mod or 'FIESTA CINE 4cn' in mod or 'tfl_loc_4-chamber_iPAT' in mod or '4CH_Function' in mod or '4CH FIESTA CINE' in mod or ('4CH' in mod and 'Cine' in mod) or ('cine_tf2d13_retro_iPAT' in mod and info.SliceThickness <= 6 )  or('FIESTA CINE' in mod and info.SliceThickness <= 6) or  ( 'cine' in mod and 'sa' not in mod and 'SA' not in mod) or ('sBTFE_BH' in mod and not 'M2D' in mod )or 'tf2d12_retro_iPAT' in mod or ('4CH' in mod and 'Function' in mod):
                        # if '4ch' in mod and 'cine' in mod:
                        #     four_ch_file_info[file] = info
                            four_ch_file_info[os.path.join(roots,file)] = info
                            pos_z = int(position(info))
                            if pos_z not in list(four_ch_slice_map.keys()):
                                # four_ch_slice_map[sn] = [file]
                                four_ch_slice_map[pos_z] = [os.path.join(roots,file)]
                            else:
                                four_ch_slice_map[pos_z].append(os.path.join(roots,file))
                        elif mod =='sBTFE_BH_M2D' or mod =='B-TFE_M2D_SA' or 'CINE_segmented_SAX' in mod or 'cine_sax' in mod or ('SA' in mod and 'Cine' in mod) or ('cine_tf2d13_retro_iPAT' in mod and info.SliceThickness > 6.0) or mod=='B-TFE_BH_M2D' or('FIESTA CINE' in mod and info.SliceThickness > 6) or ('tf2d12_retro_iPAT' in mod and 'sa' in mod) or 'SA_Function 12Slice' in mod or 'tfl_loc_short-axis_iPAT' in mod or 'sBTFE_BH_M2D' in mod or('sa' in mod.lower() and 'cine' in mod.lower()):
                        # elif 'sa' in mod and 'cine' in mod:
                        #     sax_file_info[file] = info
                            sax_file_info[os.path.join(roots,file)] = info
                            pos_z = int(position(info))
                            if pos_z not in list(sax_slice_map.keys()):
                                sax_slice_map[pos_z] = [os.path.join(roots,file)]
                            else:
                                sax_slice_map[pos_z].append(os.path.join(roots,file))
                        elif 'PSIR' in mod or 'psir' in mod :
                        # elif 'sa' in mod and 'ps' in mod:
                        #     sax_lge_file_info[file] = info
                            sax_lge_file_info[os.path.join(roots,file)] = info
                            pos_z = int(position(info))
                            if pos_z not in list(sax_lge_slice_map.keys()):
                                sax_lge_slice_map[pos_z] = [os.path.join(roots,file)]
                            else:
                                sax_lge_slice_map[pos_z].append(os.path.join(roots,file))
                        elif mod.startswith('T1') or mod.startswith('t1'):
                            t1_file_info[os.path.join(roots,file)] = info
                            pos_z = int(position(info))
                            if pos_z not in list(t1_slice_map.keys()):
                                t1_slice_map[pos_z] = [os.path.join(roots,file)]
                            else:
                                t1_slice_map[pos_z].append(os.path.join(roots,file))
                        elif mod.startswith('T2') or mod.startswith('t2'):
                            t2_file_info[os.path.join(roots,file)] = info
                            pos_z = int(position(info))
                            if pos_z not in list(t2_slice_map.keys()):
                                t2_slice_map[pos_z] = [os.path.join(roots,file)]
                            else:
                                t2_slice_map[pos_z].append(os.path.join(roots,file))
                        else:
                            print(mod)
                    except Exception as e:
                        print(e)
        except:
            print(f'unknow error occured with {roots}')

    slice_maps = {'four_ch': four_ch_slice_map, 'sax': sax_slice_map, 'sax_lge': sax_lge_slice_map, 't1':t1_slice_map, 't2':t2_slice_map}
    file_info_maps = {'four_ch': four_ch_file_info, 'sax': sax_file_info, 'sax_lge': sax_lge_file_info,'t1':t1_file_info, 't2':t2_file_info}
    for mod in ['four_ch', 'sax', 'sax_lge', 't1', 't2']:
        slice_map = slice_maps[mod]
        slices = sorted(list(slice_map.keys()))
        print(slices)
        idx = 0
        file_info = file_info_maps[mod]
        for i in range(len(slices)):
            dcms = slice_map[slices[i]]
            if ((len(dcms) < 10 or len(dcms) > 50) and mod in ['four_ch', 'sax'] or len(dcms) > 50):
                continue
            idx += 1
            dcms = slice_map[slices[i]]
            pats_name = str(file_info[dcms[0]]['0010', '0010'].value).replace(' ', '_').replace('^', '_').replace('·',
                                                                                                                  '').replace(
                '*', '')
            pats_id = file_info[dcms[0]]['0010', '0020'].value
            filetag = str(file_info[dcms[0]]['0008', '103e'].value)

            temp_folder = os.path.join(output, 'temp')
            savepath = os.path.join(output, mod, f'{pats_id}_{pats_name}_AZ')
            filename = f'{pats_id}_{pats_name}_{idx}.nii.gz'

            for j in range(len(dcms)):
                if not os.path.exists(temp_folder):
                    os.makedirs(temp_folder)
                shutil.copyfile(dcms[j], os.path.join(temp_folder, os.path.basename(dcms[j])))

            if not os.path.exists(savepath):
                os.makedirs(savepath)
            try:
                dcm2nii(temp_folder, os.path.join(savepath, filename))
                print(f'{filename} saved into {savepath}', len(dcms))
                png_path = os.path.join(png_root, mod+filename.replace('.nii.gz', '.jpg'))
                if not mod == 'sax' and not mod == 't1' and not mod == 't2':
                    nii_to_png(os.path.join(savepath, filename), png_path)
            except:
                pass
            shutil.rmtree(temp_folder)

if __name__ == '__main__':
    root_dir = r"F:\HCM\AZ-HCM\1.AZHCM2021原始图像"
    os.makedirs("D:\processed_data_AZ_HCM\imgs",exist_ok=True)
    png_root = "D:\processed_data_AZ_HCM\imgs"
    for cate in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, cate)):
            cur_root = os.path.join(root_dir,  cate)
            process_dicom2nii(cur_root, "D:\processed_data_AZ_HCM",png_root)


