import os
import shutil
import csv
import pydicom
import SimpleITK as sitk
import numpy as np
from PIL import Image
import pandas as pd

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
    img = sitk.ReadImage(nii_path)
    arr = sitk.GetArrayFromImage(img)  # (slices, H, W)
    mid = arr.shape[0] // 2
    slice_img = arr[mid]
    slice_img = (slice_img - np.min(slice_img)) / (np.ptp(slice_img) + 1e-5) * 255
    slice_img = slice_img.astype(np.uint8)
    im = Image.fromarray(slice_img)
    im.save(png_path)

def process_dicom2nii(dataroot,output, df,png_root,fps=25, thres=100):
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
        if len(dirs) == 0:
            # four_ch_slice_map = {}
            # four_ch_file_info = {}
            # sax_slice_map = {}
            # sax_file_info = {}
            # sax_lge_slice_map = {}
            # sax_lge_file_info = {}
            try:
                # Group DICOM files by series description
                for file in files:
                    if file.endswith('.dcm') or file.endswith('.DCM'):
                        try:
                            info = pydicom.dcmread(os.path.join(roots, file))
                            mod = info.SeriesDescription

                            if mod =='B-TFE_BH' or 'B-TFE_4CH' in mod or 'CINE_segmented_LAX_4Ch' in mod or 'cine_4ch' in mod or '4ch_function' in mod or 'FIESTA CINE 4cn' in mod or 'tfl_loc_4-chamber_iPAT' in mod or '4CH_Function' in mod or '4CH FIESTA CINE' in mod or ('4CH' in mod and 'Cine' in mod):
                            # if '4ch' in mod and 'cine' in mod:
                            #     four_ch_file_info[file] = info
                                four_ch_file_info[os.path.join(roots,file)] = info
                                sn = int(info['0020', '0032'].value[0])
                                if sn not in list(four_ch_slice_map.keys()):
                                    # four_ch_slice_map[sn] = [file]
                                    four_ch_slice_map[sn] = [os.path.join(roots,file)]
                                else:
                                    four_ch_slice_map[sn].append(os.path.join(roots,file))
                            elif mod =='sBTFE_BH_M2D' or mod =='B-TFE_M2D_SA' or 'CINE_segmented_SAX' in mod or 'cine_sax' in mod or ('SA' in mod and 'Cine' in mod):
                            # elif 'sa' in mod and 'cine' in mod:
                            #     sax_file_info[file] = info
                                sax_file_info[os.path.join(roots,file)] = info
                                sn = int(info['0020', '0032'].value[0])
                                if sn not in list(sax_slice_map.keys()):
                                    sax_slice_map[sn] = [os.path.join(roots,file)]
                                else:
                                    sax_slice_map[sn].append(os.path.join(roots,file))
                            elif 'PSIR' in mod or 'psir' in mod :
                            # elif 'sa' in mod and 'ps' in mod:
                            #     sax_lge_file_info[file] = info
                                sax_lge_file_info[os.path.join(roots,file)] = info
                                sn = int(info['0020', '0032'].value[0])
                                if sn not in list(sax_lge_slice_map.keys()):
                                    sax_lge_slice_map[sn] = [os.path.join(roots,file)]
                                else:
                                    sax_lge_slice_map[sn].append(os.path.join(roots,file))
                            elif mod.startswith('T1') or mod.startswith('t1'):
                                t1_file_info[os.path.join(roots,file)] = info
                                sn = int(info['0020', '0032'].value[0])
                                if sn not in list(t1_slice_map.keys()):
                                    t1_slice_map[sn] = [os.path.join(roots,file)]
                                else:
                                    t1_slice_map[sn].append(os.path.join(roots,file))
                            elif mod.startswith('T2') or mod.startswith('t2'):
                                t2_file_info[os.path.join(roots,file)] = info
                                sn = int(info['0020', '0032'].value[0])
                                if sn not in list(t2_slice_map.keys()):
                                    t2_slice_map[sn] = [os.path.join(roots,file)]
                                else:
                                    t2_slice_map[sn].append(os.path.join(roots,file))
                            else:
                                print(mod)
                        except Exception as e:
                            print(e)
            except:
                print(f'unknow error occured with {roots}')

            # slice_maps={'four_ch':four_ch_slice_map, 'sax':sax_slice_map, 'sax_lge':sax_lge_slice_map}
            # file_info_maps ={'four_ch':four_ch_file_info, 'sax':sax_file_info, 'sax_lge':sax_lge_file_info}
            # for mod in ['four_ch','sax','sax_lge']:
            #     slice_map = slice_maps[mod]
            #     slices = sorted(list(slice_map.keys()))
            #     print(slices)
            #     idx = 0
            #     file_info = file_info_maps[mod]
            #     for i in range(len(slices)):
            #         idx += 1
            #         dcms = slice_map[slices[i]]
            #         pats_name = str(file_info[dcms[0]]['0010', '0010'].value).replace(' ', '_').replace('^', '_').replace('·',
            #                                                                                                               '').replace(
            #             '*', '')
            #         pats_id = file_info[dcms[0]]['0010', '0020'].value
            #         filetag = str(file_info[dcms[0]]['0008', '103e'].value)
            #
            #         temp_folder = os.path.join(output, 'temp')
            #         savepath = os.path.join(output, mod, f'{pats_id}_{pats_name}_RJ')
            #         filename = f'{pats_id}_{pats_name}_{idx}.nii.gz'
            #
            #         for j in range(len(dcms)):
            #             if not os.path.exists(temp_folder):
            #                 os.makedirs(temp_folder)
            #             shutil.copyfile(os.path.join(roots, dcms[j]), os.path.join(temp_folder, dcms[j]))
            #
            #         if not os.path.exists(savepath):
            #             os.makedirs(savepath)
            #         try:
            #             dcm2nii(temp_folder, os.path.join(savepath, filename))
            #             print(f'{filename} saved into {savepath}', len(dcms))
            #         except:
            #             pass
            #         shutil.rmtree(temp_folder)
        else:
            #print(f'========Error========{roots}')
            continue
    csv_rows = []
    slice_maps = {'four_ch': four_ch_slice_map, 'sax': sax_slice_map, 'sax_lge': sax_lge_slice_map, 't1':t1_slice_map, 't2':t2_slice_map}
    file_info_maps = {'four_ch': four_ch_file_info, 'sax': sax_file_info, 'sax_lge': sax_lge_file_info,'t1':t1_file_info, 't2':t2_file_info}
    for mod in ['four_ch', 'sax', 'sax_lge', 't1', 't2']:
        slice_map = slice_maps[mod]
        slices = sorted(list(slice_map.keys()))
        print(slices)
        idx = 0
        file_info = file_info_maps[mod]
        for i in range(len(slices)):
            idx += 1
            dcms = slice_map[slices[i]]
            pats_name = str(file_info[dcms[0]]['0010', '0010'].value).replace(' ', '_').replace('^', '_').replace('·',
                                                                                                                  '').replace(
                '*', '')
            pats_id = file_info[dcms[0]]['0010', '0020'].value
            filetag = str(file_info[dcms[0]]['0008', '103e'].value)

            temp_folder = os.path.join(output, 'temp')
            savepath = os.path.join(output, mod, f'{pats_id}_{pats_name}_HEI_FOUR')
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
                    new_row = {
                        'img_name': mod + filename.replace('.nii.gz', '.jpg'),
                        'nii_path': os.path.join(savepath, filename),
                        '4ch': '',
                        '3ch': '',
                        '2ch': '',
                        'sax': ''
                    }
                    df = df.append(new_row, ignore_index=True)

            except:
                pass
            shutil.rmtree(temp_folder)

if __name__ == '__main__':
    root_dir = r"I:\图像"
    csv_path = 'nii_png_index.csv'
    columns = ['img_name','nii_path', '4ch', '3ch', '2ch', 'sax']
    df = pd.DataFrame(columns=columns)
    os.makedirs("I:\processed_data_hys\imgs")
    png_root = "I:\processed_data_hys\imgs"
    for cate_folder in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, cate_folder)):
            cur_root = os.path.join(root_dir, cate_folder)
            for pat in os.listdir(cur_root):
                if os.path.isdir(os.path.join(cur_root, pat)):
                    root = os.path.join(cur_root, pat)
                    process_dicom2nii(root, "I:\processed_data_hys",df,png_root)

    df.to_csv(csv_path, index=False)

