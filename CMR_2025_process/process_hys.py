import os
import shutil
import csv
import re
import pydicom
import SimpleITK as sitk
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm

import multiprocessing
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed

# 将关键函数移到顶层作用域
def process_wrapper(patient_path):
    try:
        process_dicom2nii(patient_path, r"I:\processed_data_hys_new", jpg_root=r"I:\processed_data_hys_new\imgs")
        return True, patient_path
    except Exception as e:
        return False, f"{patient_path} | Error: {str(e)}"

def position(info):
    # Get single dcm's physical position
    cosines = info['0020', '0037'].value
    ipp = info['0020', '0032'].value
    a1 = np.zeros((3,))
    a2 = np.zeros((3,))
    for i in range(3):
        a1[i] = cosines[i]
        a2[i] = cosines[i+3]
    index = np.abs(np.cross(a1, a2))
    return ipp[np.argmax(index)]

def check_dicom_consistency(file_info_map, dcms):
    """检查DICOM文件的尺寸是否一致"""
    ref_rows = file_info_map[dcms[0]]['rows']
    ref_cols = file_info_map[dcms[0]]['cols']
    for dcm in dcms[1:]:
        if file_info_map[dcm]['rows'] != ref_rows or file_info_map[dcm]['cols'] != ref_cols:
            return False
    return True

def rename_dicom(dataroot='./DICOMs'):
    '''
        This function is:
        1. to remove 'DICOM.dcm', which serves as an index for the dataset but is not typically needed for most DICOM processing tasks
        2. to append the suffix '.dcm'
    '''
    for root, dirs, files in os.walk(dataroot):
        if len(dirs)==0:
            for i in range(len(files)):
                if files[i] == 'DICOMDIR.dcm' or files[i] == 'DICOMDIR':
                    os.remove(os.path.join(root, files[i]))
                    print(f'{os.path.join(root, files[i])} deleted')
                if not files[i].endswith('.dcm'):
                    os.rename(os.path.join(root, files[i]), os.path.join(root, f'{files[i]}.dcm'))

def dcm2nii(dcm_files, path_save):
    """Convert Dicom Series Files to a Single NII File
    
    Args:
        dcm_files: list of DICOM file paths (属于同一series)
        path_save: 输出的nii文件路径
    """
    # 直接使用文件列表，无需扫描目录
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(dcm_files)
    
    try:
        image3d = series_reader.Execute()
        image3d.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        sitk.WriteImage(image3d, path_save)
    except Exception as e:
        raise RuntimeError(f"Failed to convert DICOM to NIfTI: {str(e)}")

def nii_to_jpg(nii_path, jpg_path):
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
    im.save(jpg_path)

def process_dicom2nii(dataroot,output,jpg_root,fps=25, thres=100):
    error = []
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
            try:
                # Group DICOM files by series description
                for file in files:
                    try:
                        try:
                            info = pydicom.dcmread(os.path.join(roots, file))
                            rows = info.Rows
                            cols = info.Columns
                            mod = info.SeriesDescription
                            pos_z = position(info)
                        except:
                                
                            if info is not None:
                                del info
                        
                            continue # 跳过无法读取的文件

                        info_truncated = {
                            "pats_id": str(info['0010', '0020'].value),
                            "pats_name": str(info['0010', '0010'].value).replace(' ', '_').replace('^', '_').replace('·','').replace('*', '').replace('?', '').replace('¤', ''),
                            "rows": rows,  # 存储尺寸
                            "cols": cols
                        }
                        del info

                        # 4ch 模态
                        if mod =='B-TFE_BH' or 'B-TFE_4CH' in mod or 'CINE_segmented_LAX_4Ch' in mod or 'cine_4ch' in mod or '4ch_function' in mod or 'FIESTA CINE 4cn' in mod or 'tfl_loc_4-chamber_iPAT' in mod or '4CH_Function' in mod or '4CH FIESTA CINE' in mod or ('4CH' in mod and 'Cine' in mod):
                            four_ch_file_info[os.path.join(roots,file)] = info_truncated
                            
                            if pos_z not in list(four_ch_slice_map.keys()):
                                four_ch_slice_map[pos_z] = [os.path.join(roots,file)]
                            else:
                                four_ch_slice_map[pos_z].append(os.path.join(roots,file))

                        # SAX 模态
                        elif mod =='sBTFE_BH_M2D' or mod =='B-TFE_M2D_SA' or 'CINE_segmented_SAX' in mod or 'cine_sax' in mod or ('SA' in mod and 'Cine' in mod):
                            sax_file_info[os.path.join(roots,file)] = info_truncated
                            if pos_z not in list(sax_slice_map.keys()):
                                sax_slice_map[pos_z] = [os.path.join(roots,file)]
                            else:
                                sax_slice_map[pos_z].append(os.path.join(roots,file))

                        # LGE 模态
                        elif 'PSIR' in mod or 'psir' in mod :
                            sax_lge_file_info[os.path.join(roots,file)] = info_truncated
                            if pos_z not in list(sax_lge_slice_map.keys()):
                                sax_lge_slice_map[pos_z] = [os.path.join(roots,file)]
                            else:
                                sax_lge_slice_map[pos_z].append(os.path.join(roots,file))
                        
                        # T1 模态
                        elif mod.startswith('T1') or mod.startswith('t1'):
                            t1_file_info[os.path.join(roots,file)] = info_truncated
                            if pos_z not in list(t1_slice_map.keys()):
                                t1_slice_map[pos_z] = [os.path.join(roots,file)]
                            else:
                                t1_slice_map[pos_z].append(os.path.join(roots,file))
                        
                        # T2 模态
                        elif mod.startswith('T2') or mod.startswith('t2'):
                            t2_file_info[os.path.join(roots,file)] = info_truncated
                            if pos_z not in list(t2_slice_map.keys()):
                                t2_slice_map[pos_z] = [os.path.join(roots,file)]
                            else:
                                t2_slice_map[pos_z].append(os.path.join(roots,file))
                        else:
                            pass
                    except Exception as e:
                        tqdm.write(f"Error processing {dataroot}: {str(e)}")  # 错误输出
            except:
                print(f'Unknown error occured with {roots}')

    slice_maps = {'four_ch': four_ch_slice_map, 'sax': sax_slice_map, 'sax_lge': sax_lge_slice_map, 't1':t1_slice_map, 't2':t2_slice_map}
    file_info_maps = {'four_ch': four_ch_file_info, 'sax': sax_file_info, 'sax_lge': sax_lge_file_info,'t1':t1_file_info, 't2':t2_file_info}
    for mod in ['four_ch', 'sax', 'sax_lge', 't1', 't2']:
        slice_map = slice_maps[mod]
        slices = sorted(list(slice_map.keys())) # 按照 pos_z 进行排序
        idx = 0
        file_info = file_info_maps[mod]
        for i in range(len(slices)):
            idx += 1
            dcms = slice_map[slices[i]] # 读取当前 pos_z 下的所有 DICOM 文件

            if not check_dicom_consistency(file_info, dcms):
                tqdm.write(f"Skipping inconsistent {mod} series: {dcms[0]}")
                continue

            pats_name = file_info[dcms[0]]['pats_name']
            pats_id = file_info[dcms[0]]['pats_id']

            savepath = os.path.join(output, mod, f'{pats_id}_{pats_name}_RENJI')
            filename = f'{pats_id}_{pats_name}_{idx}.nii.gz'

            if not os.path.exists(savepath):
                os.makedirs(savepath)

            try:
                # 直接使用dcms文件路径列表，无需复制
                dcm2nii(dcms, os.path.join(savepath, filename))
                
                # 生成JPEG预览
                jpg_path = os.path.join(jpg_root, mod + filename.replace('.nii.gz', '.jpg'))
                if mod not in ['sax', 't1', 't2']:
                    nii_to_jpg(os.path.join(savepath, filename), jpg_path)

            except Exception as e:
                tqdm.write(f"Failed to process {filename}: {str(e)}")

def modify_nii_to_jpg(nii_root, jpg_path):
    # 获取该 jpg 文件的文件名
    filename = os.path.basename(jpg_path)
    match = re.match(r'^(four_ch|sax|sax_lge|t1|t2)(A-Z\d+)_(.+)_(\d+)\.jpg$', filename)
    if not match:
        raise ValueError(f"Invalid filename format: {filename}")
        
    mod, pats_id, pats_name, num = match.groups()
    
    # 构建新路径
    nii_dir = os.path.join(nii_root, mod, f"{pats_id}_{pats_name}_RENJI")
    
    # 构建对应的 nii 路径
    nii_filename = f"{pats_id}_{pats_name}_{num}.nii.gz"
    nii_path = os.path.join(nii_dir, nii_filename)
    
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
    im.save(jpg_path)

def process_wrapper_modify_jpg(nii_root, jpg_path):
    try:
        modify_nii_to_jpg(nii_root, jpg_path)
        return True, jpg_path
    except Exception as e:
        return False, f"{jpg_path} | Error: {str(e)}"

if __name__ == '__main__':

    # 修复Windows多进程环境下的特殊设置
    multiprocessing.freeze_support()

    root_dir = r"I:\图像"
    dataroots = [
             r"I:\图像\RENJI\MI\急性\2004300_jiangrenzhen",
             r"I:\图像\RENJI\MI\慢性(含部分未分类亚急性）\1907787_wangaibao\DICOM",
             r"I:\图像\RENJI\DCM\20150319 zhengying\DICOM",
             r"I:\图像\RENJI\MI\急性\3448687_tayier¤aihemaiti",
             ]
    jpg_root = r"I:\processed_data_hys_new\imgs"

    # ============== 根据顶层目录分类收集患者路径 ==============
    # patient_paths = []

    # for datatroot in dataroots:
    #     for cate_folder in os.listdir(datatroot):
    #         cate_path = os.path.join(datatroot, cate_folder)
    #         if not os.path.isdir(cate_path):
    #             continue

    #         # 判断目录类型
    #         for patient_name in os.listdir(cate_path):
    #             patient_path = os.path.join(cate_path, patient_name)
    #             if os.path.isdir(patient_path):
    #                 patient_paths.append(patient_path)

    # for item in patient_paths:
    #     print(item)

    patient_paths = [
             r"I:\图像\RENJI\MI\急性\2004300_jiangrenzhen",
             r"I:\图像\RENJI\MI\慢性(含部分未分类亚急性）\1907787_wangaibao\DICOM",
             r"I:\图像\RENJI\DCM\20150319 zhengying\DICOM",
             r"I:\图像\RENJI\MI\急性\3448687_tayier¤aihemaiti",
    ]
    # ============== 多进程优化 ==============

    # 多进程执行
    max_workers = min(4, os.cpu_count())
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务（避免主模块重载问题）
        futures = {executor.submit(process_wrapper, path): path for path in patient_paths}

        # 进度监控
        with tqdm(total=len(patient_paths), desc="🚀 Processing Patients", unit="patient") as pbar:
            success_count = 0
            failed_log = []
            
            for future in as_completed(futures):
                status, msg = future.result()
                if status:
                    success_count += 1
                else:
                    failed_log.append(msg)
                pbar.update(1)
                pbar.set_postfix_str(f"OK: {success_count}, Failed: {len(failed_log)}")

    # 输出统计信息
    tqdm.write(f"\n✅ Success: {success_count}/{len(patient_paths)}")
    if failed_log:
        tqdm.write("\n❌ Failed cases:")
        for log in failed_log:
            tqdm.write(f"  {log}")



    # nii_root = r"I:\processed_data_hys_new"
    # jpg_root = r"I:\processed_data_hys_new\imgs"

    # # 收集所有需要处理的jpg路径
    # jpg_paths = []
    # for root, dirs, files in os.walk(jpg_root):
    #     for file in files:
    #         if file.endswith('.jpg'):
    #             jpg_paths.append(os.path.join(root, file))

    # # 多线程执行
    # max_workers = min(4, os.cpu_count())  # 控制最大线程数
    # with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     # 提交任务
    #     futures = {executor.submit(process_wrapper_modify_jpg, nii_root, path): path for path in jpg_paths}

    #     # 进度监控
    #     with tqdm(total=len(jpg_paths), desc="🚀 Processing Images", unit="image") as pbar:
    #         success_count = 0
    #         failed_log = []
            
    #         for future in concurrent.futures.as_completed(futures):
    #             status, msg = future.result()
    #             if status:
    #                 success_count += 1
    #             else:
    #                 failed_log.append(msg)
    #             pbar.update(1)
    #             pbar.set_postfix_str(f"OK: {success_count}, Failed: {len(failed_log)}")

    # # 输出统计信息
    # tqdm.write(f"\n✅ Success: {success_count}/{len(jpg_paths)}")
    # if failed_log:
    #     tqdm.write("\n❌ Failed cases:")
    #     for log in failed_log:
    #         tqdm.write(f"  {log}")
