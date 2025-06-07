import os
import shutil
import csv
import pydicom
import SimpleITK as sitk
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# 将关键函数移到顶层作用域
def process_wrapper(patient_path):
    try:
        process_dicom2nii(patient_path, r"I:\CMR-China-new\AZ_data\nifti_data")
        return True, patient_path
    except Exception as e:
        return False, f"{patient_path} | Error: {str(e)}"

def check_dicom_consistency(file_info_map, dcms):
    """检查DICOM文件的尺寸是否一致"""
    ref_rows = file_info_map[dcms[0]]['rows']
    ref_cols = file_info_map[dcms[0]]['cols']
    for dcm in dcms[1:]:
        if file_info_map[dcm]['rows'] != ref_rows or file_info_map[dcm]['cols'] != ref_cols:
            return False
    return True


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
    
def process_dicom2nii(dataroot,output,fps=25, thres=100):
    two_ch_slice_map = {}
    two_ch_file_info = {}
    three_ch_slice_map = {}
    three_ch_file_info = {}
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
                        if info is not None:
                            del info

                        # 3ch 模态
                        if '3ch' in mod.lower():
                            three_ch_file_info[os.path.join(roots,file)] = info_truncated
                                
                            if pos_z not in list(three_ch_slice_map.keys()):
                                three_ch_slice_map[pos_z] = [os.path.join(roots,file)]
                            else:
                                three_ch_slice_map[pos_z].append(os.path.join(roots,file))

                        # 2ch 模态
                        elif '2ch' in mod.lower():
                            two_ch_file_info[os.path.join(roots,file)] = info_truncated
                                
                            if pos_z not in list(two_ch_slice_map.keys()):
                                two_ch_slice_map[pos_z] = [os.path.join(roots,file)]
                            else:
                                two_ch_slice_map[pos_z].append(os.path.join(roots,file))
                    except Exception as e:
                        tqdm.write(f"Error processing {dataroot}: {str(e)}")  # 错误输出
            except:
                print(f'Unknown error occured with {roots}')

    slice_maps = {'2CH_data': two_ch_slice_map, '3CH_data': three_ch_slice_map}
    file_info_maps = {'2CH_data': two_ch_file_info, '3CH_data': three_ch_file_info}

    for mod in slice_maps.keys():
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

            savepath = os.path.join(output, mod, f'{pats_id}_{pats_name}_AZ')
            filename = f'{pats_id}_{pats_name}_{idx}.nii.gz'

            if not os.path.exists(savepath):
                os.makedirs(savepath)

            try:
                # 直接使用dcms文件路径列表，无需复制
                dcm2nii(dcms, os.path.join(savepath, filename))
            except Exception as e:
                tqdm.write(f"Failed to process {filename}: {str(e)}")

if __name__ == '__main__':
    # 修复Windows多进程环境下的特殊设置
    multiprocessing.freeze_support()

    dataroot = r'I:\CMR-China-new\AZ_data\AZ_processed'

    # ============== 根据顶层目录分类收集患者路径 ==============
    patient_paths = []

    for patient_name in os.listdir(dataroot):
        patient_path = os.path.join(dataroot, patient_name)
        if os.path.isdir(patient_path):
            patient_paths.append(patient_path)

    # for item in patient_paths:
    #     print(item)

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

    
    