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

def filter_dicom_files(dataroot, pats_id_to_filter):
    """
    Filter DICOM files based on patient ID.
    
    Args:
        dataroot (str): The root directory containing DICOM files.
        pats_id_to_filter (list): List of patient IDs to filter out.
        
    Returns:
        list: List of filtered DICOM file paths.
    """
    filtered_files = []
    
    for root, dirs, files in os.walk(dataroot):
        if len(dirs) == 0:
            for file in files:
                try:
                    info = pydicom.dcmread(os.path.join(root, file))
                    pats_id = str(info['0010', '0020'].value)

                    for pats in pats_id_to_filter:
                    
                        if pats in pats_id:
                            # Skip files with patient IDs in the filter list
                            filtered_files.append(os.path.join(root, file))

                    if info:
                        del info
                        
                except Exception as e:
                    tqdm.write(f"Error processing {os.path.join(root, file)}: {str(e)}")
    
    return filtered_files

def process_wrapper(patient_path):
    try:
        filter_dicom_files(patient_path, pats_id_to_filter = ['1928492', '2004300', '1907787'])
        return True, patient_path
    except Exception as e:
        return False, f"{patient_path} | Error: {str(e)}"

if __name__ == "__main__":

    # 修复Windows多进程环境下的特殊设置
    multiprocessing.freeze_support()

    root_dir = r"I:\图像"
    dataroots = [
             r'I:\图像\normal_2',
             r'I:\图像\RENJI\AMY',
             r'I:\图像\RENJI\ARVC',
             r'I:\图像\RENJI\DCM',
             r'I:\图像\RENJI\HCM',
             r'I:\图像\RENJI\HTN',
             r'I:\图像\RENJI\LVNC',
             r'I:\图像\RENJI\MI',
             r'I:\图像\RENJI\normal_1',
             r'I:\图像\RENJI\TIS',
             ]
    jpg_root = r"I:\processed_data_hys_new\imgs"

    # ============== 根据顶层目录分类收集患者路径 ==============
    patient_paths = []

    for datatroot in dataroots:
        for cate_folder in os.listdir(datatroot):
            cate_path = os.path.join(datatroot, cate_folder)
            if not os.path.isdir(cate_path):
                continue

            # 判断目录类型
            for patient_name in os.listdir(cate_path):
                patient_path = os.path.join(cate_path, patient_name)
                if os.path.isdir(patient_path):
                    patient_paths.append(patient_path)

    # ============== 多进程优化 ==============
    all_filtered_files = []  # 新增：全局收集过滤结果
    unique_pats = set()      # 新增：用于患者ID去重

    # 使用进程池来处理每个患者路径
    max_workers = min(4, os.cpu_count())
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务（避免主模块重载问题）
        futures = {executor.submit(process_wrapper, patient_path): patient_path for patient_path in patient_paths}
        
        # 使用进度条来显示处理进度
        with tqdm(total=len(patient_paths), desc="🚀 Processing Patients", unit="patient") as pbar:
            success_count = 0
            failed_log = []
            
            for future in as_completed(futures):
                status, msg, filtered = future.result()
                if status:
                    success_count += 1
                    all_filtered_files.extend(filtered)
                else:
                    failed_log.append(msg)
                pbar.update(1)
                pbar.set_postfix_str(f"OK: {success_count}, Failed: {len(failed_log)}")


    print("找到的结果:", all_filtered_files)