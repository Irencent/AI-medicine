import os
import shutil
import csv
import pydicom
import SimpleITK as sitk
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm

def sumPatients(dataroot):
    all_pats_id = []
    # 遍历文件夹
    for roots, dirs, files in os.walk(dataroot):
        if len(dirs) == 0:
            try:
                file = files[0]
                info = pydicom.dcmread(os.path.join(roots, file))
                pats_id = str(info['0010', '0020'].value)
                # pats_id 不为 None 且不在 all_pats_id 中
                if pats_id is not None and pats_id not in all_pats_id:
                    all_pats_id.append(pats_id)

            except:
                # 打印路径
                print("Error reading DICOM file in path:", os.path.join(roots, file))
                continue

    return len(all_pats_id)

root = 'I:\图像'
dataroots = [r'I:\图像\normal_2', 
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

patients_num = {}
for dataroot in dataroots:
    dataroot_patient_num = sumPatients(dataroot)
    patients_num[dataroot] = dataroot_patient_num

print("patients_num:", patients_num)

