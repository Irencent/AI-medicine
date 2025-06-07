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
from pypinyin import lazy_pinyin

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

def update_dataframe(AccessionNumber, PatientName, markers):
    """Update DataFrame with new entry or modify existing one"""
    global df
    # 使用逻辑或同时匹配examID和放射编号
    mask_id = (df['examID'] == AccessionNumber) | (df['放射编号'] == AccessionNumber) 
    # 匹配姓名
    mask_name = lazy_pinyin(df['病人姓名']).replace(" ", "").lower() == PatientName
    
    if not df[mask_id].empty:
        for marker in markers:
            df.loc[mask_id, marker] = 'Y'
    elif not df[mask_name]:
        for marker in markers:
            df.loc[mask_name, marker] = 'Y'
    else:
        print("病人",AccessionNumber, PatientName, "不存在")


def process_directory(root_path):
    """处理单个目录的核心逻辑"""
    for roots, dirs, files in os.walk(root_path):
        if len(dirs) == 0 and files:  # 叶子目录且包含文件
            try:
                dcm_file = os.path.join(roots, files[0])
                info = pydicom.dcmread(dcm_file, stop_before_pixels=True)
                
                # 获取关键DICOM元数据
                AccessionNumber = info.get('AccessionNumber', '')         # 检查号
                series_desc = str(info.get('SeriesDescription', '')).lower()  # 序列描述
                PatientName = info.get("PatientsName").replace(' ', '').lower()
                markers = []
                root_lower = roots.lower()

                # SAX检测逻辑
                if 'sa' in root_lower or 'short-axis' in series_desc:
                    markers.append('SAX')

                # LGE检测逻辑
                if ('psir' in root_lower or 'lge' in series_desc) and \
                'sa' not in root_lower and \
                '4c' not in root_lower:
                    markers.append('LGE')

                # 4CH检测逻辑
                if ('4ch' in series_desc or '4 ch' in series_desc) and \
                'sa' not in root_lower and \
                'psir' not in root_lower:
                    markers.append('4CH')

                # 更新数据表
                if markers and AccessionNumber:
                    update_dataframe(AccessionNumber, markers)

            except Exception as e:
                print(f"处理目录失败: {roots}\n错误信息: {str(e)}")

AZ_path = "I:/CMR-China-new/AZ_data/raw_dcmdata/"
RJ_path = "I:/CMR-China-new/RJ_data/raw_dcmdata/"
table_path = "I:/china_外部_统计.xlsx"

# Load DataFrame
if os.path.exists(table_path):
    df = pd.read_excel(table_path)

# 处理所有数据目录
for path in [AZ_path, RJ_path]:
    if os.path.exists(path):
        process_directory(path)
    else:
        print(f"警告: 路径不存在 {path}")

# 保存更新后的表格
df.to_excel(table_path, index=False)
print(f"表格已成功更新: {table_path}")