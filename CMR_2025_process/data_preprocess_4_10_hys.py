import os
import pydicom
import shutil
from pydicom.errors import InvalidDicomError

def organize_dicom_dataset(root_path, output_dir):
    """
    按厂商/医院/患者/序列描述分类整理DICOM文件
    
    Args:
        root_path (str): 原始DICOM数据根目录
        output_dir (str): 输出目录
        
    Returns:
        list: 处理失败的文件路径列表
    """
    error_files = []
    
    for root, _, files in os.walk(root_path):
        for file in files:
            file_path = os.path.join(root, file)
            
            try:
                # 读取DICOM元数据
                ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                
                # 提取元数据（处理可能缺失的字段）
                manufacturer = getattr(ds, 'Manufacturer', 'UnknownManufacturer').strip().replace(' ', '_')
                hospital = getattr(ds, 'InstitutionName', 'UnknownHospital').strip().replace(' ', '_')
                
                patient_id = getattr(ds, 'PatientID', 'UnknownID').strip()
                patient_name = getattr(ds, 'PatientName', 'UnknownName').strip().replace('^', '_')
                study_date = getattr(ds, 'StudyDate', 'UnknownDate').strip()
                
                series_desc = getattr(ds, 'SeriesDescription', 'UnnamedSeries').strip()
                instance_uid = getattr(ds, 'SOPInstanceUID', 'UnknownUID').strip()
                
                # 清理非法字符
                safe_replace = lambda s: s.replace('*', '_').replace('?', '_').replace('>', '_').replace('<', '_')
                
                # 构建目标路径
                patient_dir = f"{safe_replace(patient_id)}_{safe_replace(patient_name)}_{study_date}"
                series_dir = safe_replace(series_desc)[:50]  # 限制目录名长度
                
                output_path = os.path.join(
                    output_dir,
                    safe_replace(manufacturer),
                    safe_replace(hospital),
                    safe_replace(patient_dir),
                    safe_replace(series_dir)
                )
                
                # 创建目录并复制文件
                os.makedirs(output_path, exist_ok=True)
                shutil.copy2(
                    file_path,
                    os.path.join(output_path, f"{instance_uid}.dcm")
                )
                
            except InvalidDicomError:
                error_files.append(f"Invalid DICOM: {file_path}")
            except Exception as e:
                error_files.append(f"{str(e)}: {file_path}")
    
    return error_files


if __name__ == "__main__":
    errors = organize_dicom_dataset(
        root_path="I:/图像",
        output_dir="I:/CMR_2025_4_10_hys"
    )
    
    if errors:
        print("\nError Files:")
        for error in errors:
            print(error)
    else:
        print("All files processed successfully!")