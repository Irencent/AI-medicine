import os
import numpy as np
import shutil
import pydicom
from tqdm import tqdm
import pandas as pd
import SimpleITK as sitk
import dicom2nifti
import matplotlib.pyplot as plt
from PIL import Image
import shutil


def position(dicom_dir):
    file = pydicom.filereader.dcmread(dicom_dir)
    cosines = file[0x20, 0x37].value
    ipp = file[0x20, 0x32].value
    a1 = np.zeros((3,))
    a2 = np.zeros((3,))
    for i in range(3):
        a1[i] = cosines[i]
        a2[i] = cosines[i+3]
    index = np.abs(np.cross(a1, a2))
    return ipp[np.argmax(index)]


def axis(dicom_dir):
    file = pydicom.filereader.dcmread(dicom_dir)
    cosines = file[0x20, 0x37].value
    a1 = np.zeros((3,))
    a2 = np.zeros((3,))
    for i in range(3):
        a1[i] = cosines[i]
        a2[i] = cosines[i+3]
    index = np.abs(np.cross(a1, a2))
    return np.argmax(index)


def search_file(path):
    for root, dirs, files in os.walk(path):
        if len(dirs) != 0 and '_' in dirs[0]:
            data1.append(root)


def check_img(path_list):
    for path in path_list:
        files = os.listdir(path)
        for i in range(len(files)):
            files[i] = str(files[i],encoding='utf8')
        files = list(filter(lambda x: '.dcm' in x, files))
        img = sitk.ReadImage(path+'/'+files[0])
        img = sitk.GetArrayFromImage(img)
        plt.imshow(img[0], cmap='gray')
        plt.show()
    #     plt.savefig('/media/data/yanran/sax_add/'+str(cine.index(path))+'.jpg',dpi=30)
    #     plt.close('all')
        plt.clf()


def convert_nifti_cine(cine):
    if not os.path.exists('/media/data/yanran/nifti data/'):
        os.makedirs('/media/data/yanran/nifti data/')

    temp_path = cine[0]
    files = os.listdir(temp_path)
#     for i in range(len(files)):
#         files[i] = str(files[i], encoding="utf8")
    files = list(filter(lambda x: '.dcm' in x, files))
    temp = sitk.ReadImage(temp_path+'/'+files[0])
    num = np.argsort(temp.GetSize())
    pa = sitk.PermuteAxesImageFilter()
    pa.SetOrder(np.array([num[2], num[1], num[0]], dtype='int').tolist())
    temp = pa.Execute(temp)
    out_spacing = [1.7708333730698, 1.7708333730698, temp.GetSpacing()[2]]
    for folder in tqdm(cine):
        files = os.listdir(folder)
#         for i in range(len(files)):
#             files[i] = str(files[i], encoding="utf8")
        files = list(filter(lambda x: '.dcm' in x, files))
        files = sorted(files, key=embedded_numbers)
        for file in files[0:25]:
            dcm = sitk.ReadImage(folder+'/'+file)
            num = np.argsort(dcm.GetSize())
            pa = sitk.PermuteAxesImageFilter()
            pa.SetOrder(np.array([num[2], num[1], num[0]], dtype='int').tolist())
            dcm = pa.Execute(dcm)
            original_size = dcm.GetSize()
            original_spacing = dcm.GetSpacing()

            out_size = [
                int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
                int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
                int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(out_spacing)
            resampler.SetSize(out_size)
            resampler.SetOutputDirection(dcm.GetDirection())
            resampler.SetOutputOrigin(dcm.GetOrigin())
            resampler.SetInterpolator(sitk.sitkBSpline)
            resampler.SetTransform(sitk.Transform())
            resampled = resampler.Execute(dcm)
            if ('data2/正常' in folder and len(folder.split('/')) == 5) or 'no' in folder:
                new_path = '/media/data/yanran/nifti data/'+'/'.join(folder.split('/')[-3:])
            else:
                new_path = '/media/data/yanran/nifti data/'+'/'.join(folder.split('/')[-3:])
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            sitk.WriteImage(resampled, new_path+'/'+file[:-4]+'.nii')


def convert_nifti_4ch_lge(lge):
    if not os.path.exists('/media/data/yanran/nifti data'):
        os.makedirs('/media/data/yanran/nifti data')

    temp_path = lge[0]
    files = os.listdir(temp_path)
    for i in range(len(files)):
        files[i] = str(files[i], encoding="utf8")
    files = list(filter(lambda x: '.dcm' in x, files))
    temp = sitk.ReadImage(temp_path+'/'+files[0])
    num = np.argsort(temp.GetSize())
    pa = sitk.PermuteAxesImageFilter()
    pa.SetOrder(np.array([num[2], num[1], num[0]], dtype='int').tolist())
    temp = pa.Execute(temp)
    out_spacing = [1.328125, 1.328125, temp.GetSpacing()[2]]
    for folder in tqdm(lge):
        files = os.listdir(folder)
        for i in range(len(files)):
            files[i] = str(files[i], encoding="utf8")
        files = list(filter(lambda x: '.dcm' in x, files))
        files = sorted(files, key=embedded_numbers)
        for file in [files[0]]:
            dcm = sitk.ReadImage(folder+'/'+file)
            num = np.argsort(dcm.GetSize())
            pa = sitk.PermuteAxesImageFilter()
            pa.SetOrder(np.array([num[2], num[1], num[0]], dtype='int').tolist())
            dcm = pa.Execute(dcm)
            original_size = dcm.GetSize()
            original_spacing = dcm.GetSpacing()

            out_size = [
                int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
                int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
                int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(out_spacing)
            resampler.SetSize(out_size)
            resampler.SetOutputDirection(dcm.GetDirection())
            resampler.SetOutputOrigin(dcm.GetOrigin())
            resampler.SetInterpolator(sitk.sitkBSpline)
            resampler.SetTransform(sitk.Transform())
            resampled = resampler.Execute(dcm)
            if 'data2/正常' in folder or 'no' in folder:
                new_path = '/media/data/yanran/nifti data/'+'/'.join(folder.split('/')[-3:])
            else:
                new_path = '/media/data/yanran/nifti data/'+'/'.join(folder.split('/')[-4:])
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            sitk.WriteImage(resampled, new_path+'/'+file[:-4]+'.nii')


def convert_nifti_3d_lge(lge):
    wrong = []
    if not os.path.exists('/media/data/yanran/nifti data'):
        os.makedirs('/media/data/yanran/nifti data')

    out_spacing = [1.7708333730698, 1.7708333730698, 8]
    for folder in tqdm(lge):
        files = os.listdir(folder)
        for i in range(len(files)):
            files[i] = str(files[i], encoding="utf8")
        files = list(filter(lambda x: '.dcm' in x, files))
        if 'data2/正常' in folder or 'no' in folder:
            new_path = '/media/data/yanran/nifti data/'+'/'.join(folder.split('/')[-3:])
        else:
            new_path = '/media/data/yanran/nifti data/'+'/'.join(folder.split('/')[-4:])
        if len(files) >= 4:
            try:
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                dicom2nifti.dicom_series_to_nifti(folder+'/', new_path + '/1.nii', reorient_nifti=True)
                dcm = sitk.ReadImage(new_path + '/1.nii')
                num = np.argsort(dcm.GetSize())
                pa = sitk.PermuteAxesImageFilter()
                pa.SetOrder(np.array([num[2], num[1], num[0]], dtype='int').tolist())
                dcm = pa.Execute(dcm)
                original_size = dcm.GetSize()
                original_spacing = dcm.GetSpacing()

                out_size = [
                    int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
                    int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
                    int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

                resampler = sitk.ResampleImageFilter()
                resampler.SetOutputSpacing(out_spacing)
                resampler.SetSize(out_size)
                resampler.SetOutputDirection(dcm.GetDirection())
                resampler.SetOutputOrigin(dcm.GetOrigin())
                resampler.SetInterpolator(sitk.sitkBSpline)
                resampler.SetTransform(sitk.Transform())
                resampled = resampler.Execute(dcm)
                sitk.WriteImage(resampled, new_path + '/1.nii')
            except Exception as e:
                wrong.append(folder)
            continue
        else:
            wrong.append(folder)
    return wrong


def ROI(data, mask):
    x, y = np.nonzero(mask)
    crop_data = np.zeros((data.shape[0], x.max()-x.min()+1, y.max()-y.min()+1))
    for i in range(data.shape[0]):
        crop_data[i, :, :] = data[i, x.min():x.max()+1, y.min():y.max()+1]
    return crop_data


def find_origin_path(a):
    pick = a.copy()
    for i in range(len(pick)):
        if os.path.exists(pick[i].replace('media/data/yanran/nifti data','data2')):
            pick[i] = pick[i].replace('media/data/yanran/nifti data','data2')
        elif os.path.exists(pick[i].replace('media/data/yanran/nifti data','data3')):
            pick[i] = pick[i].replace('media/data/yanran/nifti data','data3')
        elif os.path.exists(pick[i].replace('media/data/yanran/nifti data','media/data/data3')):
            pick[i] = pick[i].replace('media/data/yanran/nifti data','media/data/data3')
        elif os.path.exists(pick[i].replace('media/data/yanran/nifti data','media/data/data4')):
            pick[i] = pick[i].replace('media/data/yanran/nifti data','media/data/data4')
        elif os.path.exists(pick[i].replace('media/data/yanran/nifti data','media/data/待看正常1105')):
            pick[i] = pick[i].replace('media/data/yanran/nifti data','media/data/待看正常1105')
        else:
            print(pick[i])
    
    return pick


def plane(dicom_dir):
    file = pydicom.filereader.dcmread(dicom_dir)
    cosines = file[0x20, 0x37].value
    a1 = np.zeros((3,))
    a2 = np.zeros((3,))
    for i in range(3):
        a1[i] = cosines[i]
        a2[i] = cosines[i+3]
    index = np.cross(a1, a2)
    b = np.sqrt(np.sum(index**2))
    index = index/b
    return index


def find_median_slice(data1):
    sax_cine = []
    for dr in tqdm(data1):
        try:
            folders = os.listdir(dr)
    #         for i in range(len(folders)):
    #             folders[i] = str(folders[i], encoding='utf8')
            folders = np.array(list(filter(lambda x: 'MAG' not in x and 'PSIR' not in x and 'ch' not in x and 'CH' not in x and 'RVOT' not in x and 'grid' not in x, folders)))
            if len(list(filter(lambda x: 'sax' in x or 'SAX' in x, folders))) is not 0:
                folders = np.array(list(filter(lambda x: 'sax' in x or 'SAX' in x, folders)))
            dele = []
            for i in range(len(folders)):
                files = os.listdir(dr+'/'+folders[i])
                if len(files) < 25:
                    dele.append(i)
            folders = np.delete(folders, dele)
            count = []
            files = os.listdir(dr+'/'+folders[0])
            file = dr+'/'+folders[0]+'/'+files[0]
            a = plane(file)
            for i in range(len(folders)):
                files = os.listdir(dr+'/'+folders[i])
                file = dr+'/'+folders[i]+'/'+files[0]
                b = plane(file)
                l = np.abs(np.sum(a*b))
                l = float('%.2f'%l)
                count.append(l)

            mask = np.unique(count)
            tmp = []
            for v in mask:
                tmp.append(np.sum(count==v))
            max_v = mask[np.argsort(tmp)[-1]]
            folders = folders[np.where(count==max_v)[0]]

            pre = []
            dele = []
            for i in range(len(folders)):
                files = os.listdir(dr+'/'+folders[i])
                file = dr+'/'+folders[i]+'/'+files[0]
                if position(file) in pre:
                    dele.append(i)
                    continue
                else:
                    pre.append(position(file))
            folders = np.delete(folders, dele)

            cine_pos = []
            for i in folders:
                cine_files = os.listdir(dr+'/'+i)
                cine_files = list(filter(lambda x: '.dcm' in x, cine_files))
                cine_pos.append(position(dr+'/'+i+'/'+cine_files[0]))
            sort = np.sort(cine_pos)
            if len(sort) > 2:
                center = int(np.where(cine_pos==sort[int(np.ceil(len(sort)/2)-2)])[0])
            else:
                center = int(np.where(cine_pos==sort[int(np.ceil(len(sort)/2)-1)])[0])
            sax_cine.append(dr+'/'+folders[center])
            
        except Exception as e:
            print(dr)
            continue


def find_4ch_cine(data):
    ch = []
    wrong = []
    for dr in tqdm(data):
        try:
            folders = os.listdir(dr)
            for i in range(len(folders)):
                folders[i] = str(folders[i], encoding='utf8')
            folders0 = np.array(list(filter(lambda x: 'MAG' not in x and 'PSIR' not in x and 'RVOT' not in x and 'grid' not in x and 'sax'
                                           not in x and 'SAX' not in x and 'DENSE' not in x and ('4ch' in x or '4CH' in x), folders)))
            
            dele = []
            for i in range(len(folders0)):
                files = os.listdir(dr+'/'+folders0[i])
                if len(files) < 25:
                    dele.append(i)
            folders0 = np.delete(folders0, dele)
            
            if len(folders0) is 0:
                
                folders1 = np.array(list(filter(lambda x: '4ch' in x or '4CH' in x, folders)))
                folders2 = np.array(list(filter(lambda x: 'MAG' not in x and 'PSIR' not in x and 'RVOT' not in x and 'grid' not in x and 'sax'
                                               not in x and 'SAX' not in x and 'DENSE' not in x, folders)))
                dele = []
                for i in range(len(folders2)):
                    files = os.listdir(dr+'/'+folders2[i])
                    if len(files) < 25:
                        dele.append(i)
                folders2 = np.delete(folders2, dele)
                
                count = []
                files = os.listdir(dr+'/'+folders1[0])
                for i in range(len(files)):
                    files[i] = str(files[i], encoding='utf8')
                file = dr+'/'+folders1[0]+'/'+files[0]
                a = plane(file)
                for i in range(len(folders2)):
                    files = os.listdir(dr+'/'+folders2[i])
                    file = dr+'/'+folders2[i]+'/'+files[0]
                    b = plane(file)
                    l = np.abs(np.sum(a*b))
                    if l < 0.8:
                        count.append(i)

            #         print(count)
                folders2 = np.delete(folders2, count)
                folders = folders2
            else:
                folders = folders0

            ch.append(dr+'/'+folders[0])
        except Exception as e:
            wrong.append(dr)
            print(dr)
            continue


def find_3d_lge(data):
    lge = []
    wrong = []
    for dr in tqdm(data):
        try:
            folders = os.listdir(dr)
            for i in range(len(folders)):
                folders[i] = str(folders[i], encoding='utf8')
            folders0 = np.array(list(filter(lambda x: 'PSIR' in x and ('overview' in x or 'single-shot' in x or 'singleshot' in x), folders)))
            
            dele = []
            for i in range(len(folders0)):
                files = os.listdir(dr+'/'+folders0[i])
                if len(files) < 4:
                    dele.append(i)
            folders0 = np.delete(folders0, dele)
            
            folders = folders0

            lge.append(dr+'/'+folders[0])
        except Exception as e:
            wrong.append(dr)
            print(dr)
            continue


def binary_class(text):
    new = []
    for line in text:
        if line.split()[-1] is '7':
            if line[-3] is not ' ':
                line = line[:-2]+' 0'+'\n'
            else:
                line = line[:-2]+'0'+'\n'
            new.append(line)
        else:
            if line[-3] is not ' ':
                line = line[:-2]+' 1'+'\n'
            else:
                line = line[:-2]+'1'+'\n'
            new.append(line)
    new = np.array(new, dtype='<U300')
    return new


def del_class(text):
    new = []
    for line in text:
        if line.split()[-1] not in ['9', '10', '11']:
            new.append(line)
    new = np.array(new, dtype='<U300')
    return new


def save_txt(text, path_name):
    file_write_obj = open("/media/data/yanran/cmr_data/"+path_name+'.txt', 'w')
    for var in text:
        file_write_obj.writelines(var)
        if var[-1] is not '\n':
            file_write_obj.writelines('\n')
    file_write_obj.close()


def read_txt(path_name):
    a = '/media/data/yanran/cmr_data/'+path_name+'.txt'
    cache = []
    file_obj = open(a)
    all_lines = file_obj.readlines()
    for line in all_lines:
        cache.append(line)
    file_obj.close()
    a = np.array(cache, dtype='<U200')
    
    return a


def get_mean_std(path_name):
    file = open("/media/data/yanran/cmr_data/"+path_name+".txt", "r", encoding='UTF-8')

    sum_m = 0
    sum_s = 0
    count = 0

    while 1:
        line = file.readline()
        if not line:
            break

        path = line.split(' ')[0].replace('\n', '')
        count += 1
        filelist = os.listdir(path)
        m = 0
        s = 0
        for i in range(len(filelist)):
            filename = filelist[i]
            img = cv2.imread(path + '/' + filename)
            m = m + np.mean(img)
            s = s + np.std(img)
        if len(filelist) == 0:
            continue

        sum_m += m/len(filelist)
        sum_s += s/len(filelist)

    file.close()

    res_m = sum_m/count
    res_s = sum_s/count

    print('mean:', res_m, ' std:', res_s)


def check_img_size(path_name):
    a = read_txt(path_name)
    for i in a:
        path = i.split()[0]
        img = cv2.imread(path+'/img_00001.jpg')
        if img.shape[0] > 2*img.shape[1] or img.shape[1] > 2*img.shape[0]:
            print(path)

