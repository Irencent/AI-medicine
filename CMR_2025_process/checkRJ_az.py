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