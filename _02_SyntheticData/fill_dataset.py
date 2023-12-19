import os
import pandas as pd
import numpy as np
import shutil
from config import Config_02_SyntheticData as config

def missing_views(row,current_views_max_vessel, vessel_name):
    img_list = os.listdir(path_FullData)
    pt_img = [file for file in img_list if
              row['uniq_ID'] + '_' + vessel_name in file]
    num_views = [f.split('_')[-1].split('se00')[-1].split('.')[0] for f in pt_img]
    num_views = list(map(int, num_views))
    all_views = set(current_views_max_vessel)  # set(range(0, max_val))
    all_views = list(map(int, all_views))
    missing_views = list(set(num_views).symmetric_difference(all_views))
    missing_views = list(dict.fromkeys(missing_views))  # remove duplicates

    return missing_views, num_views


def fill_dataset(path_SplitDataset, path_FullData, path_SyntheticData, list_sets):
    for data_set in list_sets:

        path_csv = path_SplitDataset+'csv/X_'+data_set+'.csv'
        folder_name = data_set
        df = pd.read_csv(path_csv)
        df = df.fillna(0)
        df['img_LCX']=df['img_LCX'].astype(int)
        df['img_LAD']=df['img_LAD'].astype(int)
        df['img_RCA']=df['img_RCA'].astype(int)
        counter = 0
        for index,row in df.iterrows():
             n_LCX = row['img_LCX']
             n_LAD = row['img_LAD']
             n_RCA = row['img_RCA']
             vessels = np.array([n_LCX,n_LAD,n_RCA])
             # max num img
             max_pos = vessels.argmax()
             max_val = vessels[max_pos]
             if max_pos == 0:
                 img_list = os.listdir(path_FullData)
                 pt_img = [file for file in img_list if
                           row['uniq_ID']+'_LCX' in file]
                 current_views_max_vessel = [f.split('_')[-1].split('se00')[-1].split('.')[0] for f in pt_img]
             elif max_pos == 1:
                 img_list = os.listdir(path_FullData)
                 pt_img = [file for file in img_list if
                           row['uniq_ID']+ '_LAD' in file]
                 current_views_max_vessel = [f.split('_')[-1].split('se00')[-1].split('.')[0] for f in pt_img]
             elif max_pos ==2:
                 img_list = os.listdir(path_FullData)
                 pt_img = [file for file in img_list if
                           row['uniq_ID']+ '_RCA' in file]
                 current_views_max_vessel = [f.split('_')[-1].split('se00')[-1].split('.')[0] for f in pt_img]

             if n_LCX < max_val:
                 views_toadd, current_views = missing_views(row,current_views_max_vessel, 'LCX')
                 for i in views_toadd:
                     shutil.copy(path_SyntheticData+'avg_LCX_se00'+str(i)+'.png',
                                 path_FullData + row['uniq_ID']+ '_LCX' + '_se00' + str(i) + '.png')
                     n_LCX = max_val
                     counter+=1
             if n_LAD < max_val:
                 views_toadd, current_views = missing_views(row,current_views_max_vessel, 'LAD')
                 for i in views_toadd:
                     shutil.copy(path_SyntheticData+'avg_LAD_se00'+str(i)+'.png',
                                 path_FullData + row['uniq_ID'] + '_LAD' + '_se00' + str(i) + '.png')
                     n_LAD = max_val
                     counter += 1
             if n_RCA < max_val:
                 views_toadd, current_views = missing_views(row,current_views_max_vessel, 'RCA')
                 for i in views_toadd:
                     shutil.copy(path_SyntheticData+'avg_RCA_se00'+str(i)+'.png',
                                 path_FullData + row['uniq_ID']+ '_RCA' + '_se00' + str(i) + '.png')
                     n_RCA = max_val
                     counter += 1

        #print(counter)

if __name__ == '__main__':

    ''' Fill train folder. Avg images generated considering single training sets for each folder. '''
    for num_set in range(1, config.n_fold + 1):
        directory = os.path.join(config.dir_fulldata, config.main_out_path, 'train', f'set{num_set}')
        if not os.path.exists(directory):
            os.makedirs(directory)
        shutil.copytree(config.dir, directory, dirs_exist_ok=True)

    ''' Fill test folder. Avg images generated considering the whole training set (train+val). '''
    directory = os.path.join(config.dir_fulldata, config.main_out_path, 'test')
    if not os.path.exists(directory):
        os.makedirs(directory)
    shutil.copytree(config.dir, directory, dirs_exist_ok=True)

    for num_set in range(1, config.n_fold + 1):
        nset = 'set' + str(num_set)
        print(f'--- fold: {num_set} ---')
        path_FullData = os.path.join(config.dir_fulldata,config.main_out_path, 'train', nset+'/')
        path_SplitDataset = os.path.join(config.main_csv_path,'train',nset+'/')
        path_SyntheticData = os.path.join(config.main_out_path,'train',nset+'/')
        fill_dataset(path_SplitDataset, path_FullData, path_SyntheticData,['train','val'])

        path_FullData = os.path.join(config.dir_fulldata, config.main_out_path, 'test/')
        path_SplitDataset = os.path.join(config.main_csv_path,'test/')
        path_SyntheticData = os.path.join(config.main_out_path,'test/')
        fill_dataset(path_SplitDataset, path_FullData, path_SyntheticData, ['train','test']) #train = train+val