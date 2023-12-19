import os.path
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.model_selection import train_test_split
from config import Config_01_SplitDataset as config


def create_files_labels(X, y, path_towrite, path_files, txt_name):
    '''
    Create txt files with the list of images and labels
    '''
    sample_list = [path_towrite + str(s) + ' ' + str(s1) for s, s1 in zip(list(X['uniq_ID'].values), y)]

    with open(os.path.join(path_files, txt_name), 'w') as f:
        for item in sample_list:
            if str(item).startswith(path_towrite):
                f.write("%s\n" % item)


def split_data(train_ratio, class_name, folder_name, seed):
    ''' Split the dataset in train and test set. 10-fold CV on training set.'''
    X_train, X_test, y_train, y_test = train_test_split(df, df[class_name],
                                                        stratify=df['multi_Class'],
                                                        # stratify on original CAD-RADS labels
                                                        test_size=1 - train_ratio,
                                                        random_state=seed)
    # stratified kfold
    skf_cv = StratifiedKFold(n_splits=10)
    for split, (idx_train, idx_val) in enumerate(skf_cv.split(X_train, X_train['multi_Class'])):
        print(f'SPLIT {split + 1}')

        # Create the directory set_i
        directory = os.path.join(folder_name, 'train', f'set{split + 1}')
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Create the txt and csv subdirectories
        txt_dir = os.path.join(directory, 'txt')
        csv_dir = os.path.join(directory, 'csv')
        os.makedirs(txt_dir, exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)

        print(f'TRAIN INDEXES: {idx_train}, VAL INDEXES: {idx_val}\n')
        print(f'TRAIN len: {len(idx_train)}, VAL len: {len(idx_val)}\n')
        X_train_cv = X_train.iloc[idx_train]
        X_val_cv = X_train.iloc[idx_val]
        y_train_cv = y_train.iloc[idx_train]
        y_val_cv = y_train.iloc[idx_val]
        nfold = 'set' + str(split + 1)

        # Create files with training/validation/test paths and labels
        path_towrite_txt = os.path.join('/Data/FullData', folder_name, nfold + '/')
        path_files_txt = os.path.join(folder_name, 'train', nfold, 'txt/')
        create_files_labels(X_train_cv, y_train_cv, path_towrite_txt, path_files_txt, 'train.txt')
        create_files_labels(X_val_cv, y_val_cv, path_towrite_txt, path_files_txt, 'val.txt')

        # create csv files
        path_files_csv = os.path.join(folder_name, 'train', nfold, 'csv/')
        X_train_cv.to_csv(path_files_csv + 'X_train.csv')
        X_val_cv.to_csv(path_files_csv + 'X_val.csv')

        # save test data to test directory
        if split == 0:

            directory = os.path.join(folder_name, 'test')
            if not os.path.exists(directory):
                os.makedirs(directory)

            # Create the txt and csv subdirectories
            txt_dir = os.path.join(directory, 'txt')
            csv_dir = os.path.join(directory, 'csv')
            os.makedirs(txt_dir, exist_ok=True)
            os.makedirs(csv_dir, exist_ok=True)

            # complete train set (train + validation)
            X_train_complete = X_train_cv.append(X_val_cv, ignore_index=True)
            y_train_complete = y_train_cv.append(y_val_cv, ignore_index=True)

            X_test.to_csv(os.path.join(csv_dir, 'X_test.csv'))
            X_train_complete.to_csv(os.path.join(csv_dir, 'X_train.csv'))
            path_files_txt_test = os.path.join(folder_name, 'test/txt/')

            path_towrite_txt = os.path.join('/Data/FullData', folder_name, 'test/')
            create_files_labels(X_train_complete, y_train_complete, path_towrite_txt, path_files_txt_test, 'train.txt')
            create_files_labels(X_test, y_test, path_towrite_txt, path_files_txt_test, 'test.txt')

    return X_train, y_train, X_test, y_test


if __name__ == '__main__':

    # binary class or multi class
    class_name = config.class_name

    # random state:
    seed = 32

    if class_name == 'bin_Class':
        folder_name = 'BinaryClass'
        print('Binary Classification')
    elif class_name == 'multi_Class_3':
        folder_name = 'MultiClass'
        print('Multi Classification (3 classes)')

    df = pd.read_csv(config.csv_path, delimiter=';')

    n_pat = len(df)
    if class_name == 'bin_Class':

        df_class0 = df[df[class_name] == 0]
        df_class1 = df[df[class_name] == 1]

        n_pat_class0 = len(df_class0)
        n_pat_class1 = len(df_class1)

        print('\nBinary classification (0-2 vs 3-5):')
        print('pat class 0 = ' + str(round(n_pat_class0 * 100 / (n_pat))) + ' %')
        print('pat class 1 = ' + str(round(n_pat_class1 * 100 / (n_pat))) + ' %')


    elif class_name == 'multi_Class_3':

        df_class0 = df[df['multi_Class'] == 0]
        df_class1 = df[df['multi_Class'] == 1]
        df_class2 = df[df['multi_Class'] == 2]

        n_pat_class0 = len(df_class0)
        n_pat_class1 = len(df_class1)
        n_pat_class2 = len(df_class2)

        print('\nMulti classification (0-2):')
        print('pat class 0 = ' + str(round(n_pat_class0 * 100 / (n_pat))) + ' %')
        print('pat class 1 = ' + str(round(n_pat_class1 * 100 / (n_pat))) + ' %')
        print('pat class 2 = ' + str(round(n_pat_class2 * 100 / (n_pat))) + ' %')

    train_ratio = config.train_ratio
    test_ratio = config.test_ratio

    X_train, y_train, X_test, y_test = split_data(train_ratio, class_name, folder_name, seed)
