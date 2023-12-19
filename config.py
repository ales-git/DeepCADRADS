import torch


class Config_00_Preprocessing:
    orig_path = '../Data/Original/Img/'
    out_path = '../Data/Preprocessed/Img/'

    thresh = 15  # threshold for binarization
    maxval = 255  # max value for binarization
    ksize = 2  # kernel size for morphological transformation
    operation = "close"  # morphological operation
    reverse = True  # reverse the image
    top_x = 1  # the top X contours to keep
    clip = 0.4  # clip the image to this value
    tile = 8  # tile the image to this value


class Config_01_SplitDataset:
    csv_path = '../Data/Original/csv/CADRADS.csv'
    train_ratio = 0.80
    test_ratio = 0.20
    class_name = 'multi_Class_3'  # or 'bin_Class' (name of the class in the csv file)


class Config_02_SyntheticData:
    dir = '../Data/Preprocessed/Img'
    dir_fulldata = '../Data/FullData/'
    main_out_path = 'MultiClass/'  # or 'BinaryClass'
    main_csv_path = '../_01_SplitDataset/MultiClass/'  # or 'BinaryClass'
    n_fold = 10


class Config_03_Train:
    project_name = 'MultiClass'  # or 'BinaryClass'
    model_name = 'maxvit_tiny_rw_224'
    checkpoints_path = '../checkpoints/'
    seed = 23

    # paths for main_cv
    main_data_path = '../Data/FullData/' + project_name + '/train/'
    main_txt_path = '../_01_SplitDataset/' + project_name + '/train/'
    # paths for main
    data_path = '../Data/FullData/' + project_name + '/test/'
    txt_path = '../_01_SplitDataset/' + project_name + '/test/txt/'

    # params for grid search (main_cv.py)
    dropout_list = [0.1, 0.3, 0.5]
    l2_list = [1e-1, 1e-2]
    lr_list = [1e-3, 1e-4, 1e-5]
    milestones_list = [[20], [30]]
    lsmooth_list = [0.1, 0.2]
    min_acc_fold = 0.8  # minimum accuracy for each fold to continue with the current param configuration

    # fixed params for final training (main.py)
    dropout = 0.5
    l2 = 1e-1
    lr = 1e-4
    milestones = [30]
    lsmooth = 0.1

    # learning params
    epochs = 50
    batch_size = 8
    pos_weight = torch.tensor(1)
    dim = 224  # dim img input

    # data augmentation
    rot_angles = [-90, 0, 90]
    hflip_prob = 0.5
    vflip_prob = 0.5


class Config_04_Test:
    exp_name = 'MultiClass'  # or 'BinaryClass'

    data_path = '../Data/FullData/' + exp_name + '/test/'
    txt_path = '../_01_SplitDataset/' + exp_name + '/test/txt/test.txt'
    csv_path = '../_01_SplitDataset/' + exp_name + '/test/csv/X_test.csv'

    # weights for computing average metrics keeping in mind the class imbalance
    class_weights = [0.2, 0.5, 0.3]
    checkpoints_path = '../checkpoints/'
    checkpoints_name = 'BEST_MultiClass'  # or 'BEST_BinaryClass'

    dim = 224  # dim img input
    batch_size = 1  # dim batch size for test

    n_img_background = 100  # number of images to use as background for SHAP
    index_img_shap = 10  # index of the image to explain with SHAP
    img_dim_plot = (800, 90)  # dimension of the image to plot
