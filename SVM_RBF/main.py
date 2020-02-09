import numpy as np
import time
import collections
import torch
from sklearn import metrics, preprocessing
import datetime
from sklearn.decomposition import PCA, IncrementalPCA

import sys
sys.path.append('../global_module/')
import network
import train
from generate_pic import aa_and_each_accuracy, sampling,load_dataset, generate_png, generate_iter, classification_map, list_to_colormap
from Utils import fdssc_model, record, extract_samll_cubic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# for Monte Carlo runs
seeds = [1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341]
ensemble = 1

day = datetime.datetime.now()
day_str = day.strftime('%m_%d_%H_%M')

print('-----Importing Dataset-----')

global Dataset  # UP,IN,KSC
dataset = input('Please input the name of Dataset(IN, UP, BS, SV, PC or KSC):')
Dataset = dataset.upper()
data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE,VALIDATION_SPLIT = load_dataset(Dataset)

print(data_hsi.shape)
image_x, image_y, BAND = data_hsi.shape
data = data_hsi.reshape(np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))
gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]),)
CLASSES_NUM = max(gt)
print('The class numbers of the HSI data is:', CLASSES_NUM)

print('-----Importing Setting Parameters-----')
ITER = 10
PATCH_LENGTH = 0
# number of training samples per class
# lr, num_epochs, batch_size = 0.0010, 200, 32

img_rows = 2*PATCH_LENGTH+1
img_cols = 2*PATCH_LENGTH+1
img_channels = data_hsi.shape[2]
INPUT_DIMENSION = data_hsi.shape[2]
ALL_SIZE = data_hsi.shape[0] * data_hsi.shape[1]
VAL_SIZE = int(TRAIN_SIZE)
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE


KAPPA = []
OA = []
AA = []
TRAINING_TIME = []
TESTING_TIME = []
ELEMENT_ACC = np.zeros((ITER, CLASSES_NUM))

data = preprocessing.scale(data)
data_ = data.reshape(data_hsi.shape[0], data_hsi.shape[1], data_hsi.shape[2])
whole_data = data_
padded_data = np.lib.pad(whole_data, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)),
                         'constant', constant_values=0)

for index_iter in range(ITER):
    time_1 = int(time.time())
    np.random.seed(seeds[index_iter])
    train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)
    _, total_indices = sampling(1, gt)

    TRAIN_SIZE = len(train_indices)
    print('Train size: ', TRAIN_SIZE)
    TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
    print('Test size: ', TEST_SIZE)
    VAL_SIZE = int(TRAIN_SIZE)
    print('Validation size: ', VAL_SIZE)

    print('-----Selecting Small Pieces from the Original Cube Data-----')
    gt_all = gt[total_indices] - 1
    y_train = gt[train_indices] - 1
    y_test = gt[test_indices] - 1

    all_data = extract_samll_cubic.select_small_cubic(TOTAL_SIZE, total_indices, whole_data,
                                                      PATCH_LENGTH, padded_data, INPUT_DIMENSION)

    train_data = extract_samll_cubic.select_small_cubic(TRAIN_SIZE, train_indices, whole_data,
                                                        PATCH_LENGTH, padded_data, INPUT_DIMENSION)
    test_data = extract_samll_cubic.select_small_cubic(TEST_SIZE, test_indices, whole_data,
                                                       PATCH_LENGTH, padded_data, INPUT_DIMENSION)
    # x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION)
    # x_test = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION)
    # x_all = all_data.reshape(all_data.shape[0], all_data.shape[1], all_data.shape[2], INPUT_DIMENSION)
    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1]*train_data.shape[2]*INPUT_DIMENSION)
    x_test = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2]*INPUT_DIMENSION)
    x_all = all_data.reshape(all_data.shape[0], all_data.shape[1], all_data.shape[2]*INPUT_DIMENSION)

    tic1 = time.clock()

    net = network.svm_rbf(x_train, y_train)
    svm_rbf = net.train()

    toc1 = time.clock()

    pred_test_fdssc = []
    tic2 = time.clock()
    with torch.no_grad():
        for i in range(x_test.shape[0]):
            pred_test_fdssc.extend(svm_rbf.predict(x_test[i]))
    toc2 = time.clock()
    collections.Counter(pred_test_fdssc)
    # print('len', len(pred_test_fdssc))
    gt_test = gt[test_indices] - 1


    overall_acc_fdssc = metrics.accuracy_score(pred_test_fdssc, gt_test)
    confusion_matrix_fdssc = metrics.confusion_matrix(pred_test_fdssc, gt_test)
    each_acc_fdssc, average_acc_fdssc = aa_and_each_accuracy(confusion_matrix_fdssc)
    kappa = metrics.cohen_kappa_score(pred_test_fdssc, gt_test)

    KAPPA.append(kappa)
    OA.append(overall_acc_fdssc)
    AA.append(average_acc_fdssc)
    TRAINING_TIME.append(toc1 - tic1)
    TESTING_TIME.append(toc2 - tic2)
    ELEMENT_ACC[index_iter, :] = each_acc_fdssc

print("--------" + net.name + " Training Finished-----------")
record.record_output(OA, AA, KAPPA, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME,
                     'records/' + net.name + day_str + '_' + Dataset + 'split：' + str(VALIDATION_SPLIT) + '.txt')


pred_test = []
for i in range(x_all.shape[0]):
    pred_test.extend(np.array(svm_rbf.predict(x_all[i])))

gt = gt_hsi.flatten()
x_label = np.zeros(gt.shape)
for i in range(len(gt)):
    if gt[i] == 0:
        gt[i] = 17
        # x[i] = 16
        x_label[i] = 16
    # else:
    #     x_label[i] = pred_test[label_list]
    #     label_list += 1
gt = gt[:] - 1
x_label[total_indices] = pred_test
x = np.ravel(x_label)

# print('-------Save the result in mat format--------')
# x_re = np.reshape(x, (gt_hsi.shape[0], gt_hsi.shape[1]))
# sio.savemat('mat/' + Dataset + '_' + '.mat', {Dataset: x_re})

y_list = list_to_colormap(x)
y_gt = list_to_colormap(gt)

y_re = np.reshape(y_list, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
gt_re = np.reshape(y_gt, (gt_hsi.shape[0], gt_hsi.shape[1], 3))

path = '../' + net.name
classification_map(y_re, gt_hsi, 300,
                   path + '/classification_maps/' + Dataset + '_' + net.name +  '.png')
classification_map(gt_re, gt_hsi, 300,
                   path + '/classification_maps/' + Dataset + '_gt.png')
print('------Get classification maps successful-------')