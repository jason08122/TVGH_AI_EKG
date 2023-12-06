import torch
import os
import shutil
import numpy as np
import yaml
from sklearn.metrics import confusion_matrix

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)


def my_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    confusion = confusion_matrix(y_true, y_pred)
    # print(confusion)
    # (tn, fp, fn, tp)
    tn = (confusion[1][1] + confusion[1][2] + confusion[2][1] + confusion[2][2])
    fp = (confusion[1][0] + confusion[2][0])
    fn = (confusion[0][1] + confusion[0][2])
    tp = (confusion[0][0])
    matrix1 = (tn, fp, fn, tp)
    tn = (confusion[0][0] + confusion[0][2] + confusion[2][0] + confusion[2][2])
    fp = (confusion[0][1] + confusion[2][1])
    fn = (confusion[1][0] + confusion[1][2])
    tp = (confusion[1][1])
    matrix2 = (tn, fp, fn, tp)
    tn = (confusion[0][0] + confusion[0][1] + confusion[1][0] + confusion[1][1])
    fp = (confusion[0][2] + confusion[1][2])
    fn = (confusion[2][0] + confusion[2][1])
    tp = (confusion[2][2])
    matrix3 = (tn, fp, fn, tp)

    return matrix1, matrix2, matrix3


def cal_metrics(matrix):
    tn, fp, fn, tp = matrix
    accuracy = (tp + tn) / (tn + fp + fn + tp)
    if tp + fp == 0:
        precision = tp / (tp + fp + 1)
    else:
        precision = tp / (tp + fp)
    if tp + fn == 0:
        recall = tp / (tp + fn + 1)
    else:
        recall = tp / (tp + fn)
    if fp + tn == 0:
        specificity = tn / (fp + tn + 1)
    else:
        specificity = tn / (fp + tn)
    if precision + recall == 0:
        f1 = 2 * precision * recall / (precision + recall + 1)
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return accuracy, precision, recall, specificity, f1

# def my_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
#     y_true = y_true.astype('bool')
#     y_pred = y_pred.astype('bool')
#     tn = (~y_true & ~y_pred).sum()
#     fp = (~y_true & y_pred).sum()
#     fn = (y_true & ~y_pred).sum()
#     tp = (y_true & y_pred).sum()

#     return tn, fp, fn, tp