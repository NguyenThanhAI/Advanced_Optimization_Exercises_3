import os


import argparse
import time
import copy
import pickle

from typing import List, Tuple, Optional

from tqdm import tqdm

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from utils import AverageMeter, exponential_moving_average, predict, compute_cost, derivative_cost_wrt_params, hessian_wrt_params, \
    backtracking_line_search, check_wolfe_II, check_goldstein, \
    adam_step, adamax_step, adabelief_step, adagrad_step, \
    rmsprop_step, momentum_step, adadelta_step



np.random.seed(1000)

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_path", type=str, default=r"kc_house_data.csv")

    args = parser.parse_args()

    return args


def create_data(csv_path: str, use_bias: bool=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)

    x = df[["sqft_living", "sqft_above"]].to_numpy(dtype=np.float32)
    y = df["price"].values.astype(np.float32)

    if use_bias:
        ones = np.ones(shape=[x.shape[0], 1], dtype=np.float32)
        x = np.append(x, ones, axis=1)

    


def init_weights(x: np.ndarray, use_bias: bool=True, initializer: str="xavier") -> np.ndarray:

    if use_bias:
        if initializer == "xavier":
            weights = np.random.normal(loc=0, scale=np.sqrt(2/(x.shape[1])), size=x.shape[1]-1)
        else:
            weights = np.random.rand(x.shape[1]-1)

        weights = np.append(weights, 0.)

    else:
        if initializer == "xavier":
            weights = np.random.normal(loc=0, scale=np.sqrt(2/(x.shape[1])), size=x.shape[1])
        else:
            weights = np.random.rand(x.shape[1])

    return weights

'''min_train_cost = np.inf
min_train_cost_weight = None

min_val_cost = np.inf
min_val_cost_weight = None

train_cost_list = []
train_acc_list = []

val_cost_list = []
val_acc_list = []

time_epoch_list = []

timestamp_epoch_list = []

wolfe_II_list = []
goldstein_list = []

delta_weights_norm_list = []
delta_train_cost = []
delta_val_cost = []
gradient_norm_list = []

inner_count_list = []'''


if __name__ == "__main__":

    args = get_args()

    csv_path = args.csv_path

    normalize = "minmax"
    use_bias = True
    initializer = "xavier"
    num_epochs = 20000
    c_1 = 1e-2
    c_2 = 0.9
    c = 0.25
    threshold = 0.6
    rho = 0.5
    batch_size = 64
    stop_condition = False
    save_dir = "."

    result_min_train_cost = np.inf
    result_min_train_cost_weight = None
    result_min_val_cost = np.inf
    result_min_val_cost_weight = None
    result_train_cost_list = []
    result_train_acc_list = []
    result_val_cost_list = []
    result_val_acc_list = []
    result_time_epoch_list = []
    result_timestamp_epoch_list = []
    result_wolfe_II_list = []
    result_goldstein_list = []
    result_delta_weights_norm_list = []
    result_delta_train_cost = []
    result_delta_val_cost = []
    result_gradient_norm_list = []
    result_inner_count_list = []

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    step_length_list = [1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 5, 10]

    