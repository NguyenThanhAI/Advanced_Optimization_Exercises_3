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

from utils import AverageMeter, amsgrad_step, exponential_moving_average, predict, compute_cost, derivative_cost_wrt_params, hessian_wrt_params, \
    backtracking_line_search, check_wolfe_II, check_goldstein, \
    adam_step, adamax_step, adabelief_step, adagrad_step, \
    rmsprop_step, momentum_step, adadelta_step, nadam_step, inverse_decay, \
    newton_step, accelerated_gradient_step, radam_step, bfgs_post_step, dfp_post_step, \
    avagrad_step, calculate_R, compute_mae, warmup_lr, cyclic_lr



np.random.seed(1000)

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_path", type=str, default=r"kc_house_data_cleaned.csv")

    args = parser.parse_args()

    return args


'''def create_data(csv_path: str, normalize: str="minmax", use_bias: bool=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)

    x = df[["sqft_living", "sqft_lot", "sqft_above", "sqft_basement", "sqft_living", "sqft_lot15"]].to_numpy(dtype=np.float32)
    y = df["price"].values.astype(np.float32)

    #print("x: {}, max: {}, min: {}".format(x.shape, np.max(x, axis=0, keepdims=True), np.min(x, axis=0, keepdims=True)))

    if normalize == "minmax":

        x = (x - np.min(x, axis=0, keepdims=True))/(np.max(x, axis=0, keepdims=True) - np.min(x, axis=0, keepdims=True))
        #y = (y - np.min(y))/ (np.max(y) - np.min(y))

    elif normalize == "standardize":
        
        x = (x-np.mean(x, axis=0, keepdims=True))/np.std(x, axis=0, keepdims=True)
        #y = (y - np.mean(y)) / np.std(y)

    else:
        raise ValueError("No normalizing initializer name {}".format(normalize))

    y = y / (1e6)
    if use_bias:
        ones = np.ones(shape=[x.shape[0], 1], dtype=np.float32)
        x = np.append(x, ones, axis=1)

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    train_indices = indices[:int(0.6 * len(indices))]
    val_indices = indices[int(0.6 * len(indices)):int(0.8 * len(indices))]
    test_indices = indices[int(0.8 * len(indices)):]
    x_train, y_train = x[train_indices], y[train_indices]
    x_val, y_val = x[val_indices], y[val_indices]
    x_test, y_test = x[test_indices], y[test_indices]

    return x_train, y_train, x_val, y_val, x_test, y_test'''


'''def create_data(csv_path: str, normalize: str="minmax", use_bias: bool=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    n = df.shape[0]

    yrs_old = np.zeros(n)
    yrs_reno = np.zeros(n)
    for i in range(n):
        year_sell = int(df['date'][i][-4:])
        yrs_old[i] = year_sell - df['yr_built'][i]
        if df['yr_renovated'][i] == 0:
            yrs_reno[i] = 0
        else:
            yrs_reno[i] = year_sell - df['yr_renovated'][i]

    y = np.array(df['price'])

    yrs_old = yrs_old.reshape((n, 1))
    yrs_reno = yrs_reno.reshape((n, 1))
    x = df.to_numpy(df.drop(columns=['id', 'date', 'price', 'yr_built', \
                                       'yr_renovated'], inplace=True)) # 'zipcode', 'lat', 'long
    x = np.concatenate((x, yrs_old, yrs_reno), axis=1)
    if normalize == "minmax":

        x = (x - np.min(x, axis=0, keepdims=True))/(np.max(x, axis=0, keepdims=True) - np.min(x, axis=0, keepdims=True))
        #y = (y - np.min(y))/ (np.max(y) - np.min(y))

    elif normalize == "standardize":
        
        x = (x-np.mean(x, axis=0, keepdims=True))/np.std(x, axis=0, keepdims=True)
        #y = (y - np.mean(y)) / np.std(y)

    else:
        raise ValueError("No normalizing initializer name {}".format(normalize))

    y = y / (1e6)
    if use_bias:
        ones = np.ones(shape=[x.shape[0], 1], dtype=np.float32)
        x = np.append(x, ones, axis=1)

    x_train, y_train, x_val, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    return x_train, y_train, x_val, y_val'''


def create_data(csv_path: str, normalize: str="minmax", use_bias: bool=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    y = np.array(df['price'])
    x = df.to_numpy(df.drop("price", axis=1,  inplace=True))

    y = y / (1e6)
    if use_bias:
        ones = np.ones(shape=[x.shape[0], 1], dtype=np.float32)
        x = np.append(x, ones, axis=1)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    return x_train, x_val, y_train, y_val


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

def train_gradient_descent(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, init_weights: np.ndarray, 
                           optimizer: str="gd", threshold: float=0.6, num_epochs: int=100000, c_1: float=1e-4, 
                           c_2: float=0.9, c: float=0.25, rho: float=0.5, init_alpha: float=2, epsilon_1: float=0.001, epsilon_2: float=0.001, 
                           epsilon_3: float=0.001, stop_condition: bool=False, lr_schdule: str="backtracking"):
    
    min_train_cost = np.inf
    min_train_cost_weights = None
    min_val_cost = np.inf
    min_val_cost_weights = None
    train_cost_list = []
    val_cost_list = []
    train_mae_list = []
    val_mae_list = []
    time_epoch_list = []
    wolfe_II_list = []
    goldstein_list = []
    delta_weights_norm_list = []
    delta_train_cost = []
    delta_val_cost = []
    train_gradient_norm_list = []
    val_gradient_norm_list = []
    train_r_2_list = []
    val_r_2_list = []
    train_adjust_r_2_list = []
    val_adjust_r_2_list = []
    inner_count_list = []

    weights = init_weights
    prev_weights = copy.deepcopy(weights)
    prev_train_cost = 0
    prev_val_cost = 0

    t = 1
    m = np.zeros_like(weights, dtype=np.float32)
    v = np.zeros_like(weights, dtype=np.float32)
    v_hat = np.zeros_like(weights, dtype=np.float32)
    d = np.zeros_like(weights, dtype=np.float32)
    u = 0.

    H = np.eye(weights.shape[0], dtype=np.float32)

    prev_w = np.zeros_like(weights, dtype=np.float32)

    #start_timestamp = time.time()

    for epoch in tqdm(range(num_epochs)):

        epoch_start = time.time()

        dweights = derivative_cost_wrt_params(x=x_train, w=weights, y=y_train)
        
        if optimizer.lower() == "gd":
            p = dweights
        elif optimizer.lower() == "accelerated":
            p, v = accelerated_gradient_step(x=x_train, w=weights, y=y_train, prev_w=prev_w, t=t)
            prev_w = copy.deepcopy(weights)
        elif optimizer.lower() == "newton":
            p = newton_step(x=x_train, dweights=dweights)
        elif optimizer.lower() in ["bfgs", "dfp"]:
            p = np.matmul(H, dweights)
            prev_w = copy.deepcopy(weights)
        elif optimizer.lower() == "adam":
            p, m, v = adam_step(dweights=dweights, m=m, v=v, t=t, beta_1=0.5, beta_2=0.9, epsilon=1e-8)
        elif optimizer.lower() == "avagrad":
            p, m, v = avagrad_step(dweights=dweights, m=m, v=v, beta_1=0.5, beta_2=0.9, epsilon=1e-8)
        elif optimizer.lower() == "radam":
            p, m, v = radam_step(dweights=dweights, m=m, v=v, t=t, beta_1=0.5, beta_2=0.9, epsilon=1e-8)
        elif optimizer.lower() == "momentum":
            p, m = momentum_step(dweights=dweights, m=m, beta_1=0.9)
        elif optimizer.lower() == "adagrad":
            p, v = adagrad_step(dweights=dweights, v=v, epsilon=1e-8)
        elif optimizer.lower() == "rmsprop":
            p, v = rmsprop_step(dweights=dweights, v=v, beta_2=0.9, epsilon=1e-8)
        elif optimizer.lower() == "adadelta":
            p, v, d = adadelta_step(dweights=dweights, v=v, d=d, alpha=init_alpha, beta_2=0.9, epsilon=1e-8)
        elif optimizer.lower() == "adamax":
            p, m, u = adamax_step(dweights=dweights, m=m, u=u, t=t, 
                                  beta_1=0.9, beta_2=0.99, epsilon=1e-8)
        elif optimizer.lower() == "nadam":
            p, m, v = nadam_step(dweights=dweights, m=m, v=v, t=t, beta_1=0.5, beta_2=0.9, epsilon=1e-8)
        elif optimizer.lower() == "amsgrad":
            p, m, v, v_hat = amsgrad_step(dweights=dweights, m=m, v=v, v_hat=v_hat, t=t,
                                          beta_1=0.5, beta_2=0.9, epsilon=1e-8)
        elif optimizer.lower() == "adabelief":
            p, m, v = adabelief_step(dweights=dweights, m=m, v=v, t=t,
                                     beta_1=0.5, beta_2=0.9, epsilon=1e-8)
        else:
            raise ValueError("No optimizer name {}".format(optimizer))

        if lr_schdule == "backtracking":
            if optimizer.lower() == "accelerated":
                alpha, inner_count = backtracking_line_search(x=x_train, w=v, y=y_train, p=-p, rho=rho, alpha=init_alpha, c=c_1)
            else:
                alpha, inner_count = backtracking_line_search(x=x_train, w=weights, y=y_train, p=-p, rho=rho, alpha=init_alpha, c=c_1)
            inner_count_list.append(inner_count)
        elif lr_schdule == "fixed":
            alpha = copy.deepcopy(init_alpha)
            inner_count_list.append(1)
        elif lr_schdule == "inverse_decay":
            alpha = inverse_decay(init_alpha=init_alpha, t=t)
            inner_count_list.append(1)
        elif lr_schdule == "warmup":
            alpha = warmup_lr(steps=t, min_lr=1e-5, max_lr=init_alpha, num_warmup_steps=500, num_total_steps=num_epochs)
            inner_count_list.append(1)
        elif lr_schdule == "cyclic":
            alpha = cyclic_lr(steps=t, min_lr=1e-5, max_lr=init_alpha, num_increase=500, num_decrease=500)
            inner_count_list.append(1)
        else:
            raise ValueError("{} scheduler is not supported".format(lr_schdule))
        
        if optimizer.lower() == "accelerated":
            weights = v - alpha * p
        elif optimizer.lower() in ["bfgs", "dfp"]:
            weights = weights - alpha * p
            delta_w = weights - prev_w
            delta_g = derivative_cost_wrt_params(x_train, w=weights, y=y_train) - dweights
            if optimizer.lower() == "bfgs":
                H = bfgs_post_step(H=H, s=delta_w, y=delta_g)
            elif optimizer.lower() == "dfp":
                H = dfp_post_step(H=H, s=delta_w, y=delta_g)
        else:
            weights = weights - alpha * p
        t += 1

        epoch_end = time.time()
        time_epoch_list.append(epoch_end - epoch_start)

        wolfe_II_list.append(check_wolfe_II(x=x_train, w=weights, y=y_train, alpha=alpha, p=-p, c_2=c_2))
        goldstein_list.append(check_goldstein(x=x_train, w=weights, y=y_train, alpha=alpha, p=-p, c=c))

        train_cost = compute_cost(x=x_train, w=weights, y=y_train)
        val_cost = compute_cost(x=x_val, w=weights, y=y_val)

        train_cost_list.append(train_cost)
        val_cost_list.append(val_cost)

        train_mae = compute_mae(x=x_train, w=weights, y=y_train)
        val_mae = compute_mae(x=x_val, w=weights, y=y_val)
        train_mae_list.append(train_mae)
        val_mae_list.append(val_mae)

        if train_cost < min_train_cost:
            min_train_cost = copy.deepcopy(train_cost)
            min_train_cost_weights = copy.deepcopy(weights)

        if val_cost < min_val_cost:
            min_val_cost = copy.deepcopy(val_cost)
            min_val_cost_weights = copy.deepcopy(weights)

        dweights = derivative_cost_wrt_params(x=x_train, w=weights, y=y_train)
        delta_weights_norm_list.append(np.linalg.norm(weights - prev_weights))
        delta_train_cost.append(np.abs(train_cost - prev_train_cost)/train_cost)
        delta_val_cost.append(np.abs(val_cost - prev_val_cost)/val_cost)
        train_gradient_norm_list.append(np.linalg.norm(dweights))
        val_dweights = derivative_cost_wrt_params(x=x_val, w=weights, y=y_val)
        val_gradient_norm_list.append(np.linalg.norm(val_dweights))
        train_r_2, train_adjust_r_2 = calculate_R(x=x_train, w=weights, y=y_train)
        val_r_2, val_adjust_r_2 = calculate_R(x=x_val, w=weights, y=y_val)
        train_r_2_list.append(train_r_2)
        train_adjust_r_2_list.append(train_adjust_r_2)
        val_r_2_list.append(val_r_2)
        val_adjust_r_2_list.append(val_adjust_r_2)

        #if (epoch + 1) % 10000 == 0:
        if (epoch + 1) % 2500 == 0:
            #print(epoch, alpha, train_cost, val_cost, train_acc, val_acc, np.linalg.norm(weights - prev_weights), np.abs(prev_train_cost - train_cost), np.linalg.norm(dweights))
            print(epoch, train_cost, val_cost, train_mae, val_mae, np.linalg.norm(weights - prev_weights), np.abs(prev_train_cost - train_cost)/train_cost, np.linalg.norm(dweights), np.linalg.norm(val_dweights), train_r_2, train_adjust_r_2, val_r_2, val_adjust_r_2)

        prev_weights = copy.deepcopy(weights)
        prev_train_cost = copy.deepcopy(train_cost)
        prev_val_cost = copy.deepcopy(val_cost)

    return weights, min_train_cost, min_train_cost_weights, min_val_cost, min_val_cost_weights, train_cost_list, val_cost_list, time_epoch_list, wolfe_II_list, goldstein_list, delta_weights_norm_list, delta_train_cost, delta_val_cost, train_gradient_norm_list, val_gradient_norm_list, train_r_2_list, train_adjust_r_2_list, val_r_2_list, val_adjust_r_2_list, train_mae_list, val_mae_list, inner_count_list

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
    #num_epochs = 20000
    num_epochs = 5000
    c_1 = 1e-2
    c_2 = 0.9
    c = 0.25
    threshold = 0.6
    rho = 0.5
    batch_size = 64
    stop_condition = False
    save_dir = "."

    #result_timestamp_epoch_list = []
    result_weights = {}
    result_min_train_cost = {}
    result_min_val_cost = {}
    result_min_train_cost_weights = {}
    result_min_val_cost_weights = {}
    result_train_cost_list = {}  
    result_val_cost_list = {}
    result_train_mae_list = {}  
    result_val_mae_list = {}
    result_time_epoch_list = {}
    result_wolfe_II_list = {}
    result_goldstein_list = {}
    result_delta_weights_norm_list = {}
    result_delta_train_cost = {}
    result_delta_val_cost = {}
    result_train_gradient_norm_list = {}
    result_val_gradient_norm_list  = {}
    result_train_r_2_list = {}
    result_train_adjust_r_2_list = {}
    result_val_r_2_list = {}
    result_val_adjust_r_2_list = {}
    result_inner_count_list = {}

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    #lr_schedule_list = ["backtracking", "fixed", "inverse_decay"]
    lr_schedule_list = ["warmup", "cyclic", "backtracking", "fixed", "inverse_decay"]

    step_length_list = [1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 5, 10]

    optimizer_list = ["gd", "Newton", "Accelerated", "BFGS", "DFP", "Adam", "Avagrad", "RAdam", "Momentum", 
                      "Adagrad", "RMSProp", "Adadelta", "Adamax", "Nadam", "AMSGrad", "AdaBelief"]
    #optimizer_list = ["gd", "Newton", "Accelerated", "BFGS", "DFP"]

    #x_train, y_train, x_val, y_val, x_test, y_test = create_data(csv_path=csv_path, normalize=normalize, use_bias=use_bias)
    x_train, x_val, y_train, y_val = create_data(csv_path=csv_path, normalize=normalize, use_bias=use_bias)

    #print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape)

    start_weights = init_weights(x=x_train, use_bias=use_bias, initializer=initializer)

    start_time = time.time()

    #print(start_weights)
    for schedule in lr_schedule_list:
        result_weights[schedule] = {}
        result_min_train_cost[schedule] = {}
        result_min_val_cost[schedule] = {}
        result_min_train_cost_weights[schedule] = {}
        result_min_val_cost_weights[schedule] = {}
        result_train_cost_list[schedule] = {}  
        result_val_cost_list[schedule] = {}
        result_train_mae_list[schedule] = {}  
        result_val_mae_list[schedule] = {}
        result_time_epoch_list[schedule] = {}
        result_wolfe_II_list[schedule] = {}
        result_goldstein_list[schedule] = {}
        result_delta_weights_norm_list[schedule] = {}
        result_delta_train_cost[schedule] = {}
        result_delta_val_cost[schedule] = {}
        result_train_gradient_norm_list[schedule] = {}
        result_val_gradient_norm_list[schedule]  = {}
        result_train_r_2_list[schedule] = {}
        result_train_adjust_r_2_list[schedule] = {}
        result_val_r_2_list[schedule] = {}
        result_val_adjust_r_2_list[schedule] = {}
        result_inner_count_list[schedule] = {}

        print("Schedule: {}".format(schedule))


        for step_length in step_length_list:
            result_weights[schedule][step_length] = {}
            result_min_train_cost[schedule][step_length] = {}
            result_min_val_cost[schedule][step_length] = {}
            result_min_train_cost_weights[schedule][step_length] = {}
            result_min_val_cost_weights[schedule][step_length] = {}
            result_train_cost_list[schedule][step_length] = {}  
            result_val_cost_list[schedule][step_length] = {}
            result_train_mae_list[schedule][step_length] = {}  
            result_val_mae_list[schedule][step_length] = {}
            result_time_epoch_list[schedule][step_length] = {}
            result_wolfe_II_list[schedule][step_length] = {}
            result_goldstein_list[schedule][step_length] = {}
            result_delta_weights_norm_list[schedule][step_length] = {}
            result_delta_train_cost[schedule][step_length] = {}
            result_delta_val_cost[schedule][step_length] = {}
            result_train_gradient_norm_list[schedule][step_length] = {}
            result_val_gradient_norm_list[schedule][step_length]  = {}
            result_train_r_2_list[schedule][step_length] = {}
            result_train_adjust_r_2_list[schedule][step_length] = {}
            result_val_r_2_list[schedule][step_length] = {}
            result_val_adjust_r_2_list[schedule][step_length] = {}
            result_inner_count_list[schedule][step_length] = {}

            print("Step length {}/{}".format(schedule, step_length))

            for optimizer in optimizer_list:

                print("Optimizer {}/{}/{}".format(schedule, step_length, optimizer))

                weights, min_train_cost, min_train_cost_weights, min_val_cost, min_val_cost_weights, train_cost_list, val_cost_list, time_epoch_list, \
                wolfe_II_list, goldstein_list, delta_weights_norm_list, delta_train_cost, delta_val_cost, train_gradient_norm_list, \
                val_gradient_norm_list, train_r_2_list, train_adjust_r_2_list, val_r_2_list, \
                val_adjust_r_2_list, train_mae_list, val_mae_list, inner_count_list = train_gradient_descent(x_train=x_train, y_train=y_train,
                                                                                                             x_val=x_val, y_val=y_val, init_weights=copy.deepcopy(start_weights),
                                                                                                             optimizer=optimizer, num_epochs=num_epochs,
                                                                                                             c_1=c_1, c_2=c_2, c=c,
                                                                                                             rho=rho, init_alpha=step_length, lr_schdule=schedule)                                           

                result_weights[schedule][step_length][optimizer] = weights
                result_min_train_cost[schedule][step_length][optimizer] = min_train_cost
                result_min_val_cost[schedule][step_length][optimizer] = min_val_cost
                result_min_train_cost_weights[schedule][step_length][optimizer] = min_train_cost_weights
                result_min_val_cost_weights[schedule][step_length][optimizer] = min_val_cost_weights
                result_train_cost_list[schedule][step_length][optimizer] = train_cost_list  
                result_val_cost_list[schedule][step_length][optimizer] = val_cost_list
                result_train_mae_list[schedule][step_length][optimizer] = train_mae_list  
                result_val_mae_list[schedule][step_length][optimizer] = val_mae_list
                result_time_epoch_list[schedule][step_length][optimizer] = time_epoch_list
                result_wolfe_II_list[schedule][step_length][optimizer] = wolfe_II_list
                result_goldstein_list[schedule][step_length][optimizer] = goldstein_list
                result_delta_weights_norm_list[schedule][step_length][optimizer] = delta_weights_norm_list
                result_delta_train_cost[schedule][step_length][optimizer] = delta_train_cost
                result_delta_val_cost[schedule][step_length][optimizer] = delta_val_cost
                result_train_gradient_norm_list[schedule][step_length][optimizer] = train_gradient_norm_list
                result_val_gradient_norm_list[schedule][step_length][optimizer] = val_gradient_norm_list
                result_train_r_2_list[schedule][step_length][optimizer] = train_r_2_list
                result_train_adjust_r_2_list[schedule][step_length][optimizer] = train_adjust_r_2_list
                result_val_r_2_list[schedule][step_length][optimizer] = val_r_2_list
                result_val_adjust_r_2_list[schedule][step_length][optimizer] = val_adjust_r_2_list
                result_inner_count_list[schedule][step_length][optimizer] = inner_count_list

    results = {"weights": result_weights, "min_train_cost_weights": result_min_train_cost_weights,
               "min_train_cost": result_min_train_cost, "min_val_cost": result_min_val_cost, 
               "min_val_cost_weights": result_min_val_cost_weights, 
               "train_cost": result_train_cost_list, "val_cost": result_val_cost_list,
               "train_mae": result_train_mae_list, "val_mae": result_val_mae_list,
               "time_epoch": result_time_epoch_list,
               "wolf_II": result_wolfe_II_list, "goldstein": result_goldstein_list,
               "delta_weights_norm": result_delta_weights_norm_list,
               "delta_train_cost": result_delta_train_cost,
               "delta_val_cost": result_delta_val_cost,
               "train_gradient_norm": result_train_gradient_norm_list,
               "val_gradient_norm": result_val_gradient_norm_list,
               "train_r2": result_train_r_2_list, "train_adjust_r2": train_adjust_r_2_list,
               "val_r2": result_val_r_2_list, "val_adjust_r2": result_val_adjust_r_2_list,
               "inner_count": result_inner_count_list}

    end_time = time.time()

    print("Takes {} to finish experiments :(".format(end_time - start_time))

    with open(os.path.join(save_dir, "results.pkl"), "wb") as f:
        pickle.dump(results, f)
        f.close()
