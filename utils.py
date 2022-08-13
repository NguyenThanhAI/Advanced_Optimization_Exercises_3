from typing import List, Tuple, Optional
import numpy as np


class AverageMeter(object):

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def exponential_moving_average(signal: List, weight: float) -> np.ndarray:
    ema = np.zeros(len(signal))
    ema[0] = signal[0]

    for i in range(1, len(signal)):
        ema[i] = (signal[i] * weight) + (ema[i - 1] * (1 - weight))

    return ema


def predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    assert len(x.shape) <= 2
    if len(x.shape) > 1:
        assert w.shape[0] == x.shape[1]
    else:
        assert w.shape[0] == x.shape[0]

    return np.matmul(w, x.T)
    #return np.matmul(x, w)
    #return np.matmul(w, x)

def compute_cost(x: np.ndarray, w: np.ndarray, y: np.ndarray) -> float:
    assert len(x.shape) <= 2
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]
    if len(x.shape) > 1:
        assert w.shape[0] == x.shape[1]
    else:
        assert w.shape[0] == x.shape[0]

    pre_y = predict(x=x, w=w)
    err = pre_y - y
    loss = np.sum(err**2)

    return loss


def derivative_cost_wrt_params(x: np.ndarray, w: np.ndarray, y: np.ndarray) -> np.ndarray:
    
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=0)
    #print(np.matmul(x.T, x))
    grad = 2 * np.matmul(w, np.matmul(x.T, x)) - 2 * np.matmul(y, x)

    return grad


def hessian_wrt_params(x: np.ndarray):
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=0)

    hessian = 2 * np.matmul(x.T, x)

    return hessian


def backtracking_line_search(x: np.ndarray, w: np.ndarray, y: np.ndarray, p: np.ndarray, rho: float=0.9, alpha: float=5, c: float=1e-3) -> Tuple[float, int]:
    # Note that p = -grad we use + alpha, if p = grad we use - alpha
    gradient = derivative_cost_wrt_params(x=x, w=w, y=y)
    f_new = compute_cost(x=x, w=w+alpha*p, y=y)
    f_old = compute_cost(x=x, w=w, y=y)
    right_term = np.sum(gradient * p)
    #while f_new > f_old + c * alpha * np.sum(gradient * p):
    inner_count: int = 0
    while f_new > f_old + c * alpha * right_term:
        alpha = rho * alpha
        #print("f_new: {}, f_old: {}, alpha: {}".format(f_new, f_old, alpha))
        f_new = compute_cost(x=x, w=w+alpha*p, y=y)
        inner_count += 1

    return alpha, inner_count


def check_wolfe_II(x: np.ndarray, w: np.ndarray, y: np.ndarray, alpha: float, p: np.ndarray, c_2: float=0.9) -> bool:
    new_gradient = derivative_cost_wrt_params(x=x, w=w+alpha*p, y=y)
    gradient = derivative_cost_wrt_params(x=x, w=w, y=y)

    if np.sum(new_gradient * p) >= c_2 * np.sum(gradient * p):
        return True

    else:
        return False


def check_goldstein(x: np.ndarray, w: np.ndarray, y: np.ndarray, alpha: float, p: np.ndarray, c: float=0.25) -> bool:
    assert 0 < c < 0.5

    gradient = derivative_cost_wrt_params(x=x, w=w, y=y)
    f_new = compute_cost(x=x, w=w+alpha*p, y=y)
    f_old = compute_cost(x=x, w=w, y=y)

    if f_old + (1 - c) * alpha * np.sum(gradient * p) <= f_new and f_new <= f_old + c * alpha * np.sum(gradient * p):
        return True
    else:
        return False


def adam_step(dweights: np.ndarray, m: np.ndarray, v: np.ndarray, t: int, beta_1: float=0.5, beta_2: float=0.99, epsilon: float=1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = beta_1 * m + (1 - beta_1) * dweights
    v = beta_2 * v + (1 - beta_2) * dweights**2

    m_hat = m / (1 - beta_1**t)
    v_hat = v / (1 - beta_2**t)

    p = m_hat / (np.sqrt(v_hat) + epsilon)

    return p, m, v


def adamax_step(dweights: np.ndarray, m: np.ndarray, u: float, t: int, beta_1: float=0.9, beta_2: float=0.99, epsilon: float=1e-8) -> Tuple[np.ndarray, np.ndarray, float]:

    m = beta_1 * m + (1 - beta_1) * dweights
    u = np.maximum(beta_2 * u, np.max(dweights))

    p =  m / ((1 - beta_1**t) * u + epsilon)

    return p, m, u

def nadam_step(dweights: np.ndarray, m: np.ndarray, v: np.ndarray, t: int, beta_1: float=0.5, beta_2: float=0.99, epsilon: float=1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = beta_1 * m + (1 - beta_1) * dweights
    v = beta_2 * v + (1 - beta_2) * dweights**2

    m_hat = m / (1 - beta_1**t)
    v_hat = v / (1 - beta_2**t)
    
    p = (beta_1 * m_hat + ((1 - beta_1)/(1 - beta_1**t))*dweights) / (np.sqrt(v_hat) + epsilon)

    return p, m, v


def amsgrad_step(dweights: np.ndarray, m: np.ndarray, v: np.ndarray, v_hat: np.ndarray, t: int, beta_1: float=0.5, beta_2: float=0.99, epsilon: float=1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    m = beta_1 * m + (1 - beta_1) * dweights
    v = beta_2 * v + (1 - beta_2) * dweights**2

    v_hat = np.maximum(v_hat, v)

    p =  m / (np.sqrt(v_hat) + epsilon)

    return p, m, v, v_hat


def adabelief_step(dweights: np.ndarray, m: np.ndarray, v: np.ndarray, t: int, beta_1: float=0.5, beta_2: float=0.99, epsilon: float=1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = beta_1 * m + (1 - beta_1) * dweights
    v = beta_2 * v + (1 - beta_2) * (dweights - m) ** 2

    m_hat = m / (1 - beta_1**t)
    v_hat = v / (1 - beta_2**t)

    p = m_hat / (np.sqrt(v_hat) + epsilon)

    return p, m, v


def adagrad_step(dweights: np.ndarray, v: np.ndarray, epsilon: float=1e-8) -> Tuple[np.ndarray, np.ndarray]:
    v = v + dweights**2

    p = dweights / (np.sqrt(v) + epsilon)

    return p, v


def rmsprop_step(dweights: np.ndarray, v: np.ndarray, beta_2: float=0.9, epsilon: float=1e-8) -> Tuple[np.ndarray, np.ndarray]:
    v = beta_2 * v + (1 - beta_2) * dweights ** 2

    p = dweights / (np.sqrt(v) + epsilon)

    return p, v


def momentum_step(dweights: np.ndarray, m: np.ndarray, beta_1: float=0.9) -> Tuple[np.ndarray, np.ndarray]:
    m = beta_1 * m + (1 - beta_1) * dweights

    p = m

    return p, m


def adadelta_step(dweights: np.ndarray, v: np.ndarray, d: np.ndarray, alpha: float, beta_2: float=0.99, epsilon: float=1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = beta_2 * v + (1 - beta_2) * dweights ** 2

    p = (np.sqrt(d + epsilon) * dweights) / (np.sqrt(v + epsilon))

    delta_w = - alpha * p

    d = beta_2 * d + (1 - beta_2) * delta_w ** 2

    return p, v, d


'''x=np.array([[2, 1, 3], [-2, 1, -3]])
#x=np.array([2, 1, 3])
w=np.array([1, 1, 1])
y=np.array([0, 1])
#y = np.array([2])

pre_y = predict(x=x, w=w)
print(pre_y)

cost = compute_cost(x=x, w=w, y=y)
print(cost)

dweights = derivative_cost_wrt_params(x=x, w=w, y=y)
print(dweights)

print(x.T, x)'''