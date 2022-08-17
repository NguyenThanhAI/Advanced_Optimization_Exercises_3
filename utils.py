from typing import List, Tuple, Optional
import numpy as np
import scipy


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
    loss = np.mean(err**2)

    return loss


def derivative_cost_wrt_params(x: np.ndarray, w: np.ndarray, y: np.ndarray, clipped: bool=True) -> np.ndarray:
    n = x.shape[0]
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=0)
    #print(np.matmul(x.T, x))
    grad = (2 * np.matmul(w, np.matmul(x.T, x)) - 2 * np.matmul(y, x))/n
    if clipped:
        grad = np.clip(grad, -2, 2)
    return grad


def hessian_wrt_params(x: np.ndarray):
    n = x.shape[0]
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=0)

    hessian = (2 * np.matmul(x.T, x))/n

    return hessian


def newton_step(x: np.ndarray, dweights: np.ndarray):
    hessian = hessian_wrt_params(x=x)
    p = scipy.linalg.solve(hessian, dweights)
    return p


def accelerated_gradient_step(x: np.ndarray, w: np.ndarray, y: np.ndarray, prev_w, t: int):
    if t == 1:
        p = derivative_cost_wrt_params(x=x, w=w, y=y)
        return p, w
    #print(w, prev_w)
    v = w + ((t - 1) * (w - prev_w))/(t + 2)
    p = derivative_cost_wrt_params(x=x, w=v, y=y)
    return p, v


def bfgs_post_step(H: np.ndarray, s: np.ndarray, y: np.ndarray) -> np.ndarray:
    assert H.shape[0] == H.shape[1] == s.shape[0] == s.shape[0]
    denom = (np.matmul(y, s.T))
    if np.abs(denom) < 1e-8:
        rho = 1 / (1e-8)
    else:
        rho = 1/(np.matmul(y, s.T))
    I = np.eye(s.shape[0], dtype=np.float32)

    left = I - rho * np.matmul(s[:, np.newaxis], y[np.newaxis,:])
    right = I - rho * np.matmul(y[:, np.newaxis], s[np.newaxis, :])

    right_sum = rho * np.matmul(s[:, np.newaxis], s[np.newaxis, :])

    left_sum = np.matmul(np.matmul(left, H), right)

    new_H = left_sum + right_sum

    return new_H


def dfp_post_step(H: np.ndarray, s: np.ndarray, y: np.ndarray) -> np.ndarray:
    assert H.shape[0] == H.shape[1] == s.shape[0] == s.shape[0]

    nom_2 = np.matmul(np.matmul(H, y[:, np.newaxis]), np.matmul(y[np.newaxis, :], H))
    denom_2 = np.matmul(y, np.matmul(H, y))

    nom_3 = np.matmul(s[:, np.newaxis], s[np.newaxis, :])
    denom_3 = np.matmul(y, s.T)

    if np.abs(denom_2) < 1e-8:
        denom_2 = 1e-8
    if np.abs(denom_3) < 1e-8:
        denom_3 = 1e-8

    new_H = H - nom_2 / denom_2 + nom_3 / denom_3

    return new_H


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

def inverse_decay(init_alpha: float, t: int) -> float:
    return init_alpha / t


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


def avagrad_step(dweights: np.ndarray, m: np.ndarray, v: np.ndarray, beta_1: float=0.5, beta_2: float=0.99, epsilon: float=1e-8) -> np.ndarray:
    d = dweights.shape[0]
    m = beta_1 * m + (1 - beta_1) * dweights

    eta = 1 / (np.sqrt(v) + epsilon)

    p = (eta / np.linalg.norm(eta/np.sqrt(d))) * m

    v = beta_2 * v + (1 - beta_2) * dweights**2

    return p, m, v


def radam_step(dweights: np.ndarray, m: np.ndarray, v: np.ndarray, t: int, beta_1: float=0.5, beta_2: float=0.99, epsilon: float=1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho_inf = (2 / (1 - beta_2)) - 1

    m = beta_1 * m + (1 - beta_1) * dweights
    v = beta_2 * v + (1 - beta_2) * dweights**2

    m_hat = m / (1 - beta_1**t)

    rho = rho_inf - (2 * t * beta_2**t) / (1 - beta_2**t)

    if rho > 4:
        v_hat = np.sqrt(v / (1 - beta_2**t))
        r = np.sqrt(((rho - 4) * (rho - 2) * rho_inf)/((rho_inf - 4) * (rho_inf - 2) * rho))
        p = (r * m_hat) / (v_hat + epsilon)
    else:
        p = m_hat

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

def calculate_R(x: np.ndarray, w: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    pre_y = predict(x=x, w=w)
    rss = np.mean((y - pre_y)**2)
    sst = np.mean((y - np.mean(y))**2)

    r_2 = 1 - rss / sst

    n = x.shape[0]
    assert x.shape[0] == y.shape[0]

    d = w.shape[0]

    adjusted_r_2 = 1 - ((n - 1) * (1 - r_2)) / (n - d)

    return r_2, adjusted_r_2


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

grad = (2 * np.linalg.multi_dot([w, x.T, x]) - 2 * np.matmul(y, x))/x.shape[0]
print(grad)

hessian = hessian_wrt_params(x=x)
print(hessian)

print(x.T, x)'''