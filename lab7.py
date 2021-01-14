import random
from math import sqrt, log, sin, cos, pi
import numpy as np
from scipy.stats import chi2, norm, laplace, uniform

alpha = 0.05
INF = 10000000
X = np.arange(-1.8, 2.2, 0.2)
NORM_PARAMS = (0, 1)

def get_k(lenght: int):
    return int(1 + 3.3*log(lenght, 10)) 

def aver(arr: np.ndarray):
    return np.mean(arr)

def med(arr: np.ndarray):
    return np.median(arr)

def dispers(arr: np.ndarray):
    av = aver(arr)
    return np.mean(np.power(arr - av, 2))

def get_normal(a, sigma, num: int) -> list:
    rands = [random.random() for _ in range(num)]
    res = []
    for i in range(0, num - 1, 2):
        res.append(a + sigma * sqrt(-2*log(rands[i])) * sin(2*pi*rands[i+1]))
        res.append(a + sigma * sqrt(-2*log(rands[i])) * cos(2*pi*rands[i+1]))
    if i < num - 2:
        res.append(a + sigma * sqrt(-2*log(rands[i])) * sin(2*pi*rands[0]))

    return res

def get_intervals(vals: np.ndarray):
    # верхние границы интервалов
    k = get_k(len(vals))
    prop = k / len(vals)
    bot, top = np.quantile(vals, prop), np.quantile(vals, 1-prop)

    res = np.linspace(bot, top, k-1)
    return np.append(res, INF)

def count_if(arr: np.ndarray, val_bot, val_top):
    res = 0
    for v in arr:
        if val_bot < v <= val_top:
            res += 1
    return res

def get_p(a_prev, a, func=norm.cdf, func_params=NORM_PARAMS):
    if a_prev == -INF:
        return func(a, *func_params)
    return func(a, *func_params) - func(a_prev, *func_params)

NI, PI, NPI, NI_NPI, FRAC = 'ni', 'pi', 'npi', 'ni_npi', 'frac'
def get_params(x_prev, x, vals, func=norm.cdf, func_params=NORM_PARAMS):
    res = dict()
    res[NI] = count_if(vals, x_prev, x)
    res[PI] = float(round(get_p(x_prev, x, func, func_params), 4))
    res[NPI] = float(round(len(vals) * res[PI], 2))
    res[NI_NPI] = float(round(res[NI] - res[NPI], 2))
    res[FRAC] = float(round(res[NI_NPI] * res[NI_NPI] / res[NPI], 4))

    return res

def get_table_of_params(vals, func=norm.cdf, func_params=NORM_PARAMS):
    res = []
    ints = get_intervals(vals)

    prev = -INF
    for t in ints:
        res.append(get_params(prev, t, vals, func, func_params))
        prev = t

    return res

def round(num, n_digits):
    s = str(num)
    point = s.find('.')
    if len(s[point+1:]) <= n_digits:
        return s
    d_next = s[point + 1 + n_digits]
    if int(d_next) >= 5:
        add = 1
    else:
        add = 0
    res = s[:point + n_digits]
    d = int(s[point + n_digits])
    return res + str(d + add)

def get_str_table(vals, params_emp):
    ints = get_intervals(vals)
    params = get_table_of_params(vals, func_params=params_emp)
    table = '\n'

    prev = '-inf'
    for i in range(len(params)):
        t = ints[i]
        if t == INF:
            t = 'inf'
        else:
            t = round(t, 2)
        if prev != '-inf':
            prev = round(prev, 2)
        table += str(i+1) + '  |  ' + str(prev) + ', ' + str(t) + '  |  '
        param = params[i]

        table += str(param[NI]) + '  |  ' + str(param[PI]) + '  |  ' + str(param[NPI]) + '  |  ' + str(param[NI_NPI])  + '  |  ' + \
            str(param[FRAC]) + '\n'

        prev = t

    results = ['S', '   --   ']
    par = np.array([ [d[NI], d[PI], d[NPI], d[NI_NPI], d[FRAC]] for d in params])
    sums = np.sum(par, axis=0)
    results += [str(round(v, 4)) for v in sums]
    table += '  |  '.join(results) 
    return table

def main():
    print('Norm, n = 100')
    vals = norm.rvs(0, 1, 100)
    params_emp = mu, sigma = aver(vals), sqrt(dispers(vals))
    table = get_str_table(vals, params_emp)
    print('mu: ', mu, '\nsigma: ', sigma)
    print(table)
    print('\n')

    print('Laplace, n = 20')
    vals = laplace.rvs(mu, sigma/sqrt(2), 20)
    table = get_str_table(vals, params_emp)
    print('mu: ', mu, '\nsigma: ', sigma)
    print(table)
    print('\n')

    print('Uniform, n = 20')
    vals = uniform.rvs(-sqrt(3), 2*sqrt(3), 20)
    table = get_str_table(vals, params_emp)
    print('mu: ', mu, '\nsigma: ', sigma)
    print(table)
    print('\n')
    
if __name__ == '__main__':
    main()




