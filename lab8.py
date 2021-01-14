from scipy.stats import norm, t, chi2, moment
import numpy as np
import statistics
from math import sqrt

size = [20, 100]
df = 0.95
    
def mean_cie(data, df):
    n = data.size
    m, s = np.mean(data), statistics.variance(data)
    term = s * t.ppf((1 + df) / 2., n - 1) / sqrt(n - 1)
    a = m - term
    b = m + term
    return a, b

def variance_cie(data, df):
    n = data.size
    m, s = np.mean(data), statistics.variance(data)
    a = s * sqrt(n / (chi2.ppf((1 + df) / 2, n - 1)))
    b = s * sqrt(n / (chi2.ppf((1 - df) / 2, n - 1)))
    return a, b

def mean_aie(data, df):
    n = data.size
    m, s = np.mean(data), statistics.variance(data)
    u = norm.ppf((1 + df) / 2)
    term = s * u / sqrt(n)
    a = m - term
    b = m + term
    return a, b


def variance_aie(data, df):
    n = data.size
    m, s = np.mean(data), statistics.variance(data)
    m4 = moment(data, 4)
    e = m4 / s**4 - 3
    u = norm.ppf((1 + df) / 2)
    U = u * sqrt((e + 2) / n)
    a = s * sqrt(1/(1 + U)) 
    b = s * sqrt(1/(1 - U)) 
    return a, b

for i in range(len(size)):
    print(f'n = {size[i]}')
    data = np.random.standard_normal(size[i])
    print('mean interval estimation:' + str(mean_cie(data, df)))
    print('variance interval estimation:' + str(variance_cie(data, df)))
    print('mean asimptotic interval estimation:' + str(mean_aie(data, df)))
    print('variance asimptotic interval estimation: ' + str(variance_aie(data, df)))