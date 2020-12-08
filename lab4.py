from scipy.stats import norm, laplace, poisson, cauchy, uniform
import numpy as np
import matplotlib.pyplot as plt
import math as m
from collections import defaultdict

borders = [-4, 4]
bordersP = [4, 16]
a = 0
b = 1


def nameRvs (name, size):
     if name == "Normal":
                return (norm.rvs(size = size))
     if name == "Cauchy":
                return (cauchy.rvs(size = size))
     if name == "Laplace":
                return (laplace.rvs(loc = 0, scale = (1 / m.sqrt(2)), size = size))              
     if name == "Poisson":
                return (poisson.rvs(mu = 10, size = size))
     if name == "Uniform":
                return (uniform.rvs(loc = -m.sqrt(3),  scale = (2 * m.sqrt(3)), size = size))
            
def nameFun (name):
     if name == "Normal":
                return norm()
     if name == "Cauchy":
                return cauchy()
     if name == "Laplace":
                return laplace(scale=1 / m.sqrt(2), loc=0)           
     if name == "Poisson":
                return poisson(10)
     if name == "Uniform":
                return uniform(loc=-m.sqrt(3), scale=2 * m.sqrt(3))


def empFunc(distr, distr_f, name, size, a, b):
    abscissa = np.arange(a,b, 0.01)
    d, sorted_x = countVals(distr)
    y = valueArray(d, sorted_x, size, abscissa)
    plt.plot(abscissa, y, color='black')
    plt.plot(abscissa, distr_f.cdf(abscissa), color='blue')
    plt.title(name)
    plt.xlabel('n = ' + str(size))
    plt.show()

def countVals(distr):
    d = defaultdict(int)
    vals = []
    for x in distr:
        d[x] += 1
    for val in d:
        vals.append(val)
    vals.sort()
    return d, vals

def densityFunc(distr, distr_f, name, size, h):
    if name == "Poisson":
        abscissa = np.arange(bordersP[a], bordersP[b] + 1, 1)
        plt.plot(abscissa, poisson.pmf(10, abscissa), lw=2, color='blue')
        abscissa = np.arange(bordersP[a], bordersP[b], 0.01)
    else:
        abscissa = np.arange(borders[a], borders[b], 0.01)
        plt.plot(abscissa, distr_f.pdf(abscissa), color='blue')
        
    vals = []
    for i in distr:
        vals.append(i)
    vals.sort()
    
    y = []
    for point in abscissa:
        y.append(fun(point, size, vals, h))
    plt.plot(abscissa, y, color='black')
    plt.ylim(0, 1)
    plt.title(name)
    plt.xlabel('n = ' + str(size) + ', h = ' + str(h) + 'h_n')
    plt.ylabel('f(x)')
    plt.show()

def fun(point, n, array, h):
    sum = 0
    for i in range(0, n):
        u = (point-array[i])/ (h*1.06*np.std(array)*((n+2)**(-1/5)))
        deg = -u*u/2
        val = m.exp(deg)
        sum += val * (1/m.sqrt(2*m.pi))
    return sum*(1/(n*(h*1.06*np.std(array)*((n+1)**(-1/5)))))

def valueArray(d, sorted_vals, size, array):
    sorted_y_val = []
    for point in sorted_vals:
        value = 0
        for val in sorted_vals:
            if val > point:
                break
            value += d[val]
        sorted_y_val.append(value / size)

    result = []
    for x in array:
        i, to_append_val = 0, 0
        while len(sorted_vals) > i and sorted_vals[i] <= x:
            to_append_val = sorted_y_val[i]
            i += 1
        result.append(to_append_val)
    return result


names = ["Normal", "Cauchy", "Laplace", "Poisson", "Uniform"]
for name  in names:
    func = nameFun(name)
    for size in [20, 60, 100]: 
        rvs = nameRvs(name, size)
        if name == "Poisson":
            empFunc(rvs, func, name, size, bordersP[a], bordersP[b])
        else:
            empFunc(rvs, func, name, size, borders[a], borders[b])
        for h_coeff in [0.5, 1, 2]:
            densityFunc(rvs, func, name, size, h_coeff)
