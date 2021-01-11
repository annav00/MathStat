import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
from scipy.optimize import minimize

a = 2
b = 2

def minimized_function(params, x, y):
    c, d = params
    res = float(0)
    for i in range(x.size):
        res += abs(c * x[i] + d - y[i])
    return res

def lsm(x, y):
    betta1 = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x * x) - np.mean(x) ** 2)
    betta0 = np.mean(y) - betta1 * np.mean(x)
    return betta0, betta1

def lad(x0, x, y):
    result = minimize(minimized_function, x0, args=(x, y), method='Nelder-Mead')
    beta0, beta1 = result.x[0], result.x[1]
    return beta0, beta1
  
def lin_reg_coef_estimates(x, y):
    beta11, beta12 = lsm(x, y)
    print("МНК:")
    print('a = ' + str(beta11))
    print('b = ' + str(beta12))
    
    x0 = [beta11, beta12]
    beta21, beta22 = lad(x0, x, y)
    
    print("МНМ:")
    print('a = ' + str(beta21))
    print('b = ' + str(beta22))

    plt.scatter(x[1:-2], y[1:-2],  label='Выборка', edgecolors="black", color='white')
    plt.plot(x, a * np.ones(20) + x * (b * np.ones(20)), label='Модель', color='red')
    plt.plot(x, beta11 * np.ones(20) + x * (beta12 * np.ones(20)), label='МНК', color='black')
    plt.plot(x, beta21 * np.ones(20) + x * (beta22 * np.ones(20)), label='МНМ', color='green')
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


x = np.arange(-1.8, 2.2, 0.2)
y = a * x + b * np.ones(x.size) + np.random.normal(size=x.size)
lin_reg_coef_estimates(x, y)
y[0] = 10
y[-1] = -10
lin_reg_coef_estimates(x, y)