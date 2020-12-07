import math as m
from scipy.stats import norm, cauchy, laplace, poisson, uniform
import numpy as np
import matplotlib.pyplot as plt

def plot_hist(hist, density, name, size):
        figure, wind = plt.subplots(1, 1)
        wind.hist(hist, density = True, histtype = 'stepfilled', alpha = 0.75)
        if hist is poisson_rvs:
            x = np.arange(poisson.ppf(0.01, 10), poisson.ppf(0.99, 10))
            wind.plot(x, density.pmf(x))
        else:
            x = np.linspace(density.ppf(0.01), density.ppf(0.99), 100)
            wind.plot(x, density.pdf(x))
        wind.set_xlabel(name)
        wind.set_ylabel("Плотность")
        wind.set_title("Размер выборки: " + str(size))
        plt.show()

for size in [10, 50, 1000]:
    norm_rvs = norm.rvs(size = size)
    cauchy_rvs = cauchy.rvs(size = size)
    laplace_rvs = laplace.rvs(loc = 0, scale = (1 / m.sqrt(2)), size = size)
    poisson_rvs = poisson.rvs(mu = 10, size = size)
    uniform_rvs = uniform.rvs(loc = -m.sqrt(3),  scale = (2 * m.sqrt(3)), size = size)
    plot_hist(norm_rvs, norm(), "Нормальное распределение", size)
    plot_hist(cauchy_rvs, cauchy(), "Распределение Коши", size)
    plot_hist(laplace_rvs, laplace(loc = 0, scale = (1 / m.sqrt(2))), "Распределение Лапласа", size)
    plot_hist(poisson_rvs, poisson(mu = 10), "Распределение Пуассона", size)
    plot_hist(uniform_rvs, uniform(loc = -m.sqrt(3), scale = (2 * m.sqrt(3))), "Равномерное распределение", size)