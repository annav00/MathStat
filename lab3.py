from scipy.stats import norm, cauchy, laplace, poisson, uniform
import numpy as np
import matplotlib.pyplot as plt
import math as m
from collections import defaultdict


def createDist(size):
    norm_rvs = norm.rvs(size = size)
    cauchy_rvs = cauchy.rvs(size = size)
    laplace_rvs = laplace.rvs(loc = 0, scale = (1 / m.sqrt(2)), size = size)
    poisson_rvs = poisson.rvs(mu = 10, size = size)
    uniform_rvs = uniform.rvs(loc = -m.sqrt(3),  scale = (2 * m.sqrt(3)), size = size)
    all_distributions = [norm_rvs, cauchy_rvs, laplace_rvs, poisson_rvs, uniform_rvs]
    return all_distributions
    

def pltDistr (distribution, names):
     build = list(zip(distribution, names))
     for distr, name in build:
        fig, ax = plt.subplots()
        ax.set_title(name + ", n = " + str(size))
        ax.boxplot(distr, vert = False)
        plt.show()
        
def blowout(distr):
    q1 = np.quantile(distr, 1/4)
    q3 = np.quantile(distr, 3/4)
    x1 = q1 - 3 / 2 * (q3 - q1)
    x2 = q3 + 3 / 2 * (q3 - q1)
    
    blowout =  [x for x in distr if x > x2 or x < x1]
    return len(blowout)



names = ["Normal", "Cauchy", "Laplace", "Poisson", "Uniform"]
numOfCalculations = 1000

for size in [20, 100]:
    main_distr = defaultdict(float)
    distribution = createDist(size)
    pltDistr(distribution, names)
        
    for i in range(numOfCalculations):
        distribution = createDist(size)
        for j in range(len(distribution)):
            name = names[j]
            main_distr[name] += blowout(distribution[j])
            
    for name in main_distr:
        main_distr[name] = main_distr[name] / size / numOfCalculations
    print(main_distr)
