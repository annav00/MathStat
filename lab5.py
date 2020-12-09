import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms



def confidence_ellipse(x, y, mean, cov, ax, n_std=3.0, facecolor='none', edgecolor = 'red', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    ell_radius_x = np.sqrt(1 + cov[0, 1])
    ell_radius_y = np.sqrt(1 - cov[0, 1])
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      edgecolor='b', lw=1,
                      facecolor=facecolor, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_x = mean[0]
    mean_y = mean[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def quadrant_coeff(array):
    x = array[:, 0] - np.median(array[:, 0])
    y = array[:, 1] - np.median(array[:, 1])
    
    n13 = ((x > 0) & (y > 0)) | ((x < 0) & (y < 0))
    n24 = ((x < 0) & (y > 0)) | ((x > 0) & (y < 0))

    return (n13.sum() - n24.sum()) / array.shape[0]

def square_mean(array):
    return np.mean(array.values ** 2)


numOfCalculations = 1000
j = 0
for size in [20, 60, 100]:
    for r in  [0, 0.5, 0.9]:
        mean = [0, 0]
        cov = np.array([[1, r], [r, 1]])
        j += 1
        
        figure, ax = plt.subplots(1, 1)
        plt.title('n = ' + str(size) + ', p = ' + str(r))
        data = np.random.multivariate_normal(mean, cov, size=size)
        ax.scatter(data[:, 0], data[:, 1], c = 'red', s = 3, marker = "X")
        confidence_ellipse(data[:, 0], data[:, 1], mean, cov, ax)
        plt.xlim((-4, 4))
        plt.ylim((-4, 4))
        plt.show()
         
        correlation_coefficients = np.zeros((numOfCalculations, 3))
        for i in range(numOfCalculations):
            data = np.random.multivariate_normal(mean, cov, size=size)
            correlation_coefficients[i, 0] = stats.pearsonr(data[:, 0], data[:, 1])[0]
            correlation_coefficients[i, 1] = stats.spearmanr(data)[0]
            correlation_coefficients[i, 2] = quadrant_coeff(data)
    
        correlation_coefficients = pd.DataFrame(correlation_coefficients, columns=['Pearson', 'Spearman', 'Quadrant'])
        characteristics = correlation_coefficients.aggregate([np.mean, square_mean, np.var], axis=0)
        
        print('n = ' + str(size) + ', p = ' + str(r))
        print(characteristics)
   

