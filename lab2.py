import numpy as np
import math
from scipy.stats import norm, cauchy, laplace, poisson, uniform

numOfCalculations = 1000


def nameRvs (name):
     if name == "Normal":
                return sorted(norm.rvs(size = size))
     if name == "Cauchy":
                return sorted(cauchy.rvs(size = size))
     if name == "Laplace":
                return sorted(laplace.rvs(loc = 0, scale = (1 / math.sqrt(2)), size = size))              
     if name == "Poisson":
                return sorted(poisson.rvs(mu = 10, size = size))
     if name == "Uniform":
                return sorted(uniform.rvs(loc = -math.sqrt(3),  scale = (2 * math.sqrt(3)), size = size))
        
def findAv(array, n):
    return 1/n * np.sum(array)


def findMed(array, n):
    l = int(n/2)
    if n % 2 == 0:
        return (array[l]+array[l-1])/2
    else:
        return array[l]

    
def findZR(array, n):
    return (array[0] + array[n - 1])/2


def findQuartile(p, array, n):
    if (n*p).is_integer():
        return array[int(n*p-1)]
    else:
        return array[int(n*p)]
    
    
def findZQ(array, n):
    return (findQuartile(1/4, array, n) + findQuartile(3/4, array, n))/2
    

def findZtr(array, n):
    r = int(n/4)
    sliceArray = 0
    for i in range(r, n-r):
        sliceArray += array[i]
    return 1/(n-2*r) * sliceArray


def findZ(array, n): 
        averageSeq.append(findAv(array, n))
        medSeq.append(findMed(array, n))
        z_rSeq.append(findZR(array, n))
        z_qSeq.append(findZQ(array, n))
        z_trSeq.append(findZtr(array, n))

def findE(averageSeq, medSeq, z_rSeq, z_qSeq, z_trSeq):
    result = []
    result.append(findAv(averageSeq,numOfCalculations))
    result.append(findAv(medSeq,numOfCalculations))
    result.append(findAv(z_rSeq,numOfCalculations))
    result.append(findAv(z_qSeq,numOfCalculations))
    result.append(findAv(z_trSeq,numOfCalculations))
    return result
#[av, med, z_r, z_q, z_tr]

def findD(averageSeq, medSeq, z_rSeq, z_qSeq, z_trSeq):
    result = []
    i = 0
    average2Seq = []
    med2Seq = []
    z_r2Seq = []
    z_q2Seq = []
    z_tr2Seq = []
    for element in averageSeq:
        average2Seq.append(element**2)
        
    for element in medSeq:
        med2Seq.append(element**2)
        
    for element in z_rSeq:
        z_r2Seq.append(element**2)
        
    for element in z_qSeq:
        z_q2Seq.append(element**2)
        
    for element in z_trSeq:
        z_tr2Seq.append(element**2)
    
    z2 = findE(average2Seq, med2Seq, z_r2Seq, z_q2Seq, z_tr2Seq)
    z = findE(averageSeq, medSeq, z_rSeq, z_qSeq, z_trSeq)
    while i < len(z2):
        result.append(z2[i] - z[i]**2)
        i += 1;
    return result

for size in [10, 100, 1000]:
    names = ["Normal", "Cauchy", "Laplace", "Poisson", "Uniform"]
    
    for name in names:
        averageSeq = []
        medSeq = []
        z_rSeq = []
        z_qSeq = []
        z_trSeq = []
        
        i = 0;
        while i < numOfCalculations:
            array = []
            array = nameRvs(name)
            findZ(array, size)
            i += 1
            
        E = findE(averageSeq, medSeq, z_rSeq, z_qSeq, z_trSeq) 
        D = findD(averageSeq, medSeq, z_rSeq, z_qSeq, z_trSeq)
        print(name, " ", size, ": ")
        print(E)
        print(D)
        
