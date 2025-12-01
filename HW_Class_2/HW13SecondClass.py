import Stats_Functions as stat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# M = 780
# S = 120
# a = 0.73
# b = 94
# sigma = 65

paramsAprior = [0.73, 94, 65, 780, 120]
paramsBprior = [0.52, 37, 71, 910, 140]

A, B, Tsqrd = stat.posteriorParameters(paramsAprior[0], paramsAprior[1], paramsAprior[2], paramsAprior[3], paramsAprior[4])
SC = stat.SC(paramsAprior[0], paramsAprior[2])
IS = stat.IS(paramsAprior[0], paramsAprior[2], paramsAprior[4])
paramsAposterior = [A, B, np.sqrt(Tsqrd), SC, IS]

A, B, Tsqrd = stat.posteriorParameters(paramsBprior[0], paramsBprior[1], paramsBprior[2], paramsBprior[3], paramsBprior[4])
SC = stat.SC(paramsBprior[0], paramsBprior[2])
IS = stat.IS(paramsBprior[0], paramsBprior[2], paramsBprior[4])
paramsBposterior = [A, B, np.sqrt(Tsqrd), SC, IS]

P = [(762, 873), (745, 813), (752, 907), (790, 858), (803, 958), (857, 993), (913, 974), (840, 918), (706, 852)]

cityA = [x[0] for x in P]
cityB = [x[1] for x in P]

gamma = stat.getCorrelation(cityA, cityB)

gammac = stat.getPosteriorCorrelation(gamma, paramsBposterior[4], paramsAposterior[4])

# print(np.sqrt(gammac))

mu = paramsAprior[3] + paramsBprior[3]
sigma = np.sqrt(paramsAprior[4] ** 2 + paramsBprior[4] ** 2 + 2 * gamma * paramsAprior[4] * paramsBprior[4])

# print(mu, sigma)
jvals = [1,2]
Avals = [paramsAposterior[0], paramsBposterior[0]]
Bvals = [paramsAposterior[1], paramsBposterior[1]]
Tvals = [paramsAposterior[2], paramsBposterior[2]]
means = [f'{str(np.round(float(a), decimals=4))}x + {str(np.round(float(b), decimals=4))}' for a,b in zip(Avals, Bvals)]

colNames = ['j', 'A', 'B', 'T', 'Ax + B']
# stat.printInLatexTable([jvals, Avals, Bvals, Tvals, means], colNames)

c2 = gamma * paramsBprior[4] / paramsAprior[4]
d2 = paramsBprior[3] - gamma * paramsAprior[3] * paramsBprior[4] / paramsAprior[4]
tau2sqrd = (paramsBprior[4] ** 2) * (1 - gamma ** 2)
# print(c2, d2, tau2sqrd)

step = 5

def phi(w1, w2, x1, x2):
    g2mean = paramsBprior[3]
    g2std = paramsBprior[4]
    g21mean = c2 * w1 + d2
    g21std = np.sqrt(tau2sqrd)
    k2mean = paramsBprior[0] * paramsBprior[3] + paramsBprior[1]
    k2std = np.sqrt((paramsBprior[0] ** 2) * (paramsBprior[4] ** 2) + (paramsBprior[2] ** 2))
    k21mean = paramsBprior[0] * (c2 * (paramsAposterior[0] * x1 + paramsAposterior[1]) + d2) + paramsBprior[1]
    k21std = np.sqrt((paramsBprior[0] ** 2) * (c2 ** 2) * (paramsAposterior[2] ** 2) + (paramsAprior[0] ** 2) * (tau2sqrd) + (paramsBprior[2] ** 2))
    phi1mean = paramsAposterior[0] * x1 + paramsAposterior[1]
    phi1std = paramsAposterior[2]
    phi2mean = paramsBposterior[0] * x2 + paramsBposterior[1]
    phi2std = paramsBposterior[2]


    k2x2 = stat.normaldf(np.array(x2), k2std, k2mean)
    k21x2 = stat.normaldf(np.array(x2), k21std, k21mean)

    g21w2 = stat.normaldf(np.array(w2), g21std, g21mean)
    g2w2 = stat.normaldf(np.array(w2), g2std, g2mean)

    phi1w1 = stat.normaldf(np.array(w1), phi1std, phi1mean)
    phi2w2 = stat.normaldf(np.array(w2), phi2std, phi2mean)

    return (k2x2 / k21x2) * (g21w2 / g2w2) * phi1w1 * phi2w2

def getPostDensity(x1, x2):
    phi_vec = np.vectorize(lambda w1, w2: phi(w1, w2, x1, x2))
    lowVal1 = paramsAprior[3] - 3 * paramsAprior[4]
    upVal1 = paramsAprior[3] + 3 * paramsAprior[4]
    lowVal2 = paramsBprior[3] - 3 * paramsBprior[4]
    upVal2 = paramsBprior[3] + 3 * paramsBprior[4]
    w1_vals = np.arange(lowVal1, upVal1, step)
    w2_vals = np.arange(lowVal2, upVal2, step)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    result = phi_vec(W1, W2)

    wvals = np.arange(lowVal1 + lowVal2, upVal1 + upVal2, step)
    outputList = []
    tol = step / 100
    for w in wvals.tolist():
        mask = np.isclose(W1, -W2 + w, atol = tol)
        outputList.append(result[mask].sum())
    return wvals, np.array(outputList)
wvals, phi600400 = getPostDensity(600, 400)
wvals, phi900700 = getPostDensity(900, 700)

def priorDensity(w1, w2):
    preExp = 1 / (2 * math.pi * paramsAprior[4] * paramsBprior[4] * np.power(1 - gamma ** 2, 1/2))
    coefExp = -1 / (2 * (1 - gamma ** 2))
    wPartExp1 = np.power((w1 - paramsAprior[3]) / paramsAprior[4], 2)
    wPartExp2 = -2 * gamma * (w1 - paramsAprior[3]) * (w2 - paramsBprior[3]) / (paramsAprior[4] * paramsAprior[4])
    wPartExp3 = np.power((w2 - paramsBprior[3]) / paramsBprior[4], 2)
    expPart = coefExp * (wPartExp1 + wPartExp2 + wPartExp3)
    return preExp * np.exp(expPart)

phi_vec = np.vectorize(lambda w1, w2: priorDensity(w1, w2))
lowVal1 = paramsAprior[3] - 3 * paramsAprior[4]
upVal1 = paramsAprior[3] + 3 * paramsAprior[4]
lowVal2 = paramsBprior[3] - 3 * paramsBprior[4]
upVal2 = paramsBprior[3] + 3 * paramsBprior[4]
w1_vals = np.arange(lowVal1, upVal1, step)
w2_vals = np.arange(lowVal2, upVal2, step)
W1, W2 = np.meshgrid(w1_vals, w2_vals)
result = phi_vec(W1, W2)

outputList = []
tol = step / 100
for w in wvals.tolist():
    mask = np.isclose(W1, -W2 + w, atol = tol)
    outputList.append(result[mask].sum())
priorPhi = np.array(outputList)

figName = 'Density Functions W = W1 + W2.png'
plt.scatter(x = wvals, y = phi600400, color = 'lightgray', s = 6, label = 'Posterior Density (X1 = 600, X2 = 400)')
plt.scatter(x = wvals, y = phi900700, color = 'darkgray', s = 6, label = 'Posterior Density (X1 = 900, X2 = 700)')
plt.scatter(x = wvals, y = priorPhi, color = 'gray', s= 6, label = 'Prior Density Function')
plt.title("Prior and Posterior Density Function of W")
plt.xlabel("Engine Oil Required")
plt.ylabel("Probability Density")
plt.legend()
plt.savefig(stat.getDownloadsTab() + figName)
stat.createFigureLatex(13, figName)
plt.clf()

def phiNoCor(w1, w2, x1, x2):
    phi1mean = paramsAposterior[0] * x1 + paramsAposterior[1]
    phi1std = paramsAposterior[2]
    phi2mean = paramsBposterior[0] * x2 + paramsBposterior[1]
    phi2std = paramsBposterior[2]

    phi1w1 = stat.normaldf(np.array(w1), phi1std, phi1mean)
    phi2w2 = stat.normaldf(np.array(w2), phi2std, phi2mean)

    return phi1w1 * phi2w2

def getPostDensityNoCor(x1, x2):
    phi_vec = np.vectorize(lambda w1, w2: phiNoCor(w1, w2, x1, x2))
    lowVal1 = paramsAprior[3] - 3 * paramsAprior[4]
    upVal1 = paramsAprior[3] + 3 * paramsAprior[4]
    lowVal2 = paramsBprior[3] - 3 * paramsBprior[4]
    upVal2 = paramsBprior[3] + 3 * paramsBprior[4]
    w1_vals = np.arange(lowVal1, upVal1, step)
    w2_vals = np.arange(lowVal2, upVal2, step)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    result = phi_vec(W1, W2)

    wvals = np.arange(lowVal1 + lowVal2, upVal1 + upVal2, step)
    outputList = []
    tol = step / 100
    for w in wvals.tolist():
        mask = np.isclose(W1, -W2 + w, atol = tol)
        outputList.append(result[mask].sum())
    return wvals, np.array(outputList)
wvals, phi600400 = getPostDensityNoCor(600, 400)
wvals, phi900700 = getPostDensityNoCor(900, 700)

figName = 'Density Functions W = W1 + W2 No Correlation.png'
plt.scatter(x = wvals, y = phi600400, color = 'lightgray', s = 6, label = 'Posterior Density (X1 = 600, X2 = 400)')
plt.scatter(x = wvals, y = phi900700, color = 'darkgray', s = 6, label = 'Posterior Density (X1 = 900, X2 = 700)')
plt.scatter(x = wvals, y = priorPhi, color = 'gray', s= 6, label = 'Prior Density Function')
plt.title("Prior and Posterior Density Function of W No Correlation")
plt.xlabel("Engine Oil Required")
plt.ylabel("Probability Density")
plt.legend()
plt.savefig(stat.getDownloadsTab() + figName)
stat.createFigureLatex(13, figName)
plt.clf()

# print(paramsAposterior)

# A, B, Tsqrd = stat.posteriorParameters(a, b, sigma, M, S)
# print(A, B, Tsqrd)
# SC = stat.SC(a, sigma)
# IS = stat.IS(a, sigma, S)
# print(SC, IS)

# lowerBound  = M - 3 * S
# upperBound = M + 3 * S
# range = upperBound - lowerBound
# nums = np.arange(lowerBound, upperBound, range / 1000)



# prior = stat.normalDF(nums, S, M)
# post600 = stat.normalDF(nums, np.sqrt(Tsqrd), A * 600 + B)
# post900 = stat.normalDF(nums, np.sqrt(Tsqrd), A * 900 + B)

# figName = 'Distribution Functions Problem 13.png'
# plt.scatter(x = nums, y= prior, color = 'gray', s= 5, label = 'Prior Distribution Function')
# plt.scatter(x = nums, y = post600, color = 'darkgray', s =5, label = 'Posterior Distribution (X = 600)')
# plt.scatter(x = nums, y = post900, color = 'lightgray', s = 5, label = 'Posterior Distribution (X = 900)')
# plt.title("Prior and Posterior Distribution Functions")
# plt.xlabel("X value")
# plt.ylabel("Cumulative Probability")
# plt.legend()
# plt.savefig(stat.getDownloadsTab() + figName)
# stat.createFigureLatex(13, figName)




# prior = stat.normaldf(nums, S, M)
# post600 = stat.normaldf(nums, np.sqrt(Tsqrd), A * 600 + B)
# post900 = stat.normaldf(nums, np.sqrt(Tsqrd), A * 900 + B)

# figName = 'Density Functions Problem 13.png'
# plt.scatter(x = nums, y= prior, color = 'gray', s= 5, label = 'Prior Density Function')
# plt.scatter(x = nums, y = post600, color = 'darkgray', s =5, label = 'Posterior Density (X = 600)')
# plt.scatter(x = nums, y = post900, color = 'lightgray', s = 5, label = 'Posterior Density (X = 900)')
# plt.title("Prior and Posterior Density Functions")
# plt.xlabel("X value")
# plt.ylabel("Probability Density")
# plt.legend()
# plt.savefig(stat.getDownloadsTab() + figName)
# stat.createFigureLatex(13, figName)


# postMean = A * nums + B
# lower50 = A * nums + B + stat.inverseStandardNormal(0.25) * np.sqrt(Tsqrd)
# upper50 = A * nums + B + stat.inverseStandardNormal(0.75)* np.sqrt(Tsqrd)
# lower98 = A * nums + B + stat.inverseStandardNormal(0.01)* np.sqrt(Tsqrd)
# upper98 = A * nums + B + stat.inverseStandardNormal(0.99)* np.sqrt(Tsqrd)

# figName = 'Problem 13 central credible intervals.png'
# plt.scatter(x = nums, y = postMean, color = 'gray', s = 5, label = 'Expected Posterior Mean')
# plt.scatter(x = nums, y = lower50, color = 'darkgray', s= 5, label = '50% Central Credible Interval')
# plt.scatter(x = nums, y = upper50, color = 'darkgray', s= 5)
# plt.scatter(x = nums, y = lower98, color = 'lightgray', s= 5, label = '98% Central Credible Interval')
# plt.scatter(x = nums, y = upper98, color = 'lightgray', s= 5)
# plt.legend()
# plt.title("Posterior Central Credible Intervals")
# plt.xlabel("X value")
# plt.ylabel("W value")
# plt.savefig(stat.getDownloadsTab() + figName)
# stat.createFigureLatex(13, figName)