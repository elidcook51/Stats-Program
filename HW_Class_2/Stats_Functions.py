import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

distributions = ['NM', 'LN', 'LRNM', 'LG', 'LP', 'GB', 'RG', 'EX', 'WB', 'IW', 'LW', 'LL', 'P1', 'P2', 'LRLG', 'LRLP', 'LRGB', 'LRGB', 'LRRG', 'RLN', 'REX', 'RWB', 'RIW', 'RLW', 'RLL']
unboundedDistributions = ['NM', 'LG', 'LP', 'GB', 'RG']
lowBound = ['LN', 'EX', 'WB', 'IW', 'LW', 'LL']
upBound = ['RLN', 'REX', 'RWB', 'RIW', 'RLW', 'RLL']
bothBound = ['P1', 'P2', 'LRLG', 'LRNM', 'LRLP', 'LRGB', 'LRRG']

def getDistributions():
    return distributions

def getUnboundedDistributions():
    return unboundedDistributions

def getBoundedBelowDistributions():
    return lowBound

def getBoundedUboveDistributions():
    return upBound

def getDoubleBoundedDistributions():
    return bothBound


def standardNormal(z):
    a1 = 0.196854
    a2 = 0.115194
    a3 = 0.000344
    a4 = 0.019527
    if z >= 0:
        return 1 - (1/2) * (1 + a1 * z + a2 * z * z + a3 * z * z * z + a4 * z * z * z *z) ** -4
    else:
        return 1 - standardNormal(-z)

def standardNormalArray(nums):
    outputList = []
    for n in nums.tolist():
        outputList.append(standardNormal(n))
    return np.array(outputList)

def inverseStandardNormal(p):
    a0 = 2.30753
    a1 = 0.27061
    b1 = 0.99229
    b2 = 0.04481
    if p < 0 or p >= 1:
        return None
    if p >= 0.5:
        t = (-2 * math.log(1 - p)) ** (1/2)
        num = a0 + a1 * t
        den = 1 + b1 * t + b2 * t * t
        return t - num/den
    else:
        return -1 * inverseStandardNormal(1-p)

def normal(x, mu, sigma):
    return standardNormal((x - mu) / sigma)

def inverseNormal(p, mu, sigma):
    return sigma * inverseStandardNormal(p) + mu

def standardNormaldf(x):
    return (1 / np.sqrt(2 * math.pi)) * np.exp( -1 * (np.power(x , 2) / 2))

def standardNormaldfArray(nums):
    list = nums.tolist()
    outputList = []
    for l in list:
        outputList.append(standardNormaldf(l))
    return np.array(outputList)
def sampleMean(list):
    sum = 0.0
    count = 0.0
    for l in list:
        sum += l
        count += 1
    return sum / count

def sampleVariance(list, unbiased = False):
    m = sampleMean(list)
    sum = 0.0
    count = 0.0
    for l in list:
        sum += (l - m)**2
        count += 1
    if unbiased:
        return sum / (count - 1)
    else:
        return sum / count

def standardPlottingPositions(len):
    output = []
    for i in range(1,len+1):
        output.append(float(i) / len)
    return output

def WeibullPlottingPositions(len):
    output = []
    for i in range(1,len+1):
        output.append(i / (len + 1.0))
    return output

def metaGaussianPlottingPositions(len):
    output = []
    if len < 3:
        tn = 3.0193 * (len ** -1.1018) + 1
    elif len < 6:
        tn = 2.4035 * (len ** -0.9096) + 1
    elif len < 11:
        tn = 2.1408 * (len ** -0.8423) + 1
    elif len < 20001:
        tn = 1.9574 * (len ** -0.8039) + 1
    else:
        tn = 1
    for n in range(1,len+ 1):
        inner = (len - n + 1) / n
        inner = inner ** tn
        output.append((inner + 1) ** -1)
    return output

def LSregression(v, u, a, b):
    N = len(v)
    v = np.array(v)
    u = np.array(u)
    vBar = np.sum(v) / N
    uBar = np.sum(u) / N
    bHat = 0
    aHat = 0
    if a and b:
        bHat = (np.sum(v * u) - N * vBar * uBar) / (np.sum(u ** 2) - N * (uBar ** 2))
        aHat = vBar - bHat * uBar
    if b and not a:
        bHat = np.sum(v * u) / np.sum(u ** 2)
    if a and not b:
        aHat = vBar - uBar
    return aHat, bHat

def calcMAD(p, estimates):
    curMax = 0
    for i in range(len(p)):
        tempMax = abs(p[i] - estimates[i])
        if tempMax > curMax:
            curMax = tempMax
    return curMax

def calcKStest(estimates):
    estimates = sorted(estimates)
    N = len(estimates)
    Dlmax = 0
    DAmax = 0
    for i in range(N):
        Dltemp = abs((i-1)/N - estimates[i])
        DAtemp = abs(i/N - estimates[i])
        if Dltemp > Dlmax:
            Dlmax = Dltemp
        if DAtemp > DAmax:
            DAmax = DAtemp
    return Dlmax, DAmax, max(Dlmax, DAmax)

def normalDensity(nums,mu, sigma):
    coef = 1 / (sigma * math.sqrt(2 * math.pi))
    output = coef * np.exp( (-1/2) * np.power((nums - mu)/sigma,2))
    return output

def getDownloadsTab():
    return "C:/Users/ucg8nb\Downloads"

def getCorrelation(nqtX, nqtY):
    nqtX = np.array(nqtX)
    nqtY = np.array(nqtY)
    bothBar = np.mean(nqtX * nqtY)
    xBar = np.mean(nqtX)
    yBar = np.mean(nqtY)
    xSTD = np.std(nqtX, ddof = 1)
    ySTD = np.std(nqtY, ddof = 1)
    return (bothBar  - xBar * yBar) / (xSTD * ySTD)

def logisticDF(nums, alpha, beta):
    return np.power(1 + np.exp(-1 * (nums - beta) / alpha), -1)

def logisticdf(nums, alpha, beta):
    return (1 / alpha) * np.exp(-1 * (nums - beta) / alpha) * np.power(1 + np.exp(-1 * (nums - beta) / alpha), -2)

def laplaceDF(nums, alpha, beta):
    numbers = nums.tolist()
    outputList = []
    for n in numbers:
        if n <= beta:
            outputList.append((1/2) * np.exp((n - beta) / alpha))
        if n > beta:
            outputList.append(1 - (1/2) * np.exp(-1 * (n - beta) / alpha))
    return np.array(outputList)

def laplacedf(nums, alpha, beta):
    return (1 / (2 * alpha)) * np.exp(-1 * np.abs(nums - beta) / alpha)

def gumbelDF(nums, alpha, beta):
    return np.exp(-1 * np.exp(-1 * (nums - beta) / alpha ))

def gumbeldf(nums, alpha, beta):
    return (1/ alpha) * np.exp(-1 * (nums - beta) / alpha) * np.exp(-1 * np.exp(-1 * (nums - beta) / alpha))

def reflectedGumbelDF(nums, alpha, beta):
    return 1 - np.exp(-1 * np.exp((nums - beta) / alpha ))

def reflectedGumbeldf(nums, alpha, beta):
    return (1 / alpha) * np.exp((nums - beta) / alpha) * np.exp(-1 * np.exp((nums - beta) / alpha ))

def exponentialDF(nums, alpha, eta):
    return 1 - np.exp(-1 * (nums - eta) / alpha)

def exponentialdf(nums, alpha, eta):
    return (1 / alpha) * np.exp(-1 * (nums - eta) / alpha)

def weibullDF(nums, alpha, beta, eta):
    return 1 - np.exp(-1 * np.power((nums - eta) / alpha, beta))

def weibulldf(nums, alpha, beta, eta):
    return (beta / alpha) * np.power((nums - eta) / alpha, beta - 1) * np.exp(-1 * np.power((nums - eta) / alpha, beta))

def invertedWeibullDF(nums, alpha, beta, eta):
    return np.exp(-1 * np.power(alpha / (nums - eta) ,beta))

def invertedWeibulldf(nums, alpha, beta, eta):
    return (beta / alpha) * np.power(alpha / (nums - eta), beta + 1) * np.exp(-1 * np.power(alpha / (nums - eta), beta))

def logWeibullDF(nums, alpha, beta, eta):
    return 1 - np.exp(-1 * np.power(np.log(nums - eta + 1) / alpha, beta))

def logWeibulldf(nums, alpha, beta, eta):
    return (beta / (alpha * np.log(nums - eta + 1))) * np.power(np.log(nums - eta + 1) / alpha, beta -1) * np.exp(-1 * np.power(np.log(nums - eta + 1) / alpha, beta))

def logLogisticDF(nums, alpha, beta, eta):
    return np.power(1 + np.power((nums - eta) / alpha, -1 * beta), -1)

def logLogisticdf(nums, alpha, beta ,eta):
    return (beta / alpha) * np.power((nums - eta) / alpha, -1 * beta - 1) * np.power(1 + np.power((nums - eta) / alpha, -1 * beta), -2)

def power1DF(nums, beta, etaL, etaU):
    return np.power((nums - etaL) / (etaU - etaL), beta)

def power1df(nums, beta, etaL, etaU):
    return (beta / (etaU - etaL)) * np.power((nums - etaL) / (etaU - etaL), beta - 1)

def power2DF(nums, beta, etaL, etaU):
    return 1 - np.power((etaU - nums) / (etaU - etaL), beta)

def power2df(nums, beta, etaL, etaU):
    return (beta / (etaU - etaL)) * np.power((etaU - nums) /  (etaU - etaL), beta - 1)

def normalDF(nums, sigma, mu):
    outputList = []
    for n in nums.tolist():
        outputList.append(normal(n, mu, sigma))
    return np.array(outputList)

def normaldf(nums, sigma, mu):
    return (1 / (sigma * math.sqrt(2 * math.pi))) * np.exp((-1 / 2) * np.power((nums - mu)/sigma, 2))

def logNormalDF(nums, sigma, mu, eta):
    transformedNums = (np.log(nums - eta) - mu) / sigma
    return normalDF(transformedNums, sigma, mu)

def logNormaldf(nums, sigma, mu, eta):
    coef = 1 / ((nums - eta) * sigma * math.sqrt(2 * math.pi))
    transformedNums = (np.log(nums - eta) - mu) / sigma
    return coef * np.exp((-1 / 2) * np.power(transformedNums, 2))

def logRatioDF(nums, type, alpha, beta, etaL, etaU):
    transformedNums = np.log((nums - etaL) / (etaU - nums))
    if type == "LG":
        return logisticDF(transformedNums, alpha, beta)
    if type == "LP":
        return laplaceDF(transformedNums, alpha, beta)
    if type == "GB":
        return gumbelDF(transformedNums, alpha, beta)
    if type == "RG":
        return reflectedGumbelDF(transformedNums, alpha, beta)
    if type == 'NM':
        return normalDF(transformedNums, alpha, beta)

def logRatiodf(nums, type, alpha, beta, etaL, etaU):
    transformedNums = np.log((nums - etaL) / (etaU - nums))
    coef = (etaU - etaL) / ((nums - etaL) * (etaU - nums))
    if type == "LG":
        return coef * logisticdf(transformedNums, alpha, beta)
    if type == "LP":
        return coef * laplacedf(transformedNums, alpha, beta)
    if type == "GB":
        return coef * gumbeldf(transformedNums, alpha, beta)
    if type == "RG":
        return coef * reflectedGumbeldf(transformedNums, alpha, beta)
    if type == "NM":
        return coef * normaldf(transformedNums, alpha, beta)

def getdf(dist, nums, params):
    if dist == 'NM':
        return normaldf(nums, params[0], params[1])
    if dist == 'LG':
        return logisticdf(nums, params[0], params[1])
    if dist == 'LP':
        return laplacedf(nums, params[0], params[1])
    if dist == 'GB':
        return gumbeldf(nums, params[0], params[1])
    if dist == 'RG':
        return reflectedGumbeldf(nums, params[0], params[1])
    if dist == 'LN':
        return logNormaldf(nums, params[0], params[1], params[2])
    if dist == 'EX':
        return exponentialdf(nums, params[0], params[2])
    if dist == 'WB':
        return weibulldf(nums, params[0], params[1], params[2])
    if dist == 'IW':
        return invertedWeibulldf(nums, params[0], params[1], params[2])
    if dist == 'LW':
        return logWeibulldf(nums, params[0], params[1], params[2])
    if dist == 'LL':
        return logLogisticdf(nums, params[0], params[1], params[2])
    if dist in bothBound:
        if dist == 'P1':
            return power1df(nums, params[1], params[2], params[3])
        if dist == 'P2':
            return power2df(nums, params[1], params[2], params[3])
        return logRatiodf(nums, dist[2:], params[0], params[1], params[2], params[3])



def empiricalNQT(sample):
    plotPos = metaGaussianPlottingPositions(len(sample))
    sortedSample = sorted(sample)
    outputList = []
    for n in sample:
        index = sortedSample.index(n)
        outputList.append(plotPos[index])
    return outputList

def inverseStandardNormalArray(nums):
    outputList = []
    for p in nums.tolist():
        outputList.append(inverseStandardNormal(p))
    return np.array(outputList)

def inverseNormalArray(nums, sigma, mu):
    outputList = []
    for p in nums.tolist():
        outputList.append(inverseNormal(p, mu, sigma))
    return np.array(outputList)

def xi(gamma, z1, z2):
    return (1 / math.sqrt(1 - gamma)) * np.exp(((-1 * gamma) / (2 * (1 - gamma**2))) * (gamma * np.power(z1, 2) + z1 * z2 + gamma * np.power(z2, 2)) )

def pix(g, lam, den0, den1):
    prior = (1 - g) /g
    return np.power(1 +  prior * lam * den0 / den1, -1)

def inverseLG(p, alpha, beta):
    return beta + alpha * np.log(p / (1 - p))

def inverseLP(p, alpha, beta):
    listP = p.to_list()
    outputList = []
    for realP in listP:
        if realP <= 0.5:
            outputList.append(beta + alpha * np.log(2 * realP))
        else:
            outputList.append(beta - alpha * np.log(2 * (1 * realP)))
    return np.array(outputList)

def inverseGB(p, alpha, beta):
    return beta - alpha * np.log(-1 * np.log(p))

def inverseRG(p, alpha, beta):
    return beta + alpha * np.log(-1 * np.log(1 - p))

def inverseEX(p, alpha, eta):
    return -1 * alpha * np.log(1 - p) + eta

def inverseWB(p, alpha, beta, eta):
    return alpha * np.power(-1 * np.log(1 - p), 1/ beta) + eta

def inverseIW(p, alpha, beta ,eta):
    return alpha * np.power(-1 * np.log(p), -1 / beta) + eta

def inverseLW(p, alpha, beta, eta):
    return np.exp(alpha * np.power(-1 * np.log(1 - p), 1 / beta)) + eta - 1

def inverseLL(p, alpha, beta, eta):
    return alpha * np.power(p / (1 - p), 1/beta) + eta

def inverseP1(p, beta, etaL, etaU):
    return (etaU - etaL) * np.power(p, 1/beta) + etaL

def inverseP2(p, beta, etaL, etaU):
    return etaU - (etaU - etaL) * np.power(1 - p, 1/beta)

def inverseLR(p, type, alpha, beta, etaL, etaU):
    if type == "LG":
        inverseP = inverseLG(p, alpha, beta)
    elif type == "LP":
        inverseP = inverseLP(p, alpha, beta)
    elif type == "GB":
        inverseP = inverseGB(p, alpha, beta)
    elif type == 'NM':
        inverseP = inverseNormalArray(p, alpha, beta)
    else:
        inverseP = inverseRG(p, alpha, beta)
    return etaU - ((etaU - etaL) / (np.exp(inverseP) + 1))

def inverseLN(p, sigma, mu, eta):
    return np.exp(sigma * inverseStandardNormalArray(p) + mu) + eta

def inverseDist(dist, p, params):
    if dist == "NM":
        return inverseNormal(p, params[1], params[0])
    if dist == 'LN':
        return inverseLN(p, params[0], params[1], params[2])
    if dist == "LG":
        return inverseLG(p, params[0], params[1])
    if dist == 'LP':
        return inverseLP(p, params[0], params[1])
    if dist == 'GB':
        return inverseGB(p, params[0], params[1])
    if dist == 'RG':
        return inverseRG(p, params[0], params[1])
    if dist == 'EX':
        return inverseEX(p, params[0], params[2])
    if dist == 'WB':
        return inverseWB(p, params[0], params[1], params[2])
    if dist == 'IW':
        return inverseIW(p, params[0], params[1], params[2])
    if dist == 'LW':
        return inverseLW(p, params[0], params[1], params[2])
    if dist == 'LL':
        return inverseLL(p, params[0], params[1], params[2])
    if dist in bothBound:
        if dist == 'P1':
            return inverseP1(p, params[1], params[2], params[3])
        if dist == 'P2':
            return inverseP2(p, params[1], params[2], params[3])
        return inverseLR(p, dist[2:], params[0], params[1], params[2], params[3])

def fitRegressUnbounded(dist, x):
    x = np.array(sorted(x))
    p = np.array(metaGaussianPlottingPositions(len(x)))
    if dist == 'LG':
        v = np.array(x)
        u = np.log( p / (1 - p))
        a,b = LSregression(v, u, True, True)
        return b, a
    if dist == "LP":
        v = np.array(x)
        u = []
        for pval in p.tolist():
            if pval <= 0.5:
                u.append(np.log(2 * pval))
            else:
                u.append(-1 * np.log(2 * (1 - pval)))
        u = np.array(u)
        a,b = LSregression(v, u, True, True)
        return b, a
    if dist == 'GB':
        v = np.array(x)
        u = -1 * np.log(-1 * np.log(p))
        a,b = LSregression(v, u, True, True)
        return b, a
    if dist == 'RG':
        v = np.array(x)
        u = np.log(-1 * np.log(1 - p))
        a,b = LSregression(v, u, True, True)
        return b, a
    if dist == 'NM':
        v = np.array(x)
        u = inverseStandardNormalArray(p)
        a, b = LSregression(v, u, True, True)
        return a, b

def getUnboundedDF(dist, alpha, beta ,nums):
    if dist == 'LG':
        return logisticDF(nums, alpha, beta)
    if dist == 'LP':
        return laplaceDF(nums, alpha, beta)
    if dist == 'GB':
        return gumbelDF(nums, alpha, beta)
    if dist == 'RG':
        return reflectedGumbelDF(nums, alpha, beta)
    if dist == 'NM':
        return normalDF(nums, alpha, beta)

def fitRegressBounded(dist, x, eta):
    x = np.array(sorted(x))
    p = np.array(metaGaussianPlottingPositions(len(x)))
    if dist == 'EX':
        v = np.log(x - eta)
        u = np.log(-1 * np.log(1 - p))
        a, b = LSregression(v, u, True, False)
        return np.exp(a), 0
    if dist == 'WB':
        v = np.log(x - eta)
        u = np.log(-1 * np.log(1 - p))
        a, b = LSregression(v, u, True, True)
        return np.exp(a), 1/ b
    if dist == 'IW':
        v = np.log(x - eta)
        u = -1 * np.log(-1 * np.log(p))
        a, b = LSregression(v, u, True, True)
        return np.exp(a), 1 / b
    if dist == 'LW':
        v = np.log(np.log(x - eta + 1))
        u = np.log(-1 * np.log(1 - p))
        a,b = LSregression(v, u, True, True)
        return np.exp(a), 1 /b
    if dist == 'LL':
        v = np.log(x - eta)
        u = np.log(p / (1 - p))
        a, b = LSregression(v, u, True, True)
        return np.exp(a), 1 /b
    if dist == 'LN':
        v = np.log(x - eta)
        u = inverseStandardNormalArray(p)
        a, b = LSregression(v, u, True, True)
        return b, a

def getBoundedDF(dist, nums, alpha, beta, eta):
    if dist == 'EX':
        return exponentialDF(nums, alpha, eta)
    if dist == 'WB':
        return weibullDF(nums, alpha, beta, eta)
    if dist == 'IW':
        return invertedWeibullDF(nums, alpha, beta, eta)
    if dist == 'LW':
        return logWeibullDF(nums, alpha, beta, eta)
    if dist == "LL":
        return logLogisticDF(nums, alpha, beta, eta)
    if dist == "LN":
        return logNormalDF(nums, alpha, beta, eta)

def fitRegressDoubleBounded(dist, x, etaL, etaU):
    x = np.array(sorted(x))
    p = np.array(metaGaussianPlottingPositions(len(x)))
    v = np.log(((etaU - etaL) / (etaU - x)) - 1)
    if dist == "P1":
        v = np.log((x - etaL) / (etaU - etaL))
        u = np.log(p)
        a, b = LSregression(v, u, False, True)
        return 0, 1 / b
    if dist == 'P2':
        v = np.log((etaU - x) / (etaU - etaL))
        u = np.log(1 - p)
        a, b = LSregression(v, u, False, True)
        return 0, 1 / b
    if dist == "LRLG":
        u = np.log(p / (1 - p))
        a, b = LSregression(v, u, True, True)
        return b, a
    if dist == "LRLP":
        u = []
        for pval in p.tolist():
            if pval <= 0.5:
                u.append(np.log(2 * pval))
            else:
                u.append(-1 * np.log(2 * (1 - pval)))
        u = np.array(u)
        a, b = LSregression(v, u, True, True)
        return b, a
    if dist == "LRGB":
        u = -1 * np.log(-1 * np.log(p))
        a, b = LSregression(v, u, True, True)
        return b, a
    if dist == "LRRG":
        u = np.log(-1 * np.log(1 - p))
        a, b = LSregression(v, u, True, True)
        return b, a
    if dist == 'LRNM':
        u = inverseStandardNormalArray(p)
        a, b = LSregression(v, u, True, True)
        return b, a



def getKSTestSig(testStat, N):
    if N <= 35:
        return 'Use table'
    sqrtN = np.power(N, 0.5)
    sigs = [0.2, 0.15, 0.10, 0.05, 0.01]
    levels = [1.07 / sqrtN, 1.14 / sqrtN, 1.22 / sqrtN, 1.36 / sqrtN, 1.63 / sqrtN]
    for i in range(len(sigs)):
        if testStat < levels[i]:
            return sigs[i]
    return 0

def normalLinearRegressionEstimator(nqtZ, nqtV):
    N = len(nqtZ)
    zBar = (1/N) * np.sum(nqtZ)
    vBar = (1/N) * np.sum(nqtV)
    a = (np.sum(nqtZ * nqtV) - N * zBar * vBar)/(np.sum(nqtV * nqtV) - N * vBar * vBar)
    b = zBar - a * vBar
    theta = []
    listZ = nqtZ.tolist()
    listV = nqtV.tolist()
    for i in range(len(listZ)):
        z = listZ[i]
        v = listV[i]
        theta.append(z - v * a - b)
    theta = np.array(theta)
    sigma = np.sqrt((1/N) * np.sum(theta * theta))
    return a, b, sigma

def SC(a, sigma):
    return np.abs(a) / sigma

def IS(a, sigma):
    return np.power(np.power(SC(a, sigma), -2) + 1, -1/2)

def posteriorParameters(a, b, sigma):
    den = a ** + sigma ** 2
    A = a / den
    B = (-1 * a * b) / den
    Tsqrd = (sigma ** 2) / den
    return A, B, Tsqrd

def getMAD(data, plottingPositions, dist, params):
    data = sorted(data)
    nums = np.array(data)
    curDistance = 0
    DF = getDF(dist, nums, params)
    DF = list(DF)
    for i in range(len(data)):
        distance = abs(plottingPositions[i] - DF[i])
        if distance > curDistance:
            curDistance = distance
    return curDistance

def getDF(dist, nums,  params):
    if dist in unboundedDistributions:
        return getUnboundedDF(dist, params[0], params[1], nums)
    if dist in lowBound:
        return getBoundedDF(dist, nums, params[0], params[1], params[2])
    if dist in bothBound:
        if dist == "P1":
            return power1DF(nums, params[1], params[2], params[3])
        if dist == "P2":
            return power2DF(nums, params[1], params[2], params[3])
        return logRatioDF(nums, dist[2:], params[0], params[1], params[2], params[3])


def calcDerivatives(params, dist, data, plottingPositions):
    d = 0.001
    lowAlpha = params.copy()
    lowAlpha[0] = lowAlpha[0] - d
    highAlpha = params.copy()
    highAlpha[0] = highAlpha[0] + d
    lowAlphaMAD = getMAD(data, plottingPositions, dist, lowAlpha)
    highAlphaMAD = getMAD(data, plottingPositions, dist, highAlpha)
    dAlpha = (highAlphaMAD - lowAlphaMAD) / (2 * d)
    lowBeta = params.copy()
    lowBeta[1] -= d
    highBeta = params.copy()
    highBeta[1] = highBeta[1] + d
    lowBetaMAD = getMAD(data, plottingPositions, dist, lowBeta)
    highBetaMAD = getMAD(data, plottingPositions, dist, highBeta)
    dBeta = (highBetaMAD - lowBetaMAD) / (2 * d)
    return dAlpha, dBeta

def gradientDescent(params, dist, data, plottingPositions, numSteps):
    for i in range(numSteps):
        dAlpha, dBeta = calcDerivatives(params, dist, data, plottingPositions)
        params[0] -= dAlpha * (1 - i / (numSteps + 1))
        params[1] -= dBeta * (1 - i / (numSteps + 1))
    return params, getMAD(data, plottingPositions, dist, params)

def findUDFit(data, lowerBound, upperBound, numSteps = 100):
    plottingPositions = metaGaussianPlottingPositions(len(data))
    outputDf = pd.DataFrame()
    for dist in unboundedDistributions:
        alpha, beta = fitRegressUnbounded(dist, data)
        LSMAD = getMAD(data, plottingPositions, dist, [alpha, beta])
        params, MAD = gradientDescent([alpha, beta], dist, data, plottingPositions, numSteps)
        estimates = getDF(dist, np.array(data), params)
        KS = calcKStest(estimates)
        testStat = KS[2]
        significance = getKSTestSig(testStat, len(data))
        newDict = {
            'Dist': dist,
            'alpha': params[0],
            'beta': params[1],
            'etaL': 0,
            'etaU': 0,
            'LSalpha': alpha,
            'LSbeta': beta,
            'LSMAD': LSMAD,
            'MAD': MAD,
            'KS': testStat,
            'KS Significance': significance,
        }
        outputDf = outputDf._append(newDict, ignore_index = True)
        print(f'Completed Row for {dist}')
    for dist in lowBound:
        alpha, beta = fitRegressBounded(dist, data, lowerBound)
        LSMAD = getMAD(data, plottingPositions, dist, [alpha, beta, lowerBound])
        params, MAD = gradientDescent([alpha, beta, lowerBound], dist, data, plottingPositions, numSteps)
        estimates = getDF(dist, np.array(data), params)
        KS = calcKStest(estimates)
        testStat = KS[2]
        significance = getKSTestSig(testStat, len(data))
        newDict = {
            'Dist': dist,
            'alpha': params[0],
            'beta': params[1],
            'etaL': params[2],
            'etaU': 0,
            'LSalpha': alpha,
            'LSbeta': beta,
            'MAD': MAD,
            'LSMAD': LSMAD,
            'KS': testStat,
            'KS Significance': significance
        }
        outputDf = outputDf._append(newDict, ignore_index = True)
        print(f'Completed Row for {dist}')
    for dist in bothBound:
        alpha, beta = fitRegressDoubleBounded(dist, data, lowerBound, upperBound)
        LSMAD = getMAD(data, plottingPositions, dist, [alpha, beta, lowerBound, upperBound])
        params, MAD = gradientDescent([alpha, beta, lowerBound, upperBound], dist, data, plottingPositions, numSteps)
        estimates = getDF(dist, np.array(data), params)
        KS = calcKStest(estimates)
        testStat = KS[2]
        significance = getKSTestSig(testStat, len(data))
        newDict = {
            'Dist': dist,
            'alpha': params[0],
            'beta': params[1],
            'etaL': params[2],
            'etaU': params[3],
            'MAD': MAD,
            'LSalpha': alpha,
            'LSbeta': beta,
            'LSMAD':LSMAD,
            'KS': testStat,
            'KS Significance': significance
        }
        outputDf = outputDf._append(newDict, ignore_index = True)
        print(f'Completed Row for {dist}')
    return outputDf

def plotDistributions(distDf, nums, data):
    plottingPositions = metaGaussianPlottingPositions(len(data))
    plt.scatter(data, plottingPositions, color = 'gray')
    for dist in unboundedDistributions:
        tempDf = distDf[distDf['Dist'] == dist]
        alpha = tempDf['alpha'].values[0]
        beta = tempDf['beta'].values[0]
        DF = getDF(dist, nums, [alpha, beta])
        plt.scatter(nums, DF, color = 'lightgray', s = 5)
    plt.title('Unbounded Distributions')
    plt.ylabel("Cumulative probability")
    plt.xlabel("Value")
    plt.show()
    plt.clf()
    plt.scatter(data, plottingPositions, color='gray')
    for dist in lowBound:
        tempDf = distDf[distDf['Dist'] == dist]
        alpha = tempDf['alpha'].values[0]
        beta = tempDf['beta'].values[0]
        eta = int(tempDf['etaL'].values[0])
        DF = getDF(dist, nums, [alpha, beta, eta])
        plt.scatter(nums, DF, color = 'lightgray', s = 5)
    plt.title('Lower Bounded Distributions')
    plt.ylabel("Cumulative probability")
    plt.xlabel("Value")
    plt.show()
    plt.clf()
    plt.scatter(data, plottingPositions, color='gray')
    for dist in bothBound:
        tempDf = distDf[distDf['Dist'] == dist]
        alpha = tempDf['alpha'].values[0]
        beta = tempDf['beta'].values[0]
        etaL = int(tempDf['etaL'].values[0])
        etaU = int(tempDf['etaU'].values[0])
        DF = getDF(dist, nums, [alpha, beta, etaL, etaU])
        plt.scatter(nums, DF, color = 'lightgray', s = 5)
    plt.title('Double Bounded Distributions')
    plt.ylabel("Cumulative probability")
    plt.xlabel("Value")
    plt.show()
    plt.clf()
    allDists = unboundedDistributions + lowBound + bothBound
    for dist in allDists:
        tempDf = distDf[distDf['Dist'] == dist]
        alpha = tempDf['alpha'].values[0]
        beta = tempDf['beta'].values[0]
        etaL = int(tempDf['etaL'].values[0])
        etaU = int(tempDf['etaU'].values[0])
        MAD = tempDf['MAD'].values[0]
        if dist in unboundedDistributions:
            params = [alpha, beta]
        elif dist in lowBound:
            params = [alpha, beta, etaL]
        else:
            params = [alpha, beta, etaL, etaU]
        DF = getDF(dist, nums, params)
        plt.scatter(nums, DF, color = 'lightgray', s = 5)
        plt.scatter(data, plottingPositions, color = 'gray')
        plt.title(f"Distribution for {dist} with MAD: {MAD}")
        plt.show()
        plt.clf()

def plotDistribution(dist, params, nums, DF = True, data = None, title = "", xlabel = "", ylabel = "", savefig = ""):
    otherColor = 'gray'
    if data != None:
        plottingPositions = metaGaussianPlottingPositions(len(data))
        plt.scatter(data, plottingPositions, color = 'gray')
        otherColor = 'lightgray'
    if DF:
        DF = getDF(dist, nums, params)
        plt.scatter(nums, DF, color = otherColor, s = 5)
    else:
        df = getdf(dist, nums, params)
        plt.scatter(nums, df, color = otherColor, s = 5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if savefig != "":
        plt.savefig(savefig)
        plt.clf()
    else:
        plt.show()
        plt.clf()

def plotDistributionFromDf(distDf, dist, nums, DF = True, data = None, title = "", xlabel = "", ylabel = "", savefig = ""):
    tempDf = distDf[distDf['Dist'] == dist]
    alpha = tempDf['alpha'].values[0]
    beta = tempDf['beta'].values[0]
    etaL = int(tempDf['etaL'].values[0])
    etaU = int(tempDf['etaU'].values[0])
    if dist in unboundedDistributions:
        params = [alpha, beta]
    elif dist in lowBound:
        params = [alpha, beta, etaL]
    else:
        params = [alpha, beta, etaL, etaU]
    plotDistribution(dist, params, nums, DF, data, title, xlabel, ylabel, savefig)

def getParamsFromDf(distDf, dist):
    tempDf = distDf[distDf['Dist'] == dist]
    alpha = tempDf['alpha'].values[0]
    beta = tempDf['beta'].values[0]
    etaL = int(tempDf['etaL'].values[0])
    etaU = int(tempDf['etaU'].values[0])
    if dist in unboundedDistributions:
        return [alpha, beta]
    if dist in lowBound:
        return [alpha, beta, etaL]
    else:
        return [alpha, beta, etaL, etaU]
    
def setUpConditionalProb(doc, real, ys):
    happen = [doc[i] for i in range(len(real)) if real[i] == 1]
    nothappen = [doc[i] for i in range(len(real)) if real[i] == 0]

    f0 = []
    f1 = []
    for y in ys:
        f1.append(len([happen[i] for i in range(len(happen)) if happen[i] == y]) / len(happen))
        f0.append(len([nothappen[i] for i in range(len(nothappen)) if nothappen[i] == y]) / len(nothappen))
    return f0, f1

def varianceScoreDiscrete(etay, kappay):
    etay = np.array(etay)
    kappay = np.array(kappay)
    return np.sum(etay * (1 - etay) * kappay)

def uncertaintyScoreDiscrete(etay, kappay, g):
    VS = varianceScoreDiscrete(etay, kappay)
    return VS / (g * (1 - g))

def calibrationScore(ys, etay, kappy):
    ys = np.array(ys)
    etay = np.array(etay)
    kappay = np.array(kappay)
    return np.sqrt(np.sum(np.power(etay - ys, 2) * kappay ))
