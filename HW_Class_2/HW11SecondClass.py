import Stats_Functions as stat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


jointSample = pd.read_excel("C:/Users/ucg8nb/Downloads/jointSampleHW11.xlsx")

X = np.array(jointSample['X'].tolist())
W = np.array(jointSample['W'].tolist())
years = list(range(1971, 1984))
W,X,years = (list(x) for x in zip(*sorted(zip(W, X, years))))
W = np.array(W)
X = np.array(X)
print(W)
print(years)
# print(W)
# print(X)

M = 421.07
S = 162.47
a = 0.522
b = 173.26
sigma = 76.86

A, B, Tsqrd = stat.posteriorParameters(a, b, sigma, M, S)
T = np.sqrt(Tsqrd)

A = 1.052
B = 7.7
T = 109.11

# paramterNames = [r'$A$', r'$B$', r'$T^2$']
# paramValues = [A, B, Tsqrd]

# stat.printInLatexTable([paramterNames, paramValues], ['Parameter', 'Value'])


def Phi(w, x):
    return stat.standardNormalArray((w - A * x - B) / T)

pn = stat.standardNormalArray((W - M) / S)
qn = []
for w in list(W):
    curSum = 0
    count = 0
    for x in list(X):
        curSum += stat.standardNormal((w - A * x - B)/ T)
        count += 1
    qn.append(curSum / count)
qn = np.array(qn)

# print(pn)
# print(qn)

diff = np.abs(pn - qn)

# print(np.max(diff))


n = list(range(1, len(W) + 1))

colNames = ['Year', '(n)', r'$w_{(n)}$', r'$p_n$', r'$q_n$', r'$|p_n - q_n|$']
stat.printInLatexTable([years, n, W, pn, qn, diff], colNames)

# figName = 'Marginal Calibration Function.png'
# plt.scatter(x = pn, y = qn, color = 'gray')
# plt.plot([0, 1], [0,1], color = 'lightgray', linestyle  = '--')
# plt.xlim(0,1)
# plt.ylim(0,1)
# plt.title("Marginal Calibration Function")
# plt.xlabel("pn (prior probability)")
# plt.ylabel("qn (mean posterior probability)")
# plt.savefig(stat.getDownloadsTab()  + figName)
# stat.createFigureLatex(11, figName)


# U = Phi(W, X)
# plotPos = stat.metaGaussianPlottingPositions(len(U))
# Un, years = zip(*sorted(zip(U, years)))
# Un = list(Un)
# years = list(years)

# figName = 'Empirical Distribution of U.png'
# plt.scatter(x = sorted(U), y = plotPos, color = 'gray')
# plt.title("Empirical distribution of U")
# plt.xlabel("Phi(w|x)")
# plt.ylabel("Cumulative distribution")
# plt.savefig(stat.getDownloadsTab() + figName)
# stat.createFigureLatex(11, figName)



# us = np.abs(np.array(Un) - np.array(plotPos))
# print(np.max(us))



# colNames = ['Year', r'$u_{(n)}$', r'$p_n$', r'$|p_n - u_{(n)}|$']
# stat.printInLatexTable([years, Un, plotPos, np.abs(U - plotPos)], colNames)

# newFigName = 'Uniform Calibration function of the forecaster.png'
# plt.scatter(x = sorted(U), y = plotPos, color = 'gray')
# plt.plot([0,1], [0,1], color = 'lightgray', linestyle = '-')
# plt.title("Uniform Calibration Function")
# plt.xlabel("Phi(w|x)")
# plt.ylabel("Cumulative distribution")
# plt.xlim(0,1)
# plt.ylim(0,1)
# plt.savefig(stat.getDownloadsTab() + newFigName)
# stat.createFigureLatex(11, newFigName)