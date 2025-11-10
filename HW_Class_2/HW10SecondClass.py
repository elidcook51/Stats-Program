import Stats_Functions as stat
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# SC = np.arange(0,1,0.001)
# T = np.sqrt(np.power(np.power(SC,2) + 1 / 36 , -1) )

# figName = '/T vs SC.png'

# plt.scatter(x = T, y = SC, color = 'gray', s =10)
# plt.title('T vs SC for S value of 6')
# plt.xlabel("T (posterior standard deviation)")
# plt.ylabel("SC (sufficiency characteristic)")
# plt.savefig(stat.getDownloadsTab() + figName)

# stat.createFigureLatex(10, figName)

priorSampleSheet = pd.read_excel("C:/Users/ucg8nb/Downloads/SYS5581WaterRunoffHW10.xlsx", sheet_name = 0)
priorSampleSheet = priorSampleSheet[priorSampleSheet['Year'] != 1977]
jointSampleSheet = pd.read_excel("C:/Users/ucg8nb/Downloads/SYS5581WaterRunoffHW10.xlsx", sheet_name = 1)
jointSampleSheet = jointSampleSheet[jointSampleSheet['Year'] != 1977]

prior = priorSampleSheet['Runoff'].tolist()
jointReal = priorSampleSheet[priorSampleSheet['Year'] > 1970]['Runoff'].tolist()
months = ['January', 'February', 'March', 'April', 'May']
jointPred = [jointSampleSheet[m].tolist() for m in months]
jointPredApril = jointSampleSheet['April'].tolist()

prior = sorted(prior)


# jointReal, jointPred = (np.array(t) for t in zip(*sorted(zip(jointReal, ))))

N = len(jointReal)
M = (1 / N) * np.sum(jointReal)
S = np.sqrt((1 / N) * np.sum((jointReal - M) ** 2))

nmparams = [S, M]

likelihoodparams = [stat.normalLinearRegressionEstimator(np.array(forecast), np.array(jointReal)) for forecast in jointPred]

colNames = ['Likelihood Parameter'] + months

# stat.printInLatexTable([['a', 'b', '$\sigma$']] + likelihoodparams, colNames )

SC = [stat.SC(params[0], params[2]) for params in likelihoodparams]
postSTD = [np.sqrt(postParams[2]) for postParams in [stat.posteriorParamters(params[0], params[1], params[2], M, S) for params in likelihoodparams] ]
IS = [stat.IS(params[0], params[2], S) for params in likelihoodparams]
leadTime = [7 - i for i in range(len(IS))]

informativessFig = 'Informativeness vs lead time.png'
plt.scatter(x = leadTime, y = IS, color = 'gray')
plt.title("Informativeness vs. lead time")
plt.ylabel("Informativeness (0-1)")
plt.xlabel("Lead Time (months)")
plt.savefig(stat.getDownloadsTab() + informativessFig)
stat.createFigureLatex(10, informativessFig, 0.65)
# stat.printInLatexTable([months, SC, postSTD, IS], ['Month', 'SC', 'Posterior Standard Deviation','IS'])

# print(M, S)

# print(len(jointPred))
# print(len(jointReal))

# priorEmp = stat.metaGaussianPlottingPositions(len(prior))

# normalFit = stat.findUDFitDist('NM', prior, numSteps = 20000)
# alpha = normalFit['alpha']
# beta = normalFit['beta']






# MAD = normalFit['MAD']

# print(alpha, beta, MAD)

# MAD = stat.getMAD(jointReal, stat.metaGaussianPlottingPositions(len(jointReal)), 'NM', nmparams)
# # print(MAD)
# nums = np.arange(170, 900, 0.01)
# nmDF = stat.getDF('NM', nums, nmparams)




# a, b, sigma = stat.normalLinearRegressionEstimator(jointPred, jointReal)

# thetaDist = stat.getDF('NM', np.arange(-150, 150, 0.01), [sigma, 0])
# residual = jointPred - a * jointReal - b

# thetaMAD = stat.getMAD(residual, stat.metaGaussianPlottingPositions(len(residual)), 'NM', [sigma, 0])
# print(thetaMAD)

# check4 = 'check4.png'
# plt.scatter(x = jointReal, y = residual, color = 'gray')
# plt.hlines(y = 0, xmin = 150, xmax = 700, color = 'lightgray', linestyle = '--')
# plt.xlim(150, 700)
# plt.title("Residuals vs. Fitted")
# plt.xlabel("w(n)")
# plt.ylabel('Theta(n)')
# plt.savefig(stat.getDownloadsTab() + check4)
# stat.createFigureLatex(10, check4)

# residName = 'residuals.png'
# plt.scatter(x = sorted(residual), y = stat.metaGaussianPlottingPositions(len(residual)), color = 'gray')
# plt.scatter(x = np.arange(-150, 150, 0.01), y = thetaDist, color = 'lightgray', s= 5)
# plt.title("Paramteric vs. Empirical distribution function of residual")
# plt.savefig(stat.getDownloadsTab() + residName)
# stat.createFigureLatex(10, residName)

# SC = stat.SC(a, sigma)
# IS = stat.IS(a, sigma, S)
# # print(SC, IS)

# # print(a, b, sigma)

# linRegression = a * nums + b
# linRegFigName = '/Linear Regression figure name.png'
# plt.scatter(x = jointReal, y = jointPred, color = 'gray')
# plt.scatter(x = nums, y = linRegression, color = 'lightgray', s= 5)
# plt.xlabel("Predicted snowfall runoff volume")
# plt.ylabel("NQT of snowfall runoff volume")
# plt.title("Linear Regression comparison to joint sample")
# plt.savefig(stat.getDownloadsTab() + linRegFigName)
# stat.createFigureLatex(10, linRegFigName)

# A, B, Tsqrd = stat.posteriorParamters(a, b, sigma, M, S)
# T = np.sqrt(Tsqrd)

# meanX = np.mean(jointPred)
# stdX = np.std(jointPred)

# x1 = M - 100
# x2 = M + 100

# x1params = [T, A * x1 + B]
# x2params = [T, A * x2 + B]

# print(x1params)
# print(x2params)

# g = stat.getdf('NM', nums, nmparams)
# g1 = stat.getdf('NM', nums, x1params)
# g2 = stat.getdf("NM", nums, x2params)

# densityComp = 'Density Function Comparison.png'
# plt.scatter(x = nums, y = g, color = 'gray', label = 'Prior Density Function')
# plt.scatter(x = nums, y = g1, color = 'darkgray', label = 'Posterior with x1')
# plt.scatter(x = nums, y = g2, color = 'lightgray', label = 'Posterior with x2')
# plt.legend()
# plt.xlabel("Snowmelt Runoff")
# plt.ylabel("Probability Density")
# plt.title("Prior and Posterior Density Functions")
# plt.savefig(stat.getDownloadsTab() + densityComp)
# stat.createFigureLatex(10, densityComp)

# G = stat.getDF('NM', nums, nmparams)
# G1 = stat.getDF('NM', nums, x1params)
# G2 = stat.getDF("NM", nums, x2params)

# distributionComp = 'Distribution Function Comparison.png'
# plt.scatter(x = nums, y = G, color = 'gray', label = 'Prior Distribution Function')
# plt.scatter(x = nums, y = G1, color = 'darkgray', label = 'Posterior with x1')
# plt.scatter(x = nums, y = G2, color = 'lightgray', label = 'Posterior with x2')
# plt.legend()
# plt.xlabel("Snowmelt Runoff")
# plt.ylabel("Cumulative Probability")
# plt.title("Prior and Posterior Distribution Functions")
# plt.savefig(stat.getDownloadsTab() + distributionComp)
# stat.createFigureLatex(10, distributionComp)


# x = 543
# probs = [0.05, 0.5, 0.95]
# ws = [A * 543 + B + T * stat.inverseStandardNormal(p) for p in probs]
# colNames = ['Probability', 'Posterior Qunatiles']
# stat.printInLatexTable([probs, ws], colNames)


# print(A, B, T ** 2)

# xvals = np.arange(meanX - 3 * stdX, meanX + 3 * stdX, 0.01)

# wmedain = A * xvals + B + T * stat.inverseStandardNormal(0.5)
# w25 = A * xvals + B + T * stat.inverseStandardNormal(0.25)
# w75 = A * xvals + B + T * stat.inverseStandardNormal(0.75)
# w5 = A * xvals + B + T * stat.inverseStandardNormal(0.05)
# w95 = A * xvals + B + T * stat.inverseStandardNormal(0.95)

# centralCredible = 'Posterior Central Credible Intervals.png'
# plt.scatter(x = xvals, y = wmedain, color = 'gray', s= 5, label = 'Expected Value of X')
# plt.scatter(x = xvals, y = w25, color = 'darkgray', s= 5, label = '50% Posterior Central Credible Interval')
# plt.scatter(x = xvals, y = w5, color = 'lightgray', s= 5, label = '90% Posterior Central Credible Interval')
# plt.scatter(x = xvals, y = w95, color = 'lightgray', s= 5)
# plt.scatter(x = xvals, y = w75, color = 'darkgray', s= 5)
# plt.xlabel("Forecast Runoff Volume")
# plt.legend()
# plt.ylabel("Posterior Qunatile Runoff Volume")
# plt.title("Posterior Central Credible Intervals")
# plt.savefig(stat.getDownloadsTab() + centralCredible)
# stat.createFigureLatex(10, centralCredible)


# print(A, B, Tsqrd)

# print(a, b, sigma)



# figName = '/Normal Prior Runoff.png'
# plt.scatter(x = nums, y =nmDF, color = 'lightgray', s = 5)
# plt.scatter(x = prior, y = priorEmp, color = 'gray')
# plt.title("Empirical vs. Parametric Distribution of Runoff Volume")
# plt.xlabel("Snowmelt runoff volume")
# plt.ylabel("Cumulative Probability")
# plt.savefig(stat.getDownloadsTab() + figName)
# stat.createFigureLatex(10, figName)

# empiricalName = '/Empirical Distribution of prior runoff no outlier.png'
# plt.scatter(x = prior, y = priorEmp, color = 'gray')
# plt.title('Empirical Distribution of Prior Runoff')
# plt.xlabel("Snowmelt Runoff")
# plt.ylabel("Cumulative Probability")
# plt.savefig(stat.getDownloadsTab() + empiricalName)
# stat.createFigureLatex(10, empiricalName)

# lowerBound = 170
# upperBound = 700

# priorFit = stat.findUDFit(prior, 170, 700, numSteps = 10000)
# priorFit.to_csv(stat.getDownloadsTab() + '/Prior Fit.csv')

# nums = np.arange(lowerBound + 0.02, upperBound - 0.02, 0.01)

# lgalpha = 0.760221266
# lgbeta = -0.038334574
# lgparams = [lgalpha, lgbeta, lowerBound, upperBound]

# nmalpha = 1.258188248
# nmbeta = -0.032991933
# nmparams = [nmalpha, nmbeta, lowerBound, upperBound]

# lgDF = stat.getDF('LRLG', nums, lgparams)
# nmDF = stat.getDF("LRNM", nums, nmparams)

# figName = '/Compare Dist.png'
# plt.scatter(x = prior, y = priorEmp, color = 'gray')
# plt.scatter(x = nums, y = lgDF, color = 'lightgray', s = 3, label = 'Log Ratio Logistic')
# plt.scatter(x = nums, y = nmDF, color = 'darkgray', s = 3, label = 'Log Ratio Normal')
# plt.legend()
# plt.title("Empirical vs. Parametric Distributions")
# plt.xlabel("Snowmelt Runoff Volumne")
# plt.ylabel("Cumulative Probability")
# plt.savefig(stat.getDownloadsTab() + figName)
# stat.createFigureLatex(10, figName)

# nqtZprior = stat.inverseStandardNormalArray(stat.getDF('LRLG', np.array(prior) , lgparams))

# nqtZ = stat.inverseStandardNormalArray(stat.getDF('LRLG', np.array(jointReal) , lgparams))
# X = np.array(jointPred)

# a, b, sigma = stat.normalLinearRegressionEstimator(X, nqtZ)

# print(a, b, sigma)

# nqtNums = stat.inverseStandardNormalArray(stat.getDF('LRLG', nums, lgparams))
# Xestimate = a * nqtZ + b

# A, B, Tsqrd = stat.posteriorParameters(a, b, sigma)

# print(A, B, Tsqrd)

# linRegFigName = '/Linear Regression figure name.png'
# plt.scatter(x = X, y = nqtZ, color = 'gray')
# plt.scatter(x = a * nqtNums + b, y = nqtNums, color = 'lightgray', s = 5)
# plt.xlabel("Predicted snowfall runoff volume")
# plt.ylabel("NQT of snowfall runoff volume")
# plt.title("Linear Regression comparison to joint sample")
# plt.savefig(stat.getDownloadsTab() + linRegFigName)
# stat.createFigureLatex(10, linRegFigName)


# stat.printInLatexTable([prior, nqtZprior], ['Snowmelt Runoff', '$Z$ value'])
