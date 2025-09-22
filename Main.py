import Stats_Functions as stat
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import statsmodels.api as sm

runoffdf = pd.read_csv("C:/Users/ucg8nb/Downloads/Prior Runoff Data.csv")
runoffdf = runoffdf[runoffdf['Year'] != 1977]
forecastdf = pd.read_csv("C:/Users/ucg8nb/Downloads/Forecast Data.csv")
forecastdf = forecastdf[forecastdf['Year'] != 1977]

runoff = runoffdf['Runoff'].tolist()
forecast1 = forecastdf['Month 1'].tolist()
forecast2 = forecastdf['Month 2'].tolist()
forecast3 = forecastdf['Month 3'].tolist()
forecast4 = forecastdf['Month 4'].tolist()

plotPosRunoff = stat.metaGaussianPlottingPositions(len(runoff))
plotPosForecast = stat.metaGaussianPlottingPositions(len(forecast3))

'''plt.scatter(x = sorted(runoff), y= plotPosRunoff, color = 'gray')
plt.title('Empirical distribution function of runoff')
plt.ylabel('Cumulative probability density')
plt.xlabel('Actual runoff volume')
plt.savefig(stat.getDownloadsTab() + '/Empirical Prior Runoff Dist.png')
plt.clf()

plt.scatter(x = sorted(forecast), y = plotPosForecast, color = 'gray')
plt.title('Empirical distribution function of forecast runoff')
plt.ylabel('Cumulative probability density')
plt.xlabel('Forecast runoff volume')
plt.savefig(stat.getDownloadsTab() + '/Empirical Forecast Dist.png')
plt.clf()'''

nums = np.arange(310, 1200, 1)
alphaRun = 461.1577
betaRun = 1.7666
eta = 300
priorDist = stat.weibullDF(nums, 461.1577, 1.7666, 300)
# plt.scatter(x = nums, y = priorDist, s = 10, color = 'lightgray', label = 'Weibull DF')
# plt.scatter(x = sorted(runoff), y = plotPosRunoff, color = 'gray', label = 'Empirical DF')
# plt.title('Fitted Weibull Distribution Function for Runoff Volume')
# plt.xlabel("Runoff Volume")
# plt.ylabel('Cumulative Probability Density')
# plt.legend()
# plt.savefig(stat.getDownloadsTab() + '/Weibull Distribution Function.png')
# plt.clf()

priorDistdf = stat.weibulldf(nums, 461.1577, 1.7666, 300)
# plt.scatter(x = nums, y = priorDistdf, s = 10, color = 'lightgray')
# plt.title('Fitted Weibull Density Function for Runoff Volume')
# plt.xlabel('Runoff Volume')
# plt.ylabel('Probability Density')
# plt.savefig(stat.getDownloadsTab() + '/Weibull density function.png')
# plt.clf()

alphaFore3 = 6.1630
betaFore3 = 16.0496
forecastedDF = stat.logWeibullDF(nums, 6.1630, 16.0496, 300)
# plt.scatter(x = nums, y =forecastedDF, color = 'lightgray', s = 10, label = 'Log-Weibull Distribution')
# plt.scatter(x = sorted(forecast), y = plotPosForecast, color = 'gray', label = 'Empirical Distribution')
# plt.title('Log-Weibull Distribution Function for Forecasted Runoff Volume')
# plt.xlabel("Forecasted Runoff Volume")
# plt.ylabel('Cumulative Probability Density')
# plt.legend()
# plt.savefig(stat.getDownloadsTab() + '/Log Weibull Distribution Function.png')
# plt.clf()

forecasteddf = stat.logWeibulldf(nums, 6.1630, 16.0496, 300)
# plt.scatter(x = nums, y = forecasteddf, color = 'lightgray')
# plt.title('Log-Weibull Density Function for Forecasted Runoff Volume')
# plt.xlabel('Forecated Runoff Volume')
# plt.ylabel('Probability Density')
# plt.savefig(stat.getDownloadsTab() + '/Log Weibull Density Function.png')
# plt.clf()


# print(a, b, sigma)

# plt.scatter(x  = forecastJoint, y = runoffJoint, color = 'gray', label = 'Joint Sample')
# plt.scatter(x = nums, y = nums * a + b, s = 10, color = 'lightgray', label = 'Median Regression')
# plt.title('Linear Regression on Scatterplot of Joint Sample')
# plt.xlabel('Transformed Value of Forecasted Runoff')
# plt.ylabel('Transformed Value of Actual Runoff')
# plt.legend()
# plt.savefig(stat.getDownloadsTab() + '/Linear Regression vs Scatter Plot.png')

# residuals = nqtZ3 - nqtV * a - b
# residuals = residuals.tolist()
# residNums = np.arange(-1,1,0.001)
# parametResid = stat.normalDF(residNums, 0, 0.4011)
# plotPosResid = stat.metaGaussianPlottingPositions(len(residuals))
# plt.scatter(x = residNums, y = parametResid, color = 'lightgray', s = 10, label = 'Parametric Reisdual Distribution')
# plt.scatter(x = sorted(residuals), y = plotPosResid, color = 'gray', label = 'Empirical Residaul Distribution')
# plt.title('Parametric Residual Distribution plotted against Empirical Residuals')
# plt.xlabel('Residual')
# plt.ylabel("Cumulative Probability Desnsity")
# plt.legend()
# plt.savefig(stat.getDownloadsTab() + '/Residuals distribution.png')
# plt.clf()

# print(np.sqrt(stat.sampleVariance(nqtV.tolist())))

# A = a / (a ** 2 + sigma ** 2)
# B = (-1 * a * b) / (a ** 2 + sigma ** 2)
# Tsqrd = (sigma ** 2) / (a ** 2 + sigma ** 2)
# print(A, B, Tsqrd)

# print(stat.calcKStest(sorted(residuals)))
# estimates = stat.normalDF(np.array(residuals), 0, sigma)
# print(stat.calcKStest(sorted(estimates)))
# print(len(residuals))

# plt.scatter(x = residuals, y = nqtV, color = 'gray')
# plt.title('Residuals versus transformed realizations')
# plt.ylabel('Transformed realizations')
# plt.xlabel('Residuals')
# plt.axhline(y = 0, color = 'lightgray', linestyle = '--')
# plt.savefig(stat.getDownloadsTab() + '/Residuals versus transformed realizations')

# lower = np.mean(forecastJoint3) - 3 * np.std(forecastJoint3)
# upper = np.mean(forecastJoint3) + 3 * np.std(forecastJoint3)
# nums = np.arange(max(lower, 301), upper, 0.1)
#
# def quantileFunction(nums, p):
#     interior = A * stat.inverseStandardNormalArray(stat.logWeibullDF(nums, 6.1630, 16.0496, 300)) + B + np.sqrt(Tsqrd) * stat.inverseStandardNormal(p)
#     return stat.inverseWB(stat.standardNormalArray(interior), 461.1577, 1.7666, 300)

# median = quantileFunction(nums, 0.5)
# low50 = quantileFunction(nums, 0.25)
# high50 = quantileFunction(nums, 0.75)
# low90 = quantileFunction(nums, 0.05)
# high90 = quantileFunction(nums, 0.95)
#
# plt.scatter(x = nums, y = median, color = 'gray', label = 'Posterior Median', s = 10)
# plt.scatter(x = nums, y = low50, color = 'darkgray',s = 10, label = '50% Posterior Credible Confidence Interval')
# plt.scatter(x = nums, y = high50, color = 'darkgray', s = 10)
# plt.scatter(x = nums, y = low90, color = 'lightgray',s = 10, label = '90% Posterior Credible Confidence Interval')
# plt.scatter(x = nums, y = high90, color = 'lightgray', s = 10)
# plt.title("Posterior Credible Confidence Intervals")
# plt.xlabel('Forecast Value')
# plt.ylabel('Runoff Value')
# plt.legend()
# plt.savefig(stat.getDownloadsTab() + '/Posterior credible confidence intervals.png')

# print(quantileFunction(np.array([543]), 0.05))
# M = np.mean(runoff)
# x1 = M - 100
# x2 = M + 100
#
# def posteriorDensity(nums, x):
#     T = np.sqrt(Tsqrd)
#     x = np.array([x])
#     coef = T * stat.standardNormaldfArray(stat.inverseStandardNormalArray(stat.logWeibullDF(x, 6.1630, 16.0496, 300)))
#     coef = 1 / coef
#     numPart1 = stat.inverseStandardNormalArray(stat.weibullDF(nums, 461.1577, 1.7666, 300))
#     numPart2 = A * stat.inverseStandardNormalArray(stat.logWeibullDF(x, 6.1630, 16.0496, 300))
#     return coef * stat.standardNormaldfArray((numPart1 - numPart2  - B) / T)
#
# def posteriorDistribution(nums, x):
#     T = np.sqrt(Tsqrd)
#     x = np.array([x])
#     numPart1 = stat.inverseStandardNormalArray(stat.weibullDF(nums, 461.1577, 1.7666, 300))
#     numPart2 = A * stat.inverseStandardNormalArray(stat.logWeibullDF(x, 6.1630, 16.0496, 300))
#     return stat.standardNormalArray((numPart1 - numPart2 - B) / T)
#
# prior = stat.weibullDF(nums, 461.1577, 1.7666, 300)
# postx1 = posteriorDistribution(nums, x1)
# postx2 = posteriorDistribution(nums, x2)
#
# plt.scatter(x  = nums, y = prior, color = 'gray', label = 'Prior Distribution', s = 5)
# plt.scatter(x  = nums, y = postx1, color = 'lightgray', label = 'Posterior Distribution for x1', s = 5)
# plt.scatter(x  = nums, y = postx2, color = 'darkgray', label = 'Posterior Distribution for x2', s = 5)
# plt.legend()
# plt.title('Prior distribution function versus posterior distribution functions')
# plt.xlabel('Runoff Volume')
# plt.ylabel("Cumulative Probability Density")
# plt.savefig(stat.getDownloadsTab() + '/Prior versus posterior distribution functions.png')


# prior = stat.weibulldf(nums, 461.1577, 1.7666, 300)
# postx1 = posteriorDensity(nums, x1) / 1000
# postx2 = posteriorDensity(nums, x2) / 1000
#
# plt.scatter(x = nums, y = prior, color = 'gray', s = 5, label = 'Prior Density function')
# plt.scatter(x = nums, y = postx1, color = 'lightgray', s = 5, label = 'Posterior with x1')
# plt.scatter(x = nums, y = postx2, color = 'darkgray', s = 5, label = 'Posterior with x2')
# plt.legend()
# plt.title("Prior density function compared with posterior density functions")
# plt.xlabel("Runoff volume")
# plt.ylabel("Probability density")
# plt.savefig(stat.getDownloadsTab() + '/Prior density versus posterior density.png')

#Finding the distributions for the three other forecast months

# plt.scatter(x = sorted(forecast1), y = plotPosForecast, color = 'gray')
# plt.title('Empirical Distribution for Forecasts for Month 1')
# plt.xlabel('Forecast Realization')
# plt.ylabel('Cumulative Probability Density')
# plt.savefig(stat.getDownloadsTab() + '/Empirical Month 1.png')
# plt.clf()
#
# plt.scatter(x = sorted(forecast2), y = plotPosForecast, color = 'gray')
# plt.title('Empirical Distribution for Forecasts for Month 2')
# plt.xlabel('Forecast Realization')
# plt.ylabel('Cumulative Probability Density')
# plt.savefig(stat.getDownloadsTab() + '/Empirical Month 2.png')
# plt.clf()
#
# plt.scatter(x = sorted(forecast4), y = plotPosForecast, color = 'gray')
# plt.title('Empirical Distribution for Forecasts for Month 4')
# plt.xlabel('Forecast Realization')
# plt.ylabel('Cumulative Probability Density')
# plt.savefig(stat.getDownloadsTab() + '/Empirical Month 4.png')
# plt.clf()

alphaFore1 = 6.0512
betaFore1 = 17.0974

alphaFore2 = 6.1543
betaFore2 = 15.2352

alphaFore4 = 508.1820
betaFore4 = 2.1988

month1DF = stat.logWeibullDF(nums, alphaFore1, betaFore1, eta)
month1df = stat.logWeibulldf(nums, alphaFore1, betaFore1, eta)

month2DF = stat.logWeibullDF(nums, alphaFore2, betaFore2, eta)
month2df = stat.logWeibulldf(nums, alphaFore2, betaFore2, eta)

month4DF= stat.weibullDF(nums, alphaFore4, betaFore4, eta)
month4df = stat.weibulldf(nums, alphaFore4, betaFore4, eta)

runoffJoint = runoffdf[runoffdf['Year'] > 1970]['Runoff'].tolist()
runoffJoint = np.array(runoffJoint)
forecastJoint1 = np.array(forecast1)
forecastJoint2 = np.array(forecast2)
forecastJoint3 = np.array(forecast3)
forecastJoint4 = np.array(forecast4)

nqtV = stat.inverseStandardNormalArray(stat.weibullDF(runoffJoint, 461.1577, 1.7666, 300))
nqtZ1 = stat.inverseStandardNormalArray(stat.logWeibullDF(forecastJoint1, alphaFore1, betaFore1, eta))
nqtZ2 = stat.inverseStandardNormalArray(stat.logWeibullDF(forecastJoint2, alphaFore2, betaFore2, eta))
nqtZ3 = stat.inverseStandardNormalArray(stat.logWeibullDF(forecastJoint3, 6.1630, 16.0496, 300))
nqtZ4 = stat.inverseStandardNormalArray(stat.weibullDF(forecastJoint4, alphaFore4, betaFore4, eta))

a1, b1, sigma1 = stat.normalLinearRegressionEstimator(nqtZ1, nqtV)
a2, b2, sigma2 = stat.normalLinearRegressionEstimator(nqtZ2, nqtV)
a3, b3, sigma3 = stat.normalLinearRegressionEstimator(nqtZ3, nqtV)
a4, b4, sigma4 = stat.normalLinearRegressionEstimator(nqtZ4, nqtV)

informative1 = stat.IS(a1, sigma1)
informative2 = stat.IS(a2, sigma2)
informative3 = stat.IS(a3, sigma3)
informative4 = stat.IS(a4, sigma4)

#onesCol = np.ones(nqtZ1.shape)
A = np.column_stack(( nqtZ2, nqtZ3, nqtZ4))
A = sm.add_constant(A)
b = np.column_stack((nqtV))
# print(A)
# print(b)

model = sm.OLS(b.T, A).fit()

betas = model.params
#betas = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), b.T)
predictions = A @ betas
residuals = b.T - predictions
modelSigma = np.std(residuals)

predictions = np.array(predictions.tolist())
a, b, sigma = stat.normalLinearRegressionEstimator(predictions, nqtV)
# print(a, b, sigma)
# print(stat.IS(a, sigma))
# print(stat.posteriorParameters(a, b, sigma))
A, B, Tsqrd = stat.posteriorParameters(a, b, sigma)
T = np.sqrt(Tsqrd)

# predictors = [ nqtZ2, nqtZ3, nqtZ4]
# for i in range(len(predictors)):
#     pred = predictors[i]
#     beta = betas[i + 1]
#     std = np.std(pred)
#     SE = np.sqrt((modelSigma ** 2) / std)
#     tval = beta / SE
#     print(f"Beta {i + 1}: {beta}, t-val: {tval}, DF = {len(pred) - 1}")



# print(f"January - a: {a1}, b: {b1}, sigma: {sigma1}, IS: {informative1}")
# print(f"February - a: {a2}, b: {b2}, sigma: {sigma2}, IS: {informative2}")
# print(f"March - a: {a3}, b: {b3}, sigma: {sigma3}, IS: {informative3}")
# print(f"April - a: {a4}, b: {b4}, sigma: {sigma4}, IS: {informative4}")

# plt.scatter(x = nums, y =month1DF, color = 'lightgray', s = 5)
# plt.scatter(x = sorted(forecast1), y = plotPosForecast, color = 'gray')
# plt.title('Parametric versus empirical distribution for January')
# plt.xlabel('Forecast in January')
# plt.ylabel('Cumulative probability density')
# plt.savefig(stat.getDownloadsTab() + '/January Distribution Function.png')
# plt.clf()
#
# plt.scatter(x = nums, y =month2DF, color = 'lightgray', s = 5)
# plt.scatter(x = sorted(forecast2), y = plotPosForecast, color = 'gray')
# plt.title('Parametric versus empirical distribution for February')
# plt.xlabel('Forecast in February')
# plt.ylabel('Cumulative probability density')
# plt.savefig(stat.getDownloadsTab() + '/February Distribution Function.png')
# plt.clf()
#
# plt.scatter(x = nums, y =month4DF, color = 'lightgray', s = 5)
# plt.scatter(x = sorted(forecast4), y = plotPosForecast, color = 'gray')
# plt.title('Parametric versus empirical distribution for April')
# plt.xlabel('Forecast in April')
# plt.ylabel('Cumulative probability density')
# plt.savefig(stat.getDownloadsTab() + '/April Distribution Function.png')
# plt.clf()

def getZ(x2, x3, x4):
    x2 = np.array([x2])
    x3 = np.array([x3])
    x4 = np.array([x4])
    beta0 = betas[0]
    beta1 = betas[1]
    beta2 = betas[2]
    beta3 = betas[3]
    z1 = stat.inverseStandardNormalArray(stat.logWeibullDF(x2, alphaFore2, betaFore2, eta))
    z2 = stat.inverseStandardNormalArray(stat.logWeibullDF(x3, alphaFore3, betaFore3, eta))
    z3 = stat.inverseStandardNormalArray(stat.weibullDF(x4, alphaFore4, betaFore4, eta))
    return beta0 + beta1 * z1 + beta2 * z2 + beta3 * z3

def getJacob(x2, x3, x4):
    x2 = np.array([x2])
    x3 = np.array([x3])
    x4 = np.array([x4])
    beta0 = betas[0]
    beta1 = betas[1]
    beta2 = betas[2]
    beta3 = betas[3]
    z1 = stat.inverseStandardNormalArray(stat.logWeibullDF(x2, alphaFore2, betaFore2, eta))
    z2 = stat.inverseStandardNormalArray(stat.logWeibullDF(x3, alphaFore3, betaFore3, eta))
    z3 = stat.inverseStandardNormalArray(stat.weibullDF(x4, alphaFore4, betaFore4, eta))
    return stat.standardNormaldf(beta1 * z1) + stat.standardNormaldf(beta2 * z2) + stat.standardNormaldf(beta3 * z3)

def densityFunction(nums, x2, x3, x4):
    z = getZ(x2, x3, x4)
    v = stat.inverseStandardNormalArray(stat.weibullDF(nums, alphaRun, betaRun, eta))
    jacob = getJacob(x2, x3, x4)
    return (1 / (T * jacob)) * stat.standardNormaldfArray((v - A * z - B) / T)

def distributionFunction(nums, x2, x3, x4):
    z = getZ(x2, x3, x4)
    v = stat.inverseStandardNormalArray(stat.weibullDF(nums, alphaRun, betaRun, eta))
    return stat.standardNormalArray((v - A * z - B) / T)

def quantileFunction(p, x2, x3, x4):
    z = getZ(x2, x3, x4)
    p = np.array([p])
    return stat.inverseWB(stat.standardNormal(B + A * z + T * stat.inverseStandardNormalArray(p)), alphaRun, betaRun, eta)

actual1974 = 1080.9
actual1978 = 730.5

print(stat.weibullDF(actual1974, alphaRun, betaRun, eta))
print(distributionFunction(np.array([actual1974]), 905.8, 816.8, 956.3))

print(stat.weibullDF(actual1978, alphaRun, betaRun, eta))
print(distributionFunction(np.array([actual1978]), 803.5, 836.7, 750.4))

# year1974X = [905.8, 816.8, 956.3]
# year1977X = [301, 301, 301]
# year1978X = [803.5, 836.7, 750.4]
# actual1974 = 1080.9
# actual1978 = 730.5
#
# probs = [0.1, 0.25, 0.5, 0.75, 0.9]
# outputs = []
# for p in probs:
#     outputs.append(quantileFunction(p, 803.5, 836.7, 750.4))
# # outputString = ""
# # for i in range(len(outputs)):
# #     outputString += f"{probs[i]}: {outputs[i]}, "
# for i in range(len(outputs)):
#     o = outputs[i]
#     p = probs[i]
#     plt.plot([0,10], [o,o], label = "Nonexceedance of " + str(p))
# plt.plot([0,10], [actual1978, actual1978], label = 'Actual Runoff Volume')
# plt.legend()
# plt.ylim(500,1200)
# plt.xlim(1,9)
# plt.xticks([])
# plt.title("Box Plot of 1978 quantiles")
# plt.savefig(stat.getDownloadsTab() + '/1978 Quantiles with real.png')
# plt.clf()
#
# outputs = []
# for p in probs:
#     outputs.append(quantileFunction(p, 905.8, 816.8, 956.3))
# # outputString = ""
# # for i in range(len(outputs)):
# #     outputString += f"{probs[i]}: {outputs[i]}, "
# for i in range(len(outputs)):
#     o = outputs[i]
#     p = probs[i]
#     plt.plot([0,10], [o,o], label = "Nonexceedance of " + str(p))
# plt.plot([0,10], [actual1974, actual1974], label = 'Actual Runoff Volume')
# plt.legend()
# plt.ylim(500,1200)
# plt.xlim(1,9)
# plt.xticks([])
# plt.title("Box Plot of 1974 quantiles")
# plt.savefig(stat.getDownloadsTab() + '/1974 Quantiles with real.png')

# year1974 = distributionFunction(nums, year1974X[0],year1974X[1],year1974X[2])
# year1977 = distributionFunction(nums, year1977X[0],year1977X[1],year1977X[2])
# year1978 = distributionFunction(nums, year1978X[0],year1978X[1],year1978X[2])
#
# plt.scatter(x = nums, y = priorDist, color = 'gray', label = 'Prior Distribution Function')
# plt.scatter(x = nums, y = year1974, color = 'lightgray', label = '1974 Distribution Function')
# plt.scatter(x = nums, y = year1977, color = 'silver', label = '1977 Distribution Function')
# plt.scatter(x = nums, y = year1978, color = 'darkgray', label = '1978 Distribution Function')
# plt.legend()
# plt.title("Posterior Distribution Functions")
# plt.xlabel('Runoff Volume')
# plt.ylabel('Cumulative Probability Density')
# plt.savefig(stat.getDownloadsTab() + '/Posterior Distribution.png')

# year1974 = densityFunction(nums, year1974X[0],year1974X[1],year1974X[2])/ 500
# year1977 = densityFunction(nums, year1977X[0],year1977X[1],year1977X[2]) / 500
# year1978 = densityFunction(nums, year1978X[0],year1978X[1],year1978X[2]) / 500
#
# plt.scatter(x = nums, y = priorDistdf, color = 'gray', label = 'Prior Density Function')
# plt.scatter(x = nums, y = year1974, color = 'lightgray', label = '1974 Density Function')
# plt.scatter(x = nums, y = year1977, color = 'silver', label = '1977 Density Function')
# plt.scatter(x = nums, y = year1978, color = 'darkgray', label = '1978 Density Function')
# plt.legend()
# plt.title("Posterior density functions")
# plt.xlabel('Runoff Volume')
# plt.ylabel('Probability Density')
# plt.savefig(stat.getDownloadsTab() + '/Posterior density.png')