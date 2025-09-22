import Stats_Functions as stat
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

df = pd.read_csv("C:/Users/ucg8nb/Downloads/Table 5.11.csv")
passDf = df[df['Passed'] == 1]
failDf = df[df['Passed'] == 0]

passRead = passDf['R'].tolist()
passWrite = passDf['W'].tolist()
passListen = passDf['L'].tolist()
passSpeak = passDf['S'].tolist()
passScore = passDf['SP'].tolist()

failRead = failDf['R'].tolist()
failWrite = failDf['W'].tolist()
failListen = failDf['L'].tolist()
failSpeak = failDf['S'].tolist()
failScore = failDf['SP'].tolist()


'''pd.DataFrame({passRead[0]: passRead[1:]}).set_index(passRead[0]).to_csv("C:/Users/ucg8nb/Downloads/passRead.csv")
pd.DataFrame({passWrite[0]: passWrite[1:]}).set_index(passWrite[0]).to_csv("C:/Users/ucg8nb/Downloads/passWrite.csv")
pd.DataFrame({passListen[0]: passListen[1:]}).set_index(passListen[0]).to_csv("C:/Users/ucg8nb/Downloads/passListen.csv")
pd.DataFrame({passSpeak[0]: passSpeak[1:]}).set_index(passSpeak[0]).to_csv("C:/Users/ucg8nb/Downloads/passSpeak.csv")
pd.DataFrame({passScore[0]: passScore[1:]}).set_index(passScore[0]).to_csv("C:/Users/ucg8nb/Downloads/passScore.csv")

pd.DataFrame({failRead[0]: failRead[1:]}).set_index(failRead[0]).to_csv("C:/Users/ucg8nb\Downloads/failRead.csv")
pd.DataFrame({failWrite[0]: failWrite[1:]}).set_index(failWrite[0]).to_csv("C:/Users/ucg8nb\Downloads/failWrite.csv")
pd.DataFrame({failListen[0]: failListen[1:]}).set_index(failListen[0]).to_csv("C:/Users/ucg8nb\Downloads/failListen.csv")
pd.DataFrame({failSpeak[0]: failSpeak[1:]}).set_index(failSpeak[0]).to_csv("C:/Users/ucg8nb\Downloads/failSpeak.csv")
pd.DataFrame({failScore[0]: failScore[1:]}).set_index(failScore[0]).to_csv("C:/Users/ucg8nb\Downloads/failScore.csv")
'''

#Starting with passing reading scores
#Creating the empirical distribution function
'''passReadPlotPos = stat.metaGaussianPlottingPositions(len(passRead))
sortedPassRead = sorted(passRead)'''
'''plt.scatter(x = sortedPassRead, y = passReadPlotPos, s = 15, color = 'gray')
plt.title('Empirical distribution function for passing reading scores')
plt.xlabel('Reading score')
plt.ylabel('Cumulative probability')
plt.savefig(stat.getDownloadsTab() + '/Empirical Distribution Passing Reading.png')
plt.clf()'''

'''failReadPlotPos = stat.metaGaussianPlottingPositions(len(failRead))
sortedFailRead = sorted(failRead)'''
'''plt.scatter(x = sortedFailRead, y = failReadPlotPos, s = 15, color = 'gray')
plt.title('Empirical distribution function for failing reading scores')
plt.xlabel('Reading score')
plt.ylabel('Cumulative probability')
plt.savefig(stat.getDownloadsTab() + '/Empirical Distribution Failing Reading.png')'''


'''nums = np.arange(16, 30, 0.001)
etaL = 12
etaU = 32
alpha1f = 0.6265
alpha2f = 0.4190
beta1f = 1.6067
beta2f = 1.3409
lrrgDFf = stat.logRatioDF(nums, "RG", alpha1f, beta1f, etaL, etaU)
lrlgDFf = stat.logRatioDF(nums, "LG", alpha2f, beta2f, etaL, etaU)
alpha1p = 0.5668
alpha2p = 0.6076
beta1p = 1.9365
beta2p = 1.7075
lrrgDFp = stat.logRatioDF(nums, "RG", alpha1p, beta1p, etaL, etaU)
lrlpDFp = stat.logRatioDF(nums, "LP", alpha2p, beta2p, etaL, etaU)'''

'''plt.scatter(x = nums, y = lrrgDFp, s = 15, color = 'gray')
plt.scatter(x = sortedPassRead, y = passReadPlotPos, color = 'lightgray')
plt.title('Log-Ratio Reflected Gumbel Distribution Function')
plt.xlabel('Reading Scores')
plt.ylabel('Cumulative Probability')
plt.savefig(stat.getDownloadsTab() + '/Log-Ratio Reflected Gumbel DF pass read.png')
plt.clf()

plt.scatter(x = nums, y = lrlpDFp, s = 15, color = 'gray')
plt.scatter(x = sortedPassRead, y = passReadPlotPos, color = 'lightgray')
plt.title('Log-Ratio Laplace Distribution Function')
plt.xlabel('Reading Scores')
plt.ylabel('Cumulative Probability')
plt.savefig(stat.getDownloadsTab() + '/Log-Ratio Laplace DF pass read.png')'''

'''lrrgdfp = stat.logRatiodf(nums, "RG", alpha1p, beta1p, etaL, etaU)
lrlpdfp = stat.logRatiodf(nums, "LP", alpha2p, beta2p, etaL, etaU)
lrrgdff = stat.logRatiodf(nums, "RG", alpha1f, beta1f, etaL, etaU)
lrlgdff = stat.logRatiodf(nums, 'LG', alpha2f, beta2f, etaL, etaU)

passingdf = [lrrgdfp, lrlpdfp]
failingdf = [lrrgdff, lrlgdff]
passingLabels = ['Log-Ratio Reflected Gumbel', "Log-Ratio Laplace"]
failingLabels = ['Log-Ratio Reflected Gumbel', 'Log-Ratio Logistic']'''

'''for i in range(len(passingdf)):
    for j in range(len(failingdf)):
        label = passingLabels[i] + ' ' + failingLabels[j]
        plt.scatter(x = nums, y = passingdf[i], color = 'gray', s = 15, label = 'Passing density')
        plt.scatter(x = nums, y = failingdf[i], color = 'lightgray', s = 15, label = 'Failing density')
        plt.legend()
        plt.title(label + "Density functions")
        plt.xlabel('Reading score')
        plt.ylabel("Probability density")
        plt.savefig(stat.getDownloadsTab() + '/' + label + '.png')
        plt.clf()'''

'''for i in range(len(passingdf)):
    for j in range(len(failingdf)):
        label = passingLabels[i] + ' ' + failingLabels[j]
        likelihoodRatio = passingdf[i] / failingdf[j]
        plt.scatter(x = nums, y = likelihoodRatio, color = 'gray', s = 15)
        plt.title(label)
        plt.xlabel('Reading score')
        plt.ylabel('Likelihood ratio score')
        plt.savefig(stat.getDownloadsTab() + '/' + label + ' likelihood ratio function.png')
        #plt.show()
        plt.clf()'''

'''plt.scatter(x = nums, y = lrrgDFf, s = 15, color = 'gray')
plt.scatter(x = sortedFailRead, y = failReadPlotPos, color = 'lightgray')
plt.title('Log-Ratio Reflected Gumbel Distribution Function')
plt.xlabel('Reading Scores')
plt.ylabel('Cumulative Probability')
plt.savefig(stat.getDownloadsTab() + '/Log-Ratio Reflected Gumbel DF fail read.png')
plt.clf()


plt.scatter(x = nums, y = lrlgDFf, s = 15, color = 'gray')
plt.scatter(x = sortedFailRead, y = failReadPlotPos, color = 'lightgray')
plt.title('Log-Ratio Logistic Distribution Function')
plt.xlabel('Reading Scores')
plt.ylabel('Cumulative Probability')
plt.savefig(stat.getDownloadsTab() + '/Log-Ratio Logistic DF fail read.png')'''

#Modeling the listening scores
#sortedPassListen = sorted(passListen)
#passListenPlotPos = stat.metaGaussianPlottingPositions(len(sortedPassListen))
'''plt.scatter(x = sortedPassListen, y = passListenPlotPos, s = 15, color = 'gray')
plt.title("Empirical distribution of passing listening scores")
plt.xlabel('Listening score')
plt.ylabel('Cumulative probability')
plt.savefig(stat.getDownloadsTab() + '/Empirical Distribution pass listen.png')
plt.clf()'''

#sortedFailListen = sorted(failListen)
#failListenPlotPos = stat.metaGaussianPlottingPositions(len(failListen))
'''plt.scatter(x = sortedFailListen, y = failListenPlotPos, s = 15, color = 'gray')
plt.title("Empirical distribution of failing listening scores")
plt.xlabel('Listening score')
plt.ylabel('Cumulative probability')
plt.savefig(stat.getDownloadsTab() + '/Empirical Distribution fail listen.png')'''

'''nums = np.arange(16, 30, 0.001)
etaL = 12
etaU = 32
alpha1p = 0.5982
alpha2p = 0.3905
beta1p = 1.5353
beta2p = 1.5282'''

'''lrlpDFp = stat.logRatioDF(nums, "LP", alpha1p, beta1p, etaL, etaU)
lrlgDFp = stat.logRatioDF(nums, "LG", alpha2p, beta2p, etaL, etaU)'''

'''plt.scatter(x = nums, y = lrlpDFp, color = 'gray', s = 15)
plt.scatter(x = sortedPassListen, y = passListenPlotPos, color = 'lightgray')
plt.title("Log-Ratio Laplace Distribution Function")
plt.xlabel("Listening Score")
plt.ylabel("Cumulative probability")
plt.savefig(stat.getDownloadsTab() + '/Log-Ratio Laplace DF pass listen.png')
plt.clf()

plt.scatter(x = nums, y = lrlgDFp, color = 'gray', s = 15)
plt.scatter(x = sortedPassListen, y = passListenPlotPos, color = 'lightgray')
plt.title("Log-Ratio Logistic Distribution Function")
plt.xlabel("Listening Score")
plt.ylabel("Cumulative probability")
plt.savefig(stat.getDownloadsTab() + '/Log-Ratio Logistic DF pass listen.png')'''

'''alpha1f = 0.7145
alpha2f = 0.4959
beta1f = 0.4170
beta2f = 0.7486

lrgbDFf = stat.logRatioDF(nums, "GB", alpha1f, beta1f, etaL, etaU)
lrlgDFf = stat.logRatioDF(nums, "LG", alpha2f, beta2f, etaL, etaU)'''

'''plt.scatter(x = nums, y = lrgbDFf, color = 'gray', s = 15)
plt.scatter(x = sortedFailListen, y = failListenPlotPos, color = 'lightgray')
plt.title("Log-Ratio Gumbel Distribution Function")
plt.xlabel("Listening Score")
plt.ylabel("Cumulative probability")
plt.savefig(stat.getDownloadsTab() + '/Log-Ratio Gumbel DF fail listen.png')
plt.clf()

plt.scatter(x = nums, y = lrlgDFf, color = 'gray', s = 15)
plt.scatter(x = sortedFailListen, y = failListenPlotPos, color = 'lightgray')
plt.title("Log-Ratio Logistic Distribution Function")
plt.xlabel("Listening Score")
plt.ylabel("Cumulative probability")
plt.savefig(stat.getDownloadsTab() + '/Log-Ratio Logistic DF fail listen.png')'''

'''lrlgdfp = stat.logRatiodf(nums, "LG", alpha2p, beta2p, etaL, etaU)
lrlpdfp = stat.logRatiodf(nums, "LP", alpha1p, beta1p, etaL, etaU)
lrgbdff = stat.logRatiodf(nums, 'GB', alpha1f, beta1f, etaL, etaU)
lrlgdff = stat.logRatiodf(nums, 'LG', alpha2f, beta2f, etaL, etaU)

passingdf = [lrlgdfp, lrlpdfp]
failingdf = [lrgbdff, lrlgdff]
passingLabels = ['Log-Ratio Logistic', "Log-Ratio Laplace"]
failingLabels = ['Log-Ratio Gumbel', 'Log-Ratio Logistic']'''

'''for i in range(len(passingdf)):
    for j in range(len(failingdf)):
        label = passingLabels[i] + ' ' + failingLabels[j]
        plt.scatter(x = nums, y = passingdf[i], color = 'gray', s = 15, label = 'Passing density')
        plt.scatter(x = nums, y = failingdf[i], color = 'lightgray', s = 15, label = 'Failing density')
        plt.legend()
        plt.title(label + "Density functions")
        plt.xlabel('Listening score')
        plt.ylabel("Probability density")
        plt.savefig(stat.getDownloadsTab() + '/' + label + ' listening.png')
        plt.clf()'''

'''for i in range(len(passingdf)):
    for j in range(len(failingdf)):
        label = passingLabels[i] + ' ' + failingLabels[j]
        likelihoodRatio = passingdf[i] / failingdf[j]
        plt.scatter(x = nums, y = likelihoodRatio, color = 'gray', s = 15)
        plt.title(label)
        plt.xlabel('Listening score')
        plt.ylabel('Likelihood ratio score')
        plt.savefig(stat.getDownloadsTab() + '/' + label + 'start at 16 likelihood ratio function listen.png')
        plt.clf()'''

'''plt.scatter(x = nums, y = lrlgdfp / lrlgdff, color = 'gray', s =15)
plt.title('Likelihood ratio function for listening scores')
plt.xlabel("Listening scores")
plt.ylabel('Likelihood ratio')
plt.savefig(stat.getDownloadsTab() + '/Likelihood Ratio for Listening.png')'''

#Getting correlation coefficient
nums = np.arange(14,30,0.001)
etaL = 12
etaU = 32
passReadDF = stat.logRatioDF(nums, "RG", 0.5668, 1.9365, etaL, etaU)
passReaddf = stat.logRatiodf(nums, "RG", 0.5668, 1.9365, etaL, etaU)
failReadDF = stat.logRatioDF(nums, 'RG', 0.6265, 1.6067, etaL, etaU)
failReaddf = stat.logRatiodf(nums, 'RG', 0.6265, 1.6067, etaL, etaU)
passListenDF = stat.logRatioDF(nums, "LG", 0.3905, 1.5282, etaL, etaU)
passListendf = stat.logRatiodf(nums, "LG", 0.3905, 1.5282, etaL, etaU)
failListenDF = stat.logRatioDF(nums, "LG", 0.4959, 0.7486, etaL, etaU)
failListendf = stat.logRatiodf(nums, "LG", 0.4959, 0.7486, etaL, etaU)

NQTpassReadEmp = stat.empiricalNQT(passRead)
NQTfailReadEmp = stat.empiricalNQT(failRead)
NQTpassListenEmp = stat.empiricalNQT(passListen)
NQTfailListenEmp = stat.empiricalNQT(failListen)

# passCorr = stat.getCorrelation(NQTpassReadEmp, NQTpassListenEmp)
# failCorr = stat.getCorrelation(NQTfailReadEmp, NQTfailListenEmp)

# print(passCorr)
# print(failCorr)

nqtPassRead = stat.inverseStandardNormalArray(passReadDF)
nqtFailRead = stat.inverseStandardNormalArray(failReadDF)
nqtPassListen = stat.inverseStandardNormalArray(passListenDF)
nqtFailListen = stat.inverseStandardNormalArray(failListenDF)

#passXi = stat.xi(passCorr, nqtPassRead, nqtPassListen)
#failXi = stat.xi(failCorr, nqtFailRead, nqtFailListen)

#passDen = passReaddf * passListendf
#failDen = failReaddf * failListendf

g = 0.65
#X,Y = np.meshgrid(nums, nums)

'''for i in range(len(nums)):
    for j in range(len(nums)):
        reading = nums[i]
        listening = nums[j]
        pR0 = stat.logRatioDF(reading, 'RG', 0.5668, 1.9365, 12, 32)
        pR1 = stat.logRatioDF(reading, "RG", 0.4190, 1.3409, 12, 32)
        pL0 = stat.logRatioDF(listening, "LG", 0.3905, 1.5282, 12, 32)
        pL1 = stat.logRatioDF(listening, "LG", 0.4959, 0.7486, 12, 32)
        xi0 = stat.xi(passCorr,stat.inverseStandardNormal(pR0), stat.inverseStandardNormal(pL0))
        xi1 = stat.xi(failCorr, stat.inverseStandardNormal(pR1), stat.inverseStandardNormal(pL1))
        lam = xi0 / xi1
        fR0 = stat.logRatiodf(reading, 'RG', 0.5668, 1.9365, 12, 32)
        fR1 = stat.logRatiodf(reading, "RG", 0.4190, 1.3409, 12, 32)
        fL0 = stat.logRatiodf(listening, "LG", 0.3905, 1.5282, 12, 32)
        fL1 = stat.logRatiodf(listening, "LG", 0.4959, 0.7486, 12, 32)
        den1 = fR1 * fL1
        den0 = fR0 * fL0
        prob = stat.pix(g, lam, den1, den0)
        grid[i][j] = prob
grid = np.array(grid)
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot_surface(nums, nums,grid)
plt.show()
'''

'''pixRead = np.power(1 + ((1 - g) / g) * passReaddf / failReaddf, -1)
plt.scatter(x = nums, y = pixRead, s = 15, color = 'gray')
plt.title('Posterior probability of failing the SPEAK exam with reading as predictor')
plt.xlabel("Reading score")
plt.ylabel("Posterior probability of failing the SPEAK exam")
plt.savefig(stat.getDownloadsTab() + "/Posterior probability with reading.png")'''
'''
plt.scatter(x = nums, y = passListendf / failListendf, color = 'gray')
plt.title("Likelihood ratio function for listening scores")
plt.xlabel("Listening score")
plt.ylabel("Likelihood ratio")
plt.show()'''

'''pixListen = np.power(1 + ((1-g)/g) * passListendf / failListendf, -1)
plt.scatter(x = nums, y = pixListen, s = 15, color = 'gray')
plt.title('Posterior probability of failing the SPEAK exam with listening as predictor')
plt.xlabel("Listening score")
plt.ylabel('Posterior probability')
plt.savefig(stat.getDownloadsTab() + '/Posteior probability with listening.png')'''

'''plt.scatter(1 - failReadDF, 1 - passReadDF, color = 'gray', s = 15, label = 'Reading predictor')
plt.scatter(1 - failListenDF, 1 - passListenDF, color = 'lightgray', s = 15, label = 'Listening predictor')
plt.legend()
plt.title("ROC for Reading and Listening")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.savefig(stat.getDownloadsTab() + '/ROC Curve for Reading and Listening')'''

'''readings = [21, 29, 29, 28, 16]
listenings = [24, 25, 30, 28, 21]

for i in range(len(readings)):
    xR = readings[i]
    xL = listenings[i]

    fR0 = stat.logRatiodf(xR, "RG", 0.5668, 1.9365, etaL, etaU)
    fR1 = stat.logRatiodf(xR, 'RG', 0.6265, 1.6067, etaL, etaU)
    fL0 = stat.logRatiodf(xL, "LG", 0.3905, 1.5282, etaL, etaU)
    fL1 = stat.logRatiodf(xL, "LG", 0.4959, 0.7486, etaL, etaU)

    FR0 = stat.logRatioDF(xR, "RG", 0.5668, 1.9365, etaL, etaU)
    FR1 = stat.logRatioDF(xR, 'RG', 0.6265, 1.6067, etaL, etaU)
    FL0 = stat.logRatioDF(xL, "LG", 0.3905, 1.5282, etaL, etaU)
    FL1 = stat.logRatioDF(xL, "LG", 0.4959, 0.7486, etaL, etaU)

    zR0 = stat.inverseStandardNormal(FR0)
    zR1 = stat.inverseStandardNormal(FR1)
    zL0 = stat.inverseStandardNormal(FL0)
    zL1 = stat.inverseStandardNormal(FL1)

    #print(xR, xL, fR0, fL0, fR1, fL1, zR0, zL0, zR1, zL1)

    likelihoodR = fR0 / fR1
    likelihoodL = fL0 / fL1

    xi0 = stat.xi(passCorr, zR0, zL0)
    xi1 = stat.xi(failCorr, zR1, zL1)

    lam = xi0 / xi1
    pi = stat.pix(g, lam, fR1 * fL1, fR0 * fL0)
    print(xR, xL, likelihoodR, likelihoodL, xi0, xi1, lam, pi)'''

passSpeakDF = stat.weibullDF(nums, 13.907, 5.9356, 12)
passSpeakdf = stat.weibulldf(nums, 13.907, 5.9356, 12)
failSpeakDF = stat.logWeibullDF(nums, 2.3703, 9.9086, 12)
failSpeakdf = stat.logWeibulldf(nums, 2.3703, 9.9086, 12)

nqtPassSpeak = stat.inverseStandardNormalArray(passSpeakDF)
nqtFailSpeak = stat.inverseStandardNormalArray(failSpeakDF)

nqtPassSpeakEmp = stat.empiricalNQT(passSpeak)
nqtFailSpeakEmp = stat.empiricalNQT(failSpeak)

passCorr = stat.getCorrelation(NQTpassReadEmp, nqtPassSpeakEmp)
failCorr = stat.getCorrelation(NQTfailReadEmp, nqtFailSpeakEmp)

# print(passCorr)
# print(failCorr)

'''plt.scatter(1 - failReadDF, 1 - passReadDF, color = 'gray', s = 15, label = 'Reading predictor')
plt.scatter(1 - failSpeakDF, 1 - passSpeakDF, color = 'lightgray', s = 15, label = 'Speaking predictor')
plt.legend()
plt.title('ROC for speaking and reading')
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.savefig(stat.getDownloadsTab() + '/ROC speaking and reading.png')'''
readings = [21, 29, 29, 28, 16, 30]
speakings = [24, 19, 19, 20, 19, 26]
for i in range(len(speakings)):
    xR = readings[i]
    xS = speakings[i]

    FS0 = stat.weibullDF(xS, 13.907, 5.9356, 12)
    fS0 = stat.weibulldf(xS, 13.907, 5.9356, 12)
    FS1 = stat.logWeibullDF(xS, 2.3703, 9.9086, 12)
    fS1 = stat.logWeibulldf(xS, 2.3703, 9.9086, 12)
    FR0 = stat.logRatioDF(xR, "RG", 0.5668, 1.9365, etaL, etaU)
    fR0 = stat.logRatiodf(xR, "RG", 0.5668, 1.9365, etaL, etaU)
    FR1 = stat.logRatioDF(xR, 'RG', 0.6265, 1.6067, etaL, etaU)
    fR1 = stat.logRatiodf(xR, 'RG', 0.6265, 1.6067, etaL, etaU)

    zR0 = stat.inverseStandardNormal(FR0)
    zR1 = stat.inverseStandardNormal(FR1)
    zS0 = stat.inverseStandardNormal(FS0)
    zS1 = stat.inverseStandardNormal(FS1)

    #print(xR, xS, zR0, zR1, zS0, zS1)

    likelihoodReading = fR0 / fR1
    likelihoodSpeaking = fS0 / fS1

    xi0 = stat.xi(passCorr, zR0, zS0)
    xi1 = stat.xi(failCorr, zR1, zS1)

    lam = xi0 / xi1

    pix = stat.pix(g, lam, fR1 * fS1, fR0 * fS0)

    # print(xR, xS, likelihoodReading, likelihoodSpeaking, xi0, xi1, lam, pix)

