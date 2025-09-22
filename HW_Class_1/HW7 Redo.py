import Stats_Functions as stat
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd



df = pd.read_csv("C:/Users/ucg8nb/Downloads/Table 5.11.csv")
passDf = df[df['Passed'] == 1]
failDf = df[df['Passed'] == 0]

#print(len(df))
#print(len(passDf))

#passRead = passDf['R'].tolist()
#passWrite = passDf['W'].tolist()
#passListen = passDf['L'].tolist()
passSpeak = passDf['S'].tolist()
#passScore = passDf['SP'].tolist()

#failRead = failDf['R'].tolist()
#failWrite = failDf['W'].tolist()
#failListen = failDf['L'].tolist()
failSpeak = failDf['S'].tolist()
#failScore = failDf['SP'].tolist()

'''
pd.DataFrame({passRead[0]: passRead[1:]}).set_index(passRead[0]).to_csv("C:/Users/ucg8nb/Downloads/passRead.csv")
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

passingPlotPos = stat.metaGaussianPlottingPositions(len(passSpeak))
sortPassSpeak = sorted(passSpeak)

failingPlotPos = stat.metaGaussianPlottingPositions(len(failSpeak))
sortFailSpeak = sorted(failSpeak)

"""plt.scatter(x = sortPassSpeak, y = passingPlotPos, color = 'gray')
plt.xlim(0,30)
plt.title('Empirical distribution of passing speaking scores')
plt.xlabel('Passing speaking score')
plt.ylabel('Empirical cumulative probability')
plt.savefig(stat.getDownloadsTab() + '/Empirical DF passing speak.png')"""

eta = 12
a1 = 13.907
b1 = 5.9356
mu2 = 0.2126
sigma2 = 2.5367

fa1 = 2.3703
fb1 = 9.9086
fa2 = 9.7405
fb2 = 3.8316

nums = np.arange(14, 31, 0.01)
weibullDf = 1 - np.exp(-1 * np.power((nums - eta) / a1, b1))
logNormalDf = stat.standardNormalArray((np.log(nums - eta) - mu2) / sigma2)

weibullDf2 = 1 - np.exp(-1 * np.power((nums - eta) / fa2, fb2))
logWeibullDf = 1 - np.exp(-1* np.power((np.log(nums - eta + 1)) / fa1, fb1))


'''plt.scatter(x = sortPassSpeak, y = passingPlotPos, color = 'lightgray', label = 'Empirical')
plt.scatter(x = nums, y = weibullDf, color = 'gray', label = 'Weibull')
plt.xlim(11, 31)
plt.title('Fitted Weibull distribution function of speaking score')
plt.xlabel('Speaking score')
plt.ylabel("Cumulative distribution")
plt.legend()
plt.savefig(stat.getDownloadsTab() + '/Weibull Distribution Function.png')
plt.clf()

plt.scatter(x = sortPassSpeak, y = passingPlotPos, color = 'lightgray', label = 'Empirical')
plt.scatter(x = nums, y = logNormalDf, color = 'gray', label = 'Log-Normal')
plt.xlim(11, 31)
plt.title('Fitted Log-Normal distribution function of speaking score')
plt.xlabel('Speaking score')
plt.ylabel("Cumulative distribution")
plt.legend()
plt.savefig(stat.getDownloadsTab() + '/logNormal Distribution Function.png')
plt.clf()

plt.scatter(x = sortFailSpeak, y = failingPlotPos, color = 'lightgray', label = 'Empirical')
plt.scatter(x = nums, y = weibullDf2, color = 'gray', label = 'Log-Normal')
plt.xlim(11, 31)
plt.title('Fitted Log-Normal distribution function of speaking score')
plt.xlabel('Speaking score')
plt.ylabel("Cumulative distribution")
plt.legend()
plt.show()
plt.clf()

plt.scatter(x = sortFailSpeak, y = failingPlotPos, color = 'lightgray', label = 'Empirical')
plt.scatter(x = nums, y = logWeibullDf, color = 'gray', label = 'Log-Normal')
plt.xlim(11, 31)
plt.title('Fitted Log-Normal distribution function of speaking score')
plt.xlabel('Speaking score')
plt.ylabel("Cumulative distribution")
plt.legend()
plt.show()
plt.clf()'''

weibulldf = (b1 / a1) * np.power((nums - eta) / a1, b1 - 1) * np.exp(-1 * np.power((nums - eta) / a1 , b1))
weibulldf2 = (fb2 / fa2) * np.power((nums - eta) / fa2, fb2 - 1) * np.exp(-1 * np.power((nums - eta) / fa2 , fb2))
logNormaldf = (1 / ((nums - eta) * sigma2 * np.sqrt(2 * math.pi))) * np.exp((-1 / 2) * np.power((np.log(nums - eta) - mu2) / sigma2 , 2))
logWeibulldf = (fb1 / (fa1 * (nums - eta + 1))) * np.power((np.log(nums - eta + 1)) / fa1, fb1 - 1) * np.exp(-1 * np.power((np.log(nums - eta + 1)) / fa1, fb1))

'''passing = [weibulldf, logNormaldf]
failing = [weibulldf2, logWeibulldf]
passLabel = ['Weibull Pass', 'Log Normal Pass']
failLabel = ['Weibull Fail', 'Log Weibull Fail']

for i in range(len(passing)):
    for j in range(len(failing)):
        label = passLabel[i] + ', ' + failLabel[j]
        likelihood = failing[j] / passing[i]
        plt.scatter(x = nums, y = likelihood, color = 'gray')
        plt.title(label)
        plt.savefig(stat.getDownloadsTab() + '/' + label + '.png')
        plt.clf()
'''

'''plt.scatter(x = sortFailSpeak, y = failingPlotPos, color = 'gray')
plt.xlim(11,31)
plt.title('Empirical distribution function for failing speaking scores')
plt.xlabel("Speaking score")
plt.ylabel("Cumulative probability")
plt.savefig(stat.getDownloadsTab() + '/Empirical distribution failing speaking.png')'''

'''plt.scatter(x = nums, y = weibullDf2, color = 'gray', label = 'Weibull')
plt.scatter(x = sortFailSpeak, y = failingPlotPos, color = 'lightgray', label = 'Empirical')
plt.title('Fitted Weibull distribution of failing speaking scores')
plt.xlim(11, 31)
plt.xlabel('Speaking score')
plt.ylabel('Cumulative probability')
plt.savefig(stat.getDownloadsTab() + '/Failing Weibull speaking distribution.png')
plt.clf()

plt.scatter(x = nums, y = logWeibullDf, color = 'gray', label = 'log-Weibull')
plt.scatter(x = sortFailSpeak, y = failingPlotPos, color = 'lightgray', label = 'Empirical')
plt.title('Fitted log-Weibull distribution of failing speaking scores')
plt.xlim(11, 31)
plt.xlabel('Speaking score')
plt.ylabel('Cumulative probability')
plt.savefig(stat.getDownloadsTab() + '/Failing log-Weibull speaking distribution.png')
plt.clf()'''

#print(np.max(logWeibullDf / weibullDf))

'''plt.scatter(x = nums, y = weibullDf, color = 'gray')
plt.xlim(13, 31)
plt.title('Passing speaking score distribution function')
plt.xlabel('Speaking score')
plt.ylabel('Cumulative probability')
plt.savefig(stat.getDownloadsTab() + '/Passing speaking score distribution.png')
plt.clf()

plt.scatter(x = nums, y = logWeibullDf, color = 'gray')
plt.xlim(13, 31)
plt.title('Failing speaking score distribution function')
plt.xlabel('Speaking score')
plt.ylabel('Cumulative probability')
plt.savefig(stat.getDownloadsTab() + '/Failing speaking score distribution.png')
plt.clf()

plt.scatter(x = nums, y = weibulldf, color = 'gray')
plt.xlim(13, 31)
plt.title('Passing speaking score density function')
plt.xlabel('Speaking score')
plt.ylabel('Probability density')
plt.savefig(stat.getDownloadsTab() + '/Passing speaking score density.png')
plt.clf()

plt.scatter(x = nums, y = logWeibulldf, color = 'gray')
plt.xlim(13, 31)
plt.title('Failing speaking score density function')
plt.xlabel('Speaking score')
plt.ylabel('Probability density')
plt.savefig(stat.getDownloadsTab() + '/Failing speaking score density.png')
plt.clf()'''

'''plt.scatter(x = nums, y = logWeibulldf / weibulldf, color = 'gray')
plt.xlim(13, 31)
plt.title('Likelihood ratio for speaking score')
plt.xlabel('Speaking score')
plt.ylabel('Likelihood ratio')
plt.savefig(stat.getDownloadsTab() + '/Likelihood ratio speaking score.png')
plt.clf()'''

'''plt.scatter(x = 1 - logWeibullDf, y = 1 - weibullDf, color = 'gray')
plt.title('ROC for speaking score')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.savefig(stat.getDownloadsTab() + '/ROC Curve.png')
plt.clf()'''
'''gold = 35/118
g = 0.35

pixold = np.power(1 + ((1-gold)/gold) * logWeibulldf / weibulldf, -1)
pix = np.power(1 + ((1-g)/g) * logWeibulldf / weibulldf, -1)
plt.scatter(x = nums, y = pixold, color = 'lightgray', label = 'Without informed prior')
plt.scatter(x = nums, y = pix, color = 'gray', label = "With informed prior")
plt.xlim(13,31)
plt.title('Posterior Probability of passing SPEAK exam by speaking score')
plt.xlabel('Speaking score')
plt.ylabel('Posterior Probability of passing SPEAK')
plt.legend()
plt.savefig(stat.getDownloadsTab() + '/Both posterior probability.png')'''
#plt.savefig(stat.getDownloadsTab() + '/Posterior Probability of SPEAK exam.png')
#plt.clf()

'''pix = np.power(1 + ((1-g)/g) * logWeibulldf / weibulldf, -1)
plt.scatter(x = nums, y = pix, color = 'gray')
plt.xlim(13,31)
plt.title('Posterior Probability of passing SPEAK exam by speaking score')
plt.xlabel('Speaking score')
plt.ylabel('Posterior Probability of passing SPEAK')
plt.savefig(stat.getDownloadsTab() + '/Posterior Probability of SPEAK exam updated.png')
plt.clf()'''