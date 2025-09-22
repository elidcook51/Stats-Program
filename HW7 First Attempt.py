import Stats_Functions as stat
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd



df = pd.read_csv("C:/Users/ucg8nb/Downloads/Table 5.11.csv")
passDf = df[df['Passed'] == 1]
failDf = df[df['Passed'] == 0]

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

'''
x = np.array(passRead)
Fx = stat.metaGaussianPlottingPositions(len(x))
y = np.array(passSpeak)
Fy = stat.metaGaussianPlottingPositions(len(y))

NQTX = [0] * len(Fx)
NQTY = [0] * len(Fx)
tempX = list(x)
tempY = list(y)
for i in range(len(Fx)):
    indexX = tempX.index(min(tempX))
    NQTX[indexX] = stat.inverseStandardNormal(Fx[i])
    tempX[indexX] = np.inf
    indexY = tempY.index(min(tempY))
    NQTY[indexY] = stat.inverseStandardNormal(Fy[i])
    tempY[indexY] = np.inf

print(stat.getCorrelation(NQTX,NQTY))


tempX = failRead
tempY = failSpeak

plotPos = stat.metaGaussianPlottingPositions(len(tempX))
NQTX = [0] * len(tempX)
NQTY = [0] * len(tempY)

for i in range(len(plotPos)):
    indexX = tempX.index(min(tempX))
    NQTX[indexX] = stat.inverseStandardNormal(plotPos[i])
    tempX[indexX] = np.inf
    indexY = tempY.index(min(tempY))
    NQTY[indexY] = stat.inverseStandardNormal(plotPos[i])
    tempY[indexY] = np.inf

print(stat.getCorrelation(NQTX,NQTY))
'''

'''
aR1 = 3.2016
bR1 = 3.4354
bR0 = 10.4109
aL1 = 1.7739
bL1 = 2.7698
aL0 = 0.8423
bL0 = 1.4643
aS1 = 0.5426
bS1 = 1.4136
aS0 = 0.3421
bS0 = 0.9940
nums = np.arange(0.001, 29.99,0.001)
FR1 = np.exp(-1 * np.exp(-1 * (np.log(nums / (30 - nums)) - bR1) / aR1 ) )
FR0 = np.power(nums / 30, bR0)
FL1 = np.exp(-1 * np.exp(-1 * (np.log(nums / (30 - nums)) - bL1) / aL1 ) )
FL0 = np.exp(-1 * np.exp(-1 * (np.log(nums / (30 - nums)) - bL0) / aL0 ) )
g = 0.35
FS1 = np.exp(-1 * np.exp(-1 * (np.log(nums / (30 - nums)) - bS1) / aS1 ) )
FS0 = 1 - np.exp(-1 * np.exp((np.log(nums / (30 - nums)) - bS0) / aS0 ) )
'''

'''
FR1emp = stat.metaGaussianPlottingPositions(len(passRead))
plt.scatter(x = nums, y = FR1, color = 'gray')
plt.scatter(x = sorted(passRead), y= FR1emp, color = 'lightgray')
plt.ylim(0,1)
plt.title('Reading distribution function conditioned on V = 1')
plt.xlabel('Reading score')
plt.ylabel('Cumulative probability')
plt.savefig(stat.getDownloadsTab() + '/FR1 Distribution Function.png')
plt.clf()

FR0emp = stat.metaGaussianPlottingPositions(len(failRead))
plt.scatter(x = nums, y = FR0, color = 'gray')
plt.scatter(x = sorted(failRead), y = FR0emp, color = 'lightgray')
plt.title('Reading distribution function conditioned on V = 0')
plt.ylim(0,1)
plt.xlabel('Reading score')
plt.ylabel('Cumulative probability')
plt.savefig(stat.getDownloadsTab() + '/FR0 Distribution Function.png')
plt.clf()



FL1emp = stat.metaGaussianPlottingPositions(len(passListen))
plt.scatter(x = nums, y = FL1, color = 'gray')
plt.scatter(x = sorted(passListen), y = FL1emp, color = 'lightgray')
plt.title('Listening distribution function conditioned on V = 1')
plt.ylim(0, 1)
plt.xlabel('Listening score')
plt.ylabel("Conditional probability")
plt.savefig(stat.getDownloadsTab() + '/FL1 Distribution Function.png')
plt.clf()




FL0emp = stat.metaGaussianPlottingPositions(len(failListen))
plt.scatter(x = nums, y = FL0, color = 'gray')
plt.scatter(x = sorted(failListen), y = FL0emp, color = 'lightgray')
plt.title('Listening distribution function conditioned on V = 0')
plt.ylim(0, 1)
plt.xlabel('Listening score')
plt.ylabel("Conditional probability")
plt.savefig(stat.getDownloadsTab() + '/FL0 Distribution Function.png')
plt.clf()


FS1emp = stat.metaGaussianPlottingPositions(len(passSpeak))
plt.scatter(x = nums, y = FS1, color = 'gray')
plt.scatter(x = sorted(passSpeak), y = FS1emp, color = 'lightgray')
plt.title('Speaking distribution function conditioned on V = 1')
plt.ylim(0,1)
plt.xlabel('Speaking score')
plt.ylabel('Conditional probability')
plt.savefig(stat.getDownloadsTab() + '/FS1 Distribution Function.png')
plt.clf()

FS0emp = stat.metaGaussianPlottingPositions(len(failSpeak))
plt.scatter(x = nums, y = FS0, color = 'gray')
plt.scatter(x = sorted(failSpeak), y = FS0emp, color = 'lightgray')
plt.title('Speaking distribution function conditioned on V = 0')
plt.ylim(0,1)
plt.xlabel('Speaking score')
plt.ylabel('Conditional probability')
plt.savefig(stat.getDownloadsTab() + '/FS0 Distribution Function.png')
plt.clf()
'''

#def fR1x(x):
#    return (1 / aR1) * np.exp(-1 * (x - bR1) /  aR1) * np.exp(-1 * np.exp(-1 * (x - bR1) / aR1))
#fR1 = (30 / (nums * (30 - nums))) * fR1x(np.log(nums / (30 - nums) ))
'''plt.scatter(x = nums, y = fR1, color = 'gray')
plt.title('Density function of reading conditioned on V = 1')
plt.xlabel('Reading score')
plt.ylabel("Probability density")
plt.savefig(stat.getDownloadsTab() + '/fR1 density function.png')
plt.clf()'''


#fR0 = (bR0 / (30 - 0)) * np.power((nums - 0) / (30 - 0), bR0 - 1)
'''
plt.scatter(x = nums, y = fR0, color = 'gray')
plt.title('Density function of reading conditioned on V = 0')
plt.xlabel('Reading score')
plt.ylabel("Probability density")
plt.savefig(stat.getDownloadsTab() + '/fR0 density function.png')
plt.clf()
'''

#def fL1x(x):
#    return (1 / aL1) * np.exp(-1 * (x - bL1) / aL1) * np.exp(-1 * np.exp(-1 * (x - bL1) / aL1))
#fL1 = (30 / (nums * (30 - nums))) * fL1x(np.log(nums / (30 - nums)))
'''
plt.scatter(x = nums, y = fL1, color = 'gray')
plt.title('Density function of listening conditioned on V = 1')
plt.xlabel('Listening score')
plt.ylabel("Probability density")
plt.savefig(stat.getDownloadsTab() + '/fL1 density function.png')
plt.clf()
'''

#def fL0x(x):
#    return (1 / aL0) * np.exp(-1 * (x - bL0) / aL0) * np.exp(-1 * np.exp(-1 * (x - bL0) / aL0))
#fL0  = (30 / (nums * (30 - nums))) * fL0x(np.log(nums / (30 - nums)))
'''
plt.scatter(x = nums, y = fL0, color = 'gray')
plt.title('Density function of listening conditioned on V = 0')
plt.xlabel('Listening score')
plt.ylabel("Probability density")
plt.savefig(stat.getDownloadsTab() + '/fL0 density function.png')
plt.clf()
'''

#def fS0x(x):
#    return (1 / aS0) * np.exp((x - bS0) / aS0) * np.exp(-1 * np.exp((x-bS0) / aS0 ))
#fS0 = (30 / (nums * (30 - nums))) * fS0x(np.log(nums / (30 - nums) ))
'''
plt.scatter(x = nums, y = fS0, color = 'gray')
plt.title('Density function of speaking conditioned on V = 0')
plt.xlabel('Speaking score')
plt.ylabel('Probability density')
plt.savefig(stat.getDownloadsTab() + '/fS0 density function.png')
plt.clf()
'''

'''def fS1x(x):
    return (1 / aS1) * np.exp(-1 * (x - bS1) / aS1) * np.exp(-1 * np.exp(-1 * (x - bS1) / aS1 ))
fS1 = (30 / (nums * (30 - nums))) * fS1x(np.log(nums / (30 - nums) ))
temp = []
for a in list(fS1):
    if a < 0.01:
        a = 0.01
    temp.append(a)
fS1 = np.array(temp)'''
'''plt.scatter(x = nums, y = fS1, color = 'gray')
plt.title('Density function of speaking conditioned on V = 1')
plt.xlabel('Speaking score')
plt.ylabel('Probability density')
plt.savefig(stat.getDownloadsTab() + '/fS1 density function.png')
plt.clf()'''


#likelihoodR = fR0 / fR1
#likelihoodL = fL0 / fL1
#likelihoodS = fS0 / fS1
'''
plt.scatter(x = nums, y = likelihoodS, color = 'gray')
plt.title("Likelihood ratio function for speaking")
plt.xlabel('Speaking score')
plt.ylabel('Likelihood ratio of speaking value')
plt.savefig(stat.getDownloadsTab() + '/Likelihood ratio function for speaking.png')
plt.clf()
'''
'''
plt.scatter(x = nums, y = likelihoodR, color = 'gray')
plt.title('Likelihood ratio function for reading')
plt.xlabel('Reading score')
plt.ylabel('Likelihood ratio of reading value')
plt.savefig(stat.getDownloadsTab() + '/Likelihood ratio function for reading.png')
plt.clf()

plt.scatter(x = nums, y = likelihoodL, color = 'gray')
plt.title('Likelihood ratio function for listening')
plt.xlabel('Listening score')
plt.ylabel('Likelihood ratio of listening value')
plt.savefig(stat.getDownloadsTab() + '/Likelihood ratio function for listening.png')
plt.clf()
'''

'''
plt.scatter(x = 1 - FR0, y = 1-FR1, color = 'gray')
plt.title("ROC Curve for Reading")
plt.xlabel("False positive rate")
plt.ylabel('True positive rate')
plt.savefig(stat.getDownloadsTab() + '/ROC Curve for Reading')
plt.clf()

plt.scatter(x = 1 - FL0, y = 1 - FL1, color = 'gray')
plt.title('ROC Curve for Listening')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.savefig(stat.getDownloadsTab() + '/ROC Curve for Listening')
plt.clf()
'''
'''
plt.scatter(x = 1 - FS0, y = 1 - FS1, color = 'gray')
plt.title("ROC Curve for Speaking")
plt.xlabel("False positive rate")
plt.ylabel('True positive rate')
plt.savefig(stat.getDownloadsTab() + '/ROC Curve for Speaking')
plt.clf()
'''
#piR = np.power(1 + ((1 - g)/g) * likelihoodR, -1)
#piL = np.power(1 + ((1 - g)/g) * likelihoodL, -1)
#piS = np.power(1 + ((1 - g)/g) * likelihoodS, -1)

#tempPiR = list(piR[~np.isnan(piR)])
#tempPiL = list(piL[~np.isnan(piL)])
#tempPiS = list(piS[~np.isnan(piS)])
#print(tempPiS.index(min(tempPiS)))
#print(tempPiR.index(min(tempPiR)))
#print(tempPiL.index(min(tempPiL)))
#print(list(nums)[26979])
#print(list(nums)[21595])
#print(list(nums)[19064])
'''
plt.scatter(x = nums, y = piS, color = 'gray')
plt.title('Posterior probability of S')
plt.xlabel('Value of speaking score')
plt.ylabel('Probability of V = 1')
plt.ylim(0,1)
plt.savefig(stat.getDownloadsTab() + '/Posterior Probability for S.png')
plt.clf()
'''
'''
plt.scatter(x = nums, y = piR, color = 'gray')
plt.title('Posterior probability of R')
plt.xlabel('Value of reading score')
plt.ylabel('Probability of V = 1')
plt.ylim(0,1)
plt.savefig(stat.getDownloadsTab() + '/Posterior Probability for R.png')
plt.clf()


plt.scatter(x = nums, y = piL, color = 'gray')
plt.title('Posterior probability of L')
plt.xlabel('Value of listening score')
plt.ylabel('Probability of V = 1')
plt.ylim(0,1)
plt.savefig(stat.getDownloadsTab() + '/Posterior Probability for L.png')
plt.clf()
'''

'''
plt.scatter(x = 1 - FL0, y = 1 - FL1, color = 'gray', label = 'Listening')
plt.scatter(x = 1 - FR0, y = 1 - FR1, color = 'lightgray', label = 'Reading')
plt.title('ROC Curves for R and L plotted on the same graph')
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.legend()
plt.savefig(stat.getDownloadsTab() + '/ROC for both R and L.png')
plt.clf()


plt.scatter(x = 1 - FS0, y = 1 - FS1, color = 'gray', label = 'Speaking')
plt.scatter(x = 1 - FR0, y = 1 - FR1, color = 'lightgray', label = 'Reading')
plt.title('ROC Curves for R and S plotted on the same graph')
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.legend()
plt.savefig(stat.getDownloadsTab() + '/ROC for both R and S.png')
plt.clf()
'''

'''R = 16
L = 24
S = 19
fr1 = (30 / (R * (30 - R))) * fR1x(np.log(R / (30 - R)))
fr0 = (bR1 / (30)) * np.power(R / (30), bR1 - 1)
fl1 = (30 / (L * (30 - L))) * fL1x(np.log(L / (30 - L)))
fl0 = (30 / (L * (30 - L))) * fL0x(np.log(L / (30 - L)))
fs1 = (30 / (S * (30 - S))) * fS1x(np.log(L / (30 - S)))
fs0 = (30 / (S * (30 - S))) * fS0x(np.log(L / (30 - S)))
Fr1 = np.exp(-1 * np.exp(-1 * (np.log(R / (30 - R)) - bR1) / aR1 ) )
Fr0 = np.power(R / 30, bR0)
Fl1 = np.exp(-1 * np.exp(-1 * (np.log(L / (30 - L)) - bL1) / aL1 ) )
Fl0 = np.exp(-1 * np.exp(-1 * (np.log(L / (30 - L)) - bL0) / aL0 ) )
Fs1 = np.exp(-1 * np.exp(-1 * (np.log(S / (30 - S)) - bS1) / aS1 ) )
Fs0 = 1 - np.exp(-1 * np.exp((np.log(S / (30 - S)) - bS0) / aS0 ) )
print(fr0 / fr1)
#print(fl0 / fl1)
print(fs0 / fs1)
zr1 = stat.inverseStandardNormal(Fr1)
zr0 = stat.inverseStandardNormal(Fr0)
zl1 = stat.inverseStandardNormal(Fl1)
zl0 = stat.inverseStandardNormal(Fl0)
zs1 = stat.inverseStandardNormal(Fs1)
zs0 = stat.inverseStandardNormal(Fs0)
#print(zr0, zl0, zr1, zl1)
print(zr0, zs0, zr1, zs1)
gamma1 = 0.3136
gamma0 = 0.2040
xi1 = (1 / np.sqrt(1 - gamma1)) * np.exp((-1 * gamma1 / (2 * (1 - gamma1 ** 2))) * (gamma1 * np.power(zr1,2) - zr1 * zs1 + gamma1 * np.power(zs1,2)))
xi0 = (1 / np.sqrt(1 - gamma0)) * np.exp((-1 * gamma0 / (2 * (1 - gamma0 ** 2))) * (gamma0 * np.power(zr0,2) - zr0 * zs0 + gamma0 * np.power(zs0,2)))
print(xi1)
print(xi0)
print(xi0 / xi1)
pix = np.power(1 + ((1-g) / g) * (xi0/xi1) * (fr0 * fs0) / (fr1 * fs1), -1)
print(pix)
'''