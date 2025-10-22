import Stats_Functions as stat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sheets = np.arange(1, 23, 1)
priorProb = 1 / (-6 * sheets + 152)
# plt.scatter(x = sheets, y = priorProb, color = 'gray')
# plt.title('Prior Probabilty as a Function of Sheet Number')
# plt.xlabel('Sheet Number')
# plt.ylabel('Prior Probability')
# plt.savefig(stat.getDownloadsTab() + '/Prior Probability.png')
# plt.clf()

alpha0 = 5.0971
alpha1 = 16.9409
params0 = [alpha0, 0, 0]
params1 = [alpha1, 0, 0]

nums = np.arange(0.01, 50, 0.01)
df0 = stat.getdf('EX', nums, params0)
df1 = stat.getdf('EX', nums, params1)

# plt.scatter(x = nums, y = df0, color = 'gray', label = 'f0(x)', s= 5)
# plt.scatter(x = nums, y = df1, color = 'lightgray', label = 'f1(x)', s= 5)
# plt.legend()
# plt.title('Conditional density functions')
# plt.xlabel('Departure')
# plt.ylabel('Probability density')
# plt.savefig(stat.getDownloadsTab() + '/Conditional density functions.png')

likelihood = df0 / df1
# plt.scatter(x = nums, y = likelihood, color = 'gray', s =5)
# plt.title("Likelihood Ratio Function")
# plt.xlabel("Departure")
# plt.ylabel('Likelihood Ratio')
# plt.savefig(stat.getDownloadsTab() + '/Likelihood Ratio Function.png')

DF0 = stat.getDF('EX', nums, params0)
DF1 = stat.getDF('EX', nums, params1)



detect = 1 - DF1
false = 1 - DF0

# plt.scatter(x = false, y = detect, color = 'gray', s = 5)
# plt.title("ROC for departure")
# plt.ylabel('Detection Rate')
# plt.xlabel("False Alarm Rate")
# plt.savefig(stat.getDownloadsTab() + '/ROC for Departure.png')
g2 = priorProb[2-1]
g12 = priorProb[12-1]
# g22 = priorProb[22-1]

df0 = stat.getdf('EX', nums, params0)
df1 = stat.getdf("EX", nums, params1)

pix2 = stat.pix(g2, 1, df0, df1)
pix12 = stat.pix(g12, 1, df0, df1)

desiredProb = 0.2097

distanceFromProb2 = np.absolute(pix2 - desiredProb)
min_index2 =  np.argmin(distanceFromProb2)
departure2 = nums[min_index2]
print(departure2)
print(pix2[min_index2])
distanceFromProb12 = np.absolute(pix12 - desiredProb)
min_index12 = np.argmin(distanceFromProb12)
departure12 = nums[min_index12]
print(departure12)
print(pix12[min_index12])

departures = [29, 34, 41]
sheetNums = [2, 12]
priors = [priorProb[x - 1] for x in sheetNums]
displaySheetNuns = [2, 2, 2, 12, 12, 12]
displayDepartures = departures * 2
cutoff = 0.2097
df0 = []
df1 = []
pix = []
disvector = [0, 156, 1566, 978]
for i in range(len(sheetNums)):
    tempdf0 = stat.getdf('EX', np.array(departures), params0)
    tempdf1 = stat.getdf('EX', np.array(departures), params1)
    temppix = stat.pix(priors[i], 1, tempdf0, tempdf1)
    df0.extend(tempdf0)
    df1.extend(tempdf1)
    pix.extend(temppix)
decision = []
disutility = []
badDis = []
perfect = []
for prob in pix:
    if prob > cutoff:
        decision.append(1)
        disutility.append(prob * disvector[3] + (1 - prob) * disvector[1])
        badDis.append(prob * disvector[2] + (1 - prob) * disvector[0])
    else:
        decision.append(0)
        disutility.append(prob * disvector[2] + (1 - prob) * disvector[0])
        badDis.append(prob * disvector[3] + (1 - prob) * disvector[1])
    perfect.append(prob * disvector[3])
colNames = ['Sheet Number', 'Departure', '$y$', '$a$', 'Disutility $(D_j^*(x))$' , 'Opposite Decision Disutility']
cols = [displaySheetNuns, displayDepartures,pix, decision, disutility, badDis]
stat.printInLatexTable(cols, colNames)  
vpf = np.array(disutility) - np.array(perfect)
colNames = ['Sheet Number', 'Departure', 'Expected Optimal Disutility', "Expected Perfect Forecaster Disutility", 'VPF']
cols = [displaySheetNuns, displayDepartures, disutility, perfect, vpf]
stat.printInLatexTable(cols, colNames)





# print(g2, g12, g22)

# departures = np.array([7, 19, 49])
# df0spec = stat.getdf('EX', departures, params0)
# df1spec = stat.getdf('EX', departures, params1)

# pix2spec = stat.pix(g2, 1, df0spec, df1spec)
# pix12spec = stat.pix(g12, 1, df0spec, df1spec)
# pix22spec = stat.pix(g22, 1, df0spec, df1spec)

# print(pix2spec)
# print(pix12spec)
# print(pix22spec)


# pix2 = stat.pix(g2, 1, df0, df1)
# pix12 = stat.pix(g12, 1, df0, df1)
# pix22 = stat.pix(g22, 1, df0, df1)

# plt.scatter(x = nums, y = pix2, color = 'gray', s= 5, label = 'Sheet 2')
# plt.scatter(x = nums, y = pix12, color = 'darkgray', s= 5, label = 'Sheet 12')
# plt.scatter(x = nums, y = pix22, color = 'lightgray', s= 5, label = 'Sheet 22')
# plt.legend()
# plt.title("Posterior Probability Function by Sheet Number")
# plt.xlabel('Departure')
# plt.ylabel('Posterior Probability')
# plt.savefig(stat.getDownloadsTab() + '/Posterior Probability by Sheet Number.png')