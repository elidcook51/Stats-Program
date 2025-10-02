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
g22 = priorProb[22-1]

print(g2, g12, g22)

departures = np.array([7, 19, 49])
df0spec = stat.getdf('EX', departures, params0)
df1spec = stat.getdf('EX', departures, params1)

pix2spec = stat.pix(g2, 1, df0spec, df1spec)
pix12spec = stat.pix(g12, 1, df0spec, df1spec)
pix22spec = stat.pix(g22, 1, df0spec, df1spec)

print(pix2spec)
print(pix12spec)
print(pix22spec)


pix2 = stat.pix(g2, 1, df0, df1)
pix12 = stat.pix(g12, 1, df0, df1)
pix22 = stat.pix(g22, 1, df0, df1)

plt.scatter(x = nums, y = pix2, color = 'gray', s= 5, label = 'Sheet 2')
plt.scatter(x = nums, y = pix12, color = 'darkgray', s= 5, label = 'Sheet 12')
plt.scatter(x = nums, y = pix22, color = 'lightgray', s= 5, label = 'Sheet 22')
plt.legend()
plt.title("Posterior Probability Function by Sheet Number")
plt.xlabel('Departure')
plt.ylabel('Posterior Probability')
plt.savefig(stat.getDownloadsTab() + '/Posterior Probability by Sheet Number.png')