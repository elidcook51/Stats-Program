import Stats_Functions as stat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


doc = [0.5, 0.5, 0.1, 0.1, 0.3, 0.5, 0.5, 0.7, 0.3, 0.3, 0.7,0.5, 0.7, 0.9, 0.7, 0.1]
real = [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0]

ys = [0.1, 0.3, 0.5, 0.7, 0.9]

happen = [doc[i] for i in range(len(real)) if real[i] == 1]
nothappen = [doc[i] for i in range(len(real)) if real[i] == 0]

f0 = []
f1 = []
for y in ys:
    f1.append(len([happen[i] for i in range(len(happen)) if happen[i] == y]) / len(happen))
    f0.append(len([nothappen[i] for i in range(len(nothappen)) if nothappen[i] == y]) / len(nothappen))

# print(f0)
# print(f1)
# print(len(happen) / len(real))

g = 0.5
kappay = []
etay = []
for i in range(len(ys)):
    y = ys[i]
    f0y = f0[i]
    f1y = f1[i]
    kappa = f0y * (1 - g) + f1y * g
    kappay.append(kappa)
    eta = f1y * g / kappa
    etay.append(eta)

# print(kappay)
# print(etay)

# plt.scatter(x = ys, y = etay, color = 'gray', s = 25)
# plt.title('Probability calibration function')
# plt.xlabel("Forecast probability")
# plt.ylabel("Posterior probability")
# plt.savefig(stat.getDownloadsTab() + '/Probability calibration function.png')

# print(np.array(f0) / np.array(f1))

# probf = [1, 0.625, 0.375, 0, 0, 0]
# probd = [1, 1, 0.875, 0.625, 0.125, 0]
# plt.scatter(x = probf, y = probd, color = 'gray')
# plt.plot(probf, probd, color = 'gray', linestyle = '-', linewidth = 1)
# plt.title("Reciever Operator Characteristic")
# plt.xlabel("Probability of False Alarm")
# plt.ylabel("Probability of Detection")
# plt.savefig(stat.getDownloadsTab() + '/ROC.png')

tot = 0
for i in range(len(ys)):
    y = ys[i]
    eta = etay[i]
    kappa = kappay[i]
    diff = eta - y
    diff = diff * diff
    tot += diff * kappa

# print(tot)
# print(np.sqrt(tot))

print(stat.varianceScoreDiscrete(etay, kappay))
print(stat.uncertaintyScoreDiscrete(etay, kappay, g))