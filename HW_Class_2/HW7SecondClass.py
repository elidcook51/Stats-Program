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

# tot = 0
# for i in range(len(ys)):
#     y = ys[i]
#     eta = etay[i]
#     kappa = kappay[i]
#     diff = eta - y
#     diff = diff * diff
#     tot += diff * kappa

# # print(tot)
# # print(np.sqrt(tot))

# # print(stat.varianceScoreDiscrete(etay, kappay))
# # print(stat.uncertaintyScoreDiscrete(etay, kappay, g))

# y = [0.4, 0.6]
# f0 = [0.5, 0.5]
# f1 = [0.3333, 0.6667]

# k = [f0[i] * (1 - g) + f1[i] * g for i in range(len(f0))]
# eta = [f1[i] * g / k[i] for i in range(len(f0))]

# k = np.array(k)
# eta = np.array(eta)

# CS = stat.calibrationScore(y, k, eta)
# print(CS)

# VS = np.sum(eta * (1 - eta) * k)
# print(k)
# print(eta)
# print(VS)

# probfA = [0, 0.4 ,1]
# probdA = [0, 0.8 ,1]
# probfB = [0, 0.5, 1]
# probdB = [0, 0.6667, 1]

# plt.scatter(x = probfA, y = probdA, color = 'gray', label = 'Radiologist A')
# plt.scatter(x = probfB, y = probdB, color = 'lightgray', label = 'Radiologist B')
# plt.plot(probfA, probdA, color = 'gray')
# plt.plot(probfB, probdB, color = 'lightgray')
# plt.legend()
# plt.title("ROC for Radiologist A and B")
# plt.ylabel("Probability of Detection")
# plt.xlabel("Probability of False Alarm")
# plt.savefig(stat.getDownloadsTab() + '/ROC for two radiologists.png')


y = [0, 0.02, 0.05, 0.1, 0.2, 0.3 ,0.4 ,0.5 ,0.6, 0.7, 0.8, 0.9, 1]
w0 = [208, 48, 660, 285, 495, 112, 99, 140, 15, 84, 36, 12, 1]
w1 = [ 1, 0, 20, 32, 92, 48, 49, 99, 22, 160, 73, 88, 37]

f0 = [w0[i] / np.sum(w0) for i in range(len(w0))]
f1 = [w1[i] / np.sum(w1) for i in range(len(w0))]

# for i in range(len(f0)):
#     print(y[i], np.round(f0[i], decimals=4), np.round(f1[i],decimals=4))

# plt.scatter(x = y, y = f0, color = 'gray', label = 'f0(y)')
# plt.scatter(x = y, y = f1, color = 'lightgray', label = 'f1(y)')
# plt.title('Conditional Probability Functions')
# plt.ylabel("Conditional Probability")
# plt.xlabel("Forecast Probability")
# plt.legend()
# plt.savefig(stat.getDownloadsTab() + '/Conditional Prob Function.png')

# print(np.sum(w1) / (np.sum(w0) + np.sum(w1)))

g = np.sum(w1) / (np.sum(w0) + np.sum(w1))

k = [f0[i] * (1 - g) + f1[i] * g for i in range(len(f0))]
eta = [f1[i] * g / k[i] for i in range(len(f0))]

# stat.printInLatexTable([y, f0, f1])

# plt.scatter(x = y, y = k, color = 'gray')
# plt.title("Expected Probability Function")
# plt.xlabel("Forecast Probability")
# plt.ylabel("Expected Probability")
# plt.savefig(stat.getDownloadsTab() + '/Expected Prob.png')

# stat.printInLatexTable([y, k, eta])

# plt.scatter(x = y, y = eta, color = 'gray')
# plt.title('Probability Calibration Function')
# plt.xlabel("Forecast Probability")
# plt.ylabel("Calibrated Forecast Probability")
# plt.savefig(stat.getDownloadsTab() +'/Probability Calibration Function weather.png')

# f1f0 = [f1[i] / f0[i] for i in range(len(f0))]
# plt.scatter(x = y, y = f1f0, color = 'gray')
# plt.title("f1 / f0 plotted for monotonicity")
# plt.xlabel("Forecast Probability")
# plt.ylabel("f1 / f0")
# plt.savefig(stat.getDownloadsTab() + '/f1f0.png')

probd = np.cumsum(f1[::-1])[::-1]
probd = [np.round(probd[i], decimals = 4) for i in range(len(probd))]
probd.append(0)
probf = np.cumsum(f0[::-1])[::-1]
probf = [np.round(probf[i], decimals = 4) for i in range(len(probf))]
probf.append(0)

# plt.scatter(x = probf, y = probd, color = 'gray')
# plt.plot(probf, probd, color = 'gray')
# plt.title("ROC for forecaster A")
# plt.xlabel("Probability of False Alarm")
# plt.ylabel("Probability of Detection")
# plt.savefig(stat.getDownloadsTab() + '/ROC for weather.png')


# stat.printInLatexTable([y, probd, probf])

print(stat.calibrationScore(y, eta, k))
print(stat.varianceScoreDiscrete(eta, k))
print(stat.uncertaintyScoreDiscrete(eta, k, g))

# print(probd)