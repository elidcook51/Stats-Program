import Stats_Functions as stat
import numpy as np
import matplotlib.pyplot as plt
import math


nums = np.array([15,17,19])
#nums = np.arange(0,36,0.01)
sigma = 3
mu0 = 18
mu1 = 16

pix = np.power(1 + 0.5835 * np.exp(0.2222 * nums -3.7778) , -1)
print(pix)
#plt.scatter(x = nums, y = pix, color = 'lightgrey')
#plt.title('Posterior probability of W given a realization x')
#plt.xlabel("Realization of x")
#plt.ylabel('Posterior probability of W')
#plt.savefig(stat.getDownloadsTab() + '/SYS5581 Posterior Probability Graph')
#plt.show()

#f0 = stat.normalDensity(nums, mu0, sigma)
#f1 = stat.normalDensity(nums, mu1, sigma)

#plt.scatter(x = nums, y = f0, color = 'lightgrey', label = 'f0')
#plt.scatter(x = nums, y = f1, color = 'grey', label = 'f1')
#plt.title('Conditional density functions plotted on (0, 36)')
#plt.legend()
#plt.xlabel("Realization of X")
#plt.ylabel("Density of probability")
#plt.show()
#plt.savefig(stat.getDownloadsTab() + '/SYS5581 HW6 Conditional Density')
