import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=[10, 10])

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=25)

k = np.linspace(0.2, 39.8, 100)
for i in range(1, 10):
    shap = np.loadtxt("shap_%i.txt"%i)
    plt.plot(k, shap)
plt.ylabel("std of Shapley values")
plt.xlabel("$kd$")
plt.savefig("shap_all.png")

plt.show()

shaps = np.zeros((9, 100))

for i in range(1, 10):
    shaps[i-1, :] = np.loadtxt("shap_%i.txt"%i)
    
plt.figure(figsize=[10, 10])
shapmean = np.mean(shaps, axis=0)
shapstd = np.std(shaps, axis=0)
plt.errorbar(k, shapmean, yerr=shapstd, capsize=4)
plt.ylabel("std of Shapley values (mean $\pm$ std)")
plt.xlabel("$kd$")
plt.savefig("shap_mean.png")
plt.show()

