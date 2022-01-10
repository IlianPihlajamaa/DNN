
import matplotlib.pyplot as plt
import glob, os
import numpy as np
from shutil import copyfile
i = 0
k_array = np.loadtxt("k_array.txt")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=12)
plt.figure()
filelist = glob.glob("Full Data\\S(k)\\*Sk*.txt")
np.random.shuffle(filelist)
for file in filelist:
    i += 1
    print(i)
    data = np.loadtxt(file)
    plt.plot(k_array, data, "o-", ms=3)
    if i > 5:
        break
    # if np.max(data) > 9 or np.min(data) < 0:
    #     continue
    # rho_string = file.split("%5C")[2].split("_")[2][:-4]
    # for file in glob.glob("Cluster\\*%s*.txt" % rho_string):
    #     folder1 = file.split("%5C")[0].split("\\")[1]
    #     folder2 = file.split("%5C")[1]
    #     filename = file.split("%5C")[2]
    #     new_path = os.path.join(folder1, folder2, filename)
    #     copyfile(file, new_path)
plt.xlabel(r"$k$")
plt.ylabel(r"$S(k)$")
plt.savefig("RandomsampleSk.png", dpi=500)
plt.show()