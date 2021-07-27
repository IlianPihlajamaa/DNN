# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 11:27:14 2021

@author: s158686
"""

import matplotlib.pyplot as plt
import numpy as np

t = 10**np.linspace(-10, 10, 214)
plt.plot(np.log(t), np.log(t), label="logt")
plt.plot(np.log(t), np.log(t+1), label = "logt+1")
plt.plot(np.log(t), np.log(np.log(t+1)+1), label="log(log(t+1)+1)")
plt.legend()
plt.show()
