import numpy as np 
import matplotlib.pyplot as plt 

t_eval = np.load("1t_eval.npy")
Pgp = np.load("1Pg+.npy")
Pgm = np.load("1Pg-.npy")

plt.plot(t_eval*1e6, Pgp, label="g+")
plt.plot(t_eval*1e6, Pgm, label="g-")

plt.legend()
plt.savefig("g+g-diff.png")
plt.show()