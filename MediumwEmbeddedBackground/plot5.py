import numpy as np
import matplotlib.pyplot as plt
import torch
import gzip
import json


N_bkg = [0,        20,       80,       240,      440,     1000,    1600,    2000,    2400 ]
Loss =  [0.00777,  0.00977,  0.01013,  0.01126,  0.01045, 0.01138, 0.01110, 0.01179, 0.01159]
Loss2 = [0.00945,  0.01082,  0.01037,  0.01045,  0.01111, 0.01252, 0.01206, 0.01267, 0.01283]
Loss3 = [0.01188,  0.01197,  0.01211,  0.01162,  0.01282, 0.01206, 0.01427, 0.01269, 0.01346]



plt.plot( N_bkg, Loss, label='No Subtraction or Cut', color='black')
plt.plot(N_bkg[:len(Loss2)], Loss2, color='blue', label='With Background Subtraction')
plt.plot(N_bkg, Loss3, label='$k_T$ > 1.0', color='red')
plt.xlabel("$dN_{bkg}$/dy")
plt.ylabel("Average MSELoss")
plt.legend()
plt.show()