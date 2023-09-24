# imports
import math
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# Def K, Q, w0
C1, C2, R2, R4, R5, R6, R11, P2 = 4.7e-9, 4.7e-9, 100e3, 10e3, 100e3, 10e3, 10e3, 10e3

w0 = np.sqrt(R5/(R2*R4*R11*C1*C2))
K = 1/(w0*P2*C1)
Q = w0*C1*R6

def T1_dB(x, K, w0, Q):
    return 20 * np.log10(np.abs(-K*w0*1j*x / (w0**2 + 1j * x * w0 / Q - x**2)))

def T2_dB(x, K, w0, Q):
    return 20 * np.log10(np.abs(K*w0**2 / (w0**2 + 1j * x * w0 / Q - x**2)))

def T3_dB(x, K, w0, Q):
    return 20 * np.log10(np.abs(-K*w0**2 / (w0**2 + 1j * x * w0 / Q - x**2)))

#  curvas teóricas
w_values = np.linspace(0, 20000*6, 10000)
T1_magnitude_dB = T1_dB(w_values, K, w0, Q)
T2_magnitude_dB = T2_dB(w_values, K, w0, Q)
T3_magnitude_dB = T3_dB(w_values, K, w0, Q)

plt.plot(w_values, T1_magnitude_dB, label='Curva Teórica T1', color='green')
plt.plot(w_values, T2_magnitude_dB, label='Curva Teórica T2', linestyle='--', color='blue')
plt.plot(w_values, T3_magnitude_dB, label='Curva Teórica T3', linestyle=':', color='red')

plt.xlabel('w [rad/s]')
plt.ylabel('|T| [dB]')
plt.xlim(2500, 20000*6)
plt.ylim(-30, 5)
plt.xscale('log')
plt.legend()
plt.grid(True)
#plt.savefig('HP_gain.pdf')
plt.show()


