# imports
import math
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Part 1 ----------------------------------------------------------------------------------
#-------------------------------------------------------------------------------- 20/9/2023
#------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
# 1.5.4
# Bode plots (Gain and Phase)

# Def K, Q, w0
P2, R2, R5, R3, R6, R11, C1, C2 = 10*10**3, 100*10**3, 100*10**3, 10*10**3, 10*10**3, 10*10**3, 4.7*10**-9, 4.7*10**-9
K, Q, w0 = 1, 1, 1/C1/R6

def gain_dB(gain):
    return 20*math.log10(abs(gain))

Amp_1 = 1.09
err_Amp_1 = 0.01
err_phase = 2
err_Amp_2 = 0.02

#-- HP ------------------------------------------------------------------------------------------------

#f_hp = [4.7*10**3,4.3*10**3,5*10**3,6*10**3,8*10**3,7*10**3,10*10**3,11*10**3,4.5*10**3,4*10**3,3.5*10**3,3*10**3,3.7*10**3,2.5*10**3,1*10**3,5.2*10**3]
#Amp_2_hp = [1.25,1.21,1.29,1.27,1.21,1.24,1.18,1.17,1.25,1.17,1.01,0.8,1.08,0.56,0.14,1.29]
#phase_hp = [66,75,61,45,30,36,22,20,70,84,100,115,94,130,160,56]
#w_hp = [f*2*math.pi for f in f_hp]
#gain_dB_hp = [gain_dB(abs(amp2/Amp_1)) for amp2 in Amp_2_hp]
#
##ajuste
#def transfer_function_dB(x, K, w0, Q):
#    return 20 * np.log10(np.abs(K * x**2 / (w0**2 + 1j * x * w0 / Q - x**2)))
#
#initial_guess = (1.0, w0, 1.0)  # (K, w0, Q)
#
#params, covariance = curve_fit(transfer_function_dB, w_hp, gain_dB_hp, p0=initial_guess)
#x_fit = np.linspace(min(w_hp), max(w_hp), 1000)
#y_fit = transfer_function_dB(x_fit, *params)
#
##  curva teórica
#w_values = np.linspace(0, 70*10**3, 10000)
#magnitude = np.abs(K*w_values**2 / (w0**2 + 1j * w_values * w0 / Q - w_values**2))
#magnitude_dB = 20 * np.log10(magnitude)
#phase_degrees = np.degrees(np.angle(K*w_values**2 / (w0**2 + 1j * w_values * w0 / Q - w_values**2)))+180
#
#
#plt.plot(w_values, magnitude_dB, label='Curva Teórica')
#plt.scatter(w_hp, gain_dB_hp, color='red', marker='o', label='Pontos Experimentais')
#plt.plot(x_fit, np.real(y_fit), 'g-', label='Ajuste aos pontos experimentais', color='green')
##plt.title('Magnitude da Função de Transferência e Pontos Experimentais')
#plt.xlabel('w [rad/s]')
#plt.ylabel('|T| [dB]')
#plt.xlim(5000, 70000)
#plt.ylim(-20, 5)
#plt.legend()
#plt.grid(True)
#plt.savefig('HP_gain.pdf')
#plt.show()
#
#
#plt.scatter(w_hp, phase_hp, c='green', marker='o', label = 'Pontos Experimentais')
#plt.plot(w_values, phase_degrees, label = 'Curva Teórica')
#plt.xlabel('w [rad/s]')
#plt.ylabel('Phase [\u00b0]')
##plt.title('Filtro highpass - ganho em dB')
#plt.legend()
#plt.grid(True)
#plt.savefig('HP_phase.pdf')
#plt.show()

##-- BP -----------------------------------------------------------------------------------------------

#f_bp = [3.7*10**3, 3.5*10**3, 3.4*10**3, 3*10**3, 2*10**3, 1.5*10**3, 1*10**3, 3.9*10**3, 4*10**3, 4.2*10**3, 5*10**3, 6*10**3, 7*10**3, 8*10**3, 9*10**3, 12*10**3, 15*10**3]
#Amp_2_bp = [1.21,1.15,1.13,1.05,0.72,0.56,0.4,1.21,1.21,1.17,1.05,0.88,0.8,0.7,0.64,0.408,0.4]
#phase_bp = [-174,-168,-163,-151,-122,-112,-100,-180,176,170,152,138,130,122,118,112,109]
#w_bp = [f*2*math.pi for f in f_bp]
#gain_dB_bp = [gain_dB(abs(amp2/Amp_1)) for amp2 in Amp_2_bp]
##err_gain_dB_bp = abs(20*math.log10(err_Amp_2))
#
##ajuste
#def transfer_function_dB(x, K, w0, Q):
#    return 20 * np.log10(np.abs(-x*1j*K*w0 / (w0**2 + 1j * x * w0 / Q - x**2)))
#
#initial_guess = (1.0, w0, 1.0)  # (K, w0, Q)
#
#params, covariance = curve_fit(transfer_function_dB, w_bp, gain_dB_bp, p0=initial_guess)
#x_fit = np.linspace(min(w_bp), max(w_bp), 1000)
#y_fit = transfer_function_dB(x_fit, *params)
#
##  curva teórica
#w_values = np.linspace(0, 95*10**3, 10000)
#magnitude = np.abs(-w_values*1j*K*w0 / (w0**2 + 1j * w_values * w0 / Q - w_values**2))
#magnitude_dB = 20 * np.log10(magnitude)
#phase_degrees = np.degrees(np.angle(-w_values*1j*K*w0 / (w0**2 + 1j * w_values * w0 / Q - w_values**2)))
#
#plt.plot(w_values, magnitude_dB, label='Curva Teórica')
#plt.scatter(w_bp, gain_dB_bp, color='red', marker='o', label='Pontos Experimentais')
#plt.plot(x_fit, np.real(y_fit), 'g-', label='Ajuste aos pontos experimentais', color='green')
#plt.xlabel('w [rad/s]')
#plt.ylabel('|T| [dB]')
#plt.xlim(5000, 95000)
#plt.ylim(-15, 2.5)
#plt.legend()
#plt.grid(True)
#plt.savefig('BP_gain.pdf')
#plt.show()
#
#plt.scatter(w_bp, phase_bp, c='green', marker='o', label = 'Pontos Experimentais')
#plt.plot(w_values, phase_degrees, label = 'Curva Teórica')
#plt.xlabel('w [rad/s]')
#plt.ylabel('Phase [\u00b0]')
#plt.xlim(5000, 95000)
##plt.ylim(-15, 5)
##plt.title('Filtro highpass - ganho em dB')
#plt.legend()
#plt.grid(True)
#plt.savefig('BP_phase.pdf')
#plt.show()


##-- LP --------------------------------------------------------------------------------------------

f_lp = [8.8*10**3, 7*10**3, 6*10**3, 5*10**3, 4.5*10**3, 4*10**3, 3.8*10**3, 3.6*10**3, 3.3*10**3, 3*10**3, 2.6*10**3, 2.3*10**3, 2*10**3, 1.5*10**3, 1.2*10**3, 0.5*10**3]
Amp_2_lp = [0.32,0.48,0.64,0.88,1.05,1.17,1.21,1.27,1.31,1.37,1.37,1.33,1.29,1.23,1.21,1.21]
phase_lp = [-150,-140,-132,-117,-108,-94,-88,-83,-72,-61,-50,-41,-34,-25,-18,-8]
w_lp = [f*2*math.pi for f in f_lp]
gain_dB_lp = [gain_dB(abs(amp2/Amp_1)) for amp2 in Amp_2_lp]

#ajuste
def transfer_function_dB(x, K, w0, Q):
    return 20 * np.log10(np.abs(K*w0**2 / (w0**2 + 1j * x * w0 / Q - x**2)))

initial_guess = (1.0, w0, 1.0)  # (K, w0, Q)

params, covariance = curve_fit(transfer_function_dB, w_lp, gain_dB_lp, p0=initial_guess)
x_fit = np.linspace(min(w_lp), max(w_lp), 1000)
y_fit = transfer_function_dB(x_fit, *params)
#
#  curva teórica
w_values = np.linspace(0, 56*10**3, 10000)
magnitude = np.abs(K*w0**2 / (w0**2 + 1j * w_values * w0 / Q - w_values**2))
magnitude_dB = 20 * np.log10(magnitude)
phase_degrees = np.degrees(np.angle(K*w0**2 / (w0**2 + 1j * w_values * w0 / Q - w_values**2)))

plt.plot(w_values, magnitude_dB, label='Curva Teórica')
plt.scatter(w_lp, gain_dB_lp, color='red', marker='o', label='Pontos Experimentais')
plt.plot(x_fit, np.real(y_fit), 'g-', label='Ajuste aos pontos experimentais', color='green')
plt.xlabel('w [rad/s]')
plt.ylabel('|T| [dB]')
plt.xlim(0, 56000)
plt.ylim(-12, 2.5)
plt.legend()
plt.grid(True)
plt.savefig('LP_gain.pdf')
plt.show()

plt.scatter(w_lp, phase_lp, c='green', marker='o', label = 'Pontos Experimentais')
plt.plot(w_values, phase_degrees, label = 'Curva Teórica')
plt.xlabel('w [rad/s]')
plt.ylabel('Phase [\u00b0]')
plt.xlim(0000, 56000)
#plt.ylim(-15, 5)
#plt.title('Filtro highpass - ganho em dB')
plt.legend()
plt.grid(True)
plt.savefig('LP_phase.pdf')
plt.show()

print("Parâmetros Ajustados:")
print("K:", params[0])
print("w0:", params[1])
print("Q:", params[2])