
import numpy as np
import matplotlib.pyplot as plt

x = [-1.94977, -1.71698, -1.50535, -1.27256, -0.95512, -0.74349, -0.5107, -0.32023, -0.12977, 0.12419, 0.3993, 0.63209, 0.90721, 1.11884, 1.33046, 1.60558, 1.83837, 2.07116, 2.32512, 1.45744, 0.48395, 0.23]
y = [-2.8633, -2.71538, -2.57316, -2.43662, -2.27163, -2.12372, -1.9758, -1.86202, -1.74255, -1.60601, -1.44671, -1.27035, -1.11675, -0.96883, -0.8778, -0.68437, -0.54215, -0.4113, -0.252, -0.78678, -1.37276, -1.52636]
slope, intercept = np.polyfit(x, y, 1)

plt.scatter(x, y, label='Dados', color='blue')

plt.plot(x, slope*np.array(x) + intercept, label=f'Ajuste Linear (y = {slope:.3f}x + {intercept:.3f})', color='red')

equacao_ajuste = f'y = {slope:.3f}x + {intercept:.3f}'
plt.text(1, 7, equacao_ajuste, fontsize=12, color='black')

plt.xlabel('$t$ [$\mu s$]') 
plt.ylabel('$V_0$ [V]')
plt.legend()

# Exibindo o gráfico
plt.show()


















