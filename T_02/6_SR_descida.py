
import numpy as np
import matplotlib.pyplot as plt



x = [8.14191, 8.37587, 8.60984, 8.84381, 9.16286, 9.46064, 9.58825, 9.84349, 10.07746, 10.3327, 10.58794, 10.80064, 11.0346, 11.2473, 11.46, 11.6727, 11.8854, 12.11937, 12.35333, 9.01397, 9.31175]
y = [-0.56391, -0.69452, -0.81377, -0.95574, -1.14313, -1.30782, -1.38164, -1.52361, -1.66557, -1.79618, -1.94383, -2.06308, -2.19369, -2.31862, -2.43788, -2.55145, -2.67638, -2.80131, -2.92625, -1.0466, -1.22263]

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