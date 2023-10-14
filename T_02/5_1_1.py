import numpy as np
import matplotlib.pyplot as plt

# Def vars
R = 12000 
Rf = 4 * R
S1 = 5
S2 = 5
S3 = 5
S4 = 5

def paralelo(r1, r2):
    return 1/(1/r1+1/r2)

def calcular_R_eq1(R, Ri):
    return R +  paralelo(2 *R,Ri)

def calcular_R_eq2(R, Ri):
    Req1 = calcular_R_eq1(R, Ri)
    return R +  paralelo(2 *R, Req1)

def calcular_R_eq3(R, Ri):
    Req2 = calcular_R_eq2(R, Ri)
    return R +  paralelo(2 *R,Req2)

def calcular_V0_1(R1, R, Rf, S1, S2, S3, S4):
    Req1 = calcular_R_eq1(R, R1)
    Req2 = calcular_R_eq2(R, R1)
    Req3 = calcular_R_eq3(R, R1)
    
    V0 = -Rf * (S1 / (8 * (R1 + R)) + (1/4) * (S2 / (2 * R + paralelo(2*R, calcular_R_eq1(R, R1))))*calcular_R_eq1(R,R1)/(2*R+calcular_R_eq1(R,R1)) +
         (1/2) * (S3 / (2 * R + paralelo(2*R, calcular_R_eq2(R, R1))))* (calcular_R_eq2(R,R1)/(2*R + calcular_R_eq2(R,R1))) + (S4 / (2 * R + paralelo(2*R, calcular_R_eq3(R, R1))))*(calcular_R_eq3(R,R1)/(2*R + calcular_R_eq3(R,R1))))
    
    return V0

# Crie uma lista de valores para R1/R
R1_values = np.linspace(0.00001, 2*48000, 10000)  # 1000 valores de 0 a 2000

# Calcule V0_1 para cada valor de R1/R
V0_1_values = [calcular_V0_1(R1, R, Rf, S1, S2, S3, S4) for R1 in R1_values]

R1_R_values = [R1/R for R1 in R1_values]

## Plote os resultados
#plt.plot(R1_R_values, V0_1_values)
#plt.xlabel(f'$R_1/R$')
#plt.ylabel(f'$V_0(R_1/R) [V]$')
#plt.grid(True)
#plt.show()




# V_0_2

def calcular_V0_2(R2, R, Rf, S1, S2, S3, S4):
    Req1 = calcular_R_eq1(R, R2)
    Req2 = calcular_R_eq2(R, R2)
    Req3 = calcular_R_eq3(R, R2)
    
    V0 = -Rf * (S1 / (4 * (2*R+paralelo(2*R, calcular_R_eq1(R,R2))))*R2/(R2+2*R)*2*R/(2*R+paralelo(2*R, calcular_R_eq1(R,R2))) + (1/8) * (S2 / (R2 + R)) +
         (1/2) * (S3 / (2 * R + paralelo(2*R, calcular_R_eq1(R, R2))))* (calcular_R_eq1(R,R2)/(2*R + calcular_R_eq1(R,R2))) + (S4 / (2 * R + paralelo(2*R, calcular_R_eq2(R, R2))))*(calcular_R_eq2(R,R2)/(2*R + calcular_R_eq2(R,R2))))
    
    return V0

# Crie uma lista de valores para R1/R
R2_values = np.linspace(0.00001, 2*48000, 10000)  # 1000 valores de 0 a 2000

# Calcule V0_1 para cada valor de R1/R
V0_2_values = [calcular_V0_2(R2, R, Rf, S1, S2, S3, S4) for R2 in R2_values]

R2_R_values = [R2/R for R2 in R2_values]

## Plote os resultados
#plt.plot(R2_R_values, V0_2_values, color='red')
#plt.xlabel(f'$R_2/R$')
#plt.ylabel(f'$V_0(R_2/R) [V]$')
#plt.grid(True)
#plt.show()

# V_0_3

def calcular_V0_3(R3, R, Rf, S1, S2, S3, S4):
    Req1 = calcular_R_eq1(R, R3)
    Req2 = calcular_R_eq2(R, R3)
    Req3 = calcular_R_eq3(R, R3)
    
    V0 = -Rf * (S1 / (2*(2*R+paralelo(2*R, calcular_R_eq2(R,R3))))*R3/(R3+2*R)*2*R/(2*R+ calcular_R_eq1(R,R3))*2*R/(2*R+ calcular_R_eq2(R,R3)) + (1/2) * (S2 / (2*R + paralelo(2*R, calcular_R_eq1(R,R3))))*2*R/(2*R + paralelo(2*R, calcular_R_eq1(R,R3)))*R3/(R3+2*R) +
         (1/4) * (S3 / (R3+R)) + (S4 / (2 * R + paralelo(2*R, calcular_R_eq1(R, R3))))*(calcular_R_eq1(R,R3)/(2*R + calcular_R_eq1(R,R3))))
    
    return V0

# Crie uma lista de valores para R1/R
R3_values = np.linspace(0.00001, 2*48000, 10000)  # 1000 valores de 0 a 2000

# Calcule V0_1 para cada valor de R1/R
V0_3_values = [calcular_V0_3(R3, R, Rf, S1, S2, S3, S4) for R3 in R3_values]

R3_R_values = [R3/R for R3 in R3_values]

## Plote os resultados
#plt.plot(R3_R_values, V0_3_values, color='green')
#plt.xlabel(f'$R_3/R$')
#plt.ylabel(f'$V_0(R_3/R) [V]$')
#
#plt.grid(True)
#plt.show()

# V_0_4

def calcular_V0_4(R4, R, Rf, S1, S2, S3, S4):
    Req1 = calcular_R_eq1(R, R4)
    Req2 = calcular_R_eq2(R, R4)
    Req3 = calcular_R_eq3(R, R4)
    
    V0 = -Rf * (S1 / (2*R+paralelo(2*R, calcular_R_eq3(R,R4)))*R4/(R4+2*R)*2*R/(2*R+ calcular_R_eq1(R,R4))*2*R/(2*R+ calcular_R_eq2(R,R4))*2*R/(2*R+ calcular_R_eq3(R,R4)) + (S2 / (2*R + paralelo(2*R, calcular_R_eq2(R,R4))))*2*R/(2*R +  calcular_R_eq2(R,R4))*2*R/(2*R +  calcular_R_eq1(R,R4))*R4/(R4+2*R) +
         (S3 / (2*R+paralelo(2*R, calcular_R_eq1(R,R4))))*2*R/(2*R+calcular_R_eq1(R,R4))*R4/(R4+2*R) + 0.5*(S4 / (R4+R)))
    
    return V0

# Crie uma lista de valores para R1/R
R4_values = np.linspace(0.00001, 2*48000, 10000)  # 1000 valores de 0 a 2000

# Calcule V0_1 para cada valor de R1/R
V0_4_values = [calcular_V0_4(R4, R, Rf, S1, S2, S3, S4) for R4 in R4_values]

R4_R_values = [R4/R for R4 in R4_values]

## Plote os resultados
#plt.plot(R4_R_values, V0_4_values, color='brown')
#plt.xlabel(f'$R_4/R$')
#plt.ylabel(f'$V_0(R_4/R) [V]$')
#plt.grid(True)
#plt.show()


# Crie uma figura
plt.figure()

# Plote os resultados de V0_1 em função de R1/R
plt.plot(R1_R_values, V0_1_values, label=f'$R_i = R_1$', color='blue')

# Plote os resultados de V0_2 em função de R2/R
plt.plot(R2_R_values, V0_2_values, label=f'$R_i = R_2$', color='red')

# Plote os resultados de V0_3 em função de R3/R
plt.plot(R3_R_values, V0_3_values, label=f'$R_i = R_3$', color='green')

# Plote os resultados de V0_4 em função de R4/R
plt.plot(R4_R_values, V0_4_values, label=f'$R_i = R_4$', color='brown')

# Configuração de rótulos e título
plt.xlabel(f'$R_i/R$')
plt.ylabel(f'$V_0(R_i/R) [V]$')
plt.grid(True)

# Adicione uma legenda
plt.legend()

# Exibir o gráfico
plt.show()