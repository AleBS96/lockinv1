import numpy as np

# Crear una matriz de 2x2 con números complejos
matriz = np.array([[1+2j, 2+3j], [3+4j, 4+5j]])

real = np.real(matriz)
imaginaria = np.imag(matriz)

# Elevar al cuadrado cada elemento de la matriz
real_cuadrada = np.square(real)
imag_cuadrada = np.square(imaginaria)

resultado = np.sqrt(real_cuadrada + imag_cuadrada)

print(matriz)
print(real_cuadrada)
print(imag_cuadrada)
print(resultado)