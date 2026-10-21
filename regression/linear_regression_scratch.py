"""Regresión lineal simple (Hours -> Scores) con descenso de gradiente implementado desde cero."""
import pandas as pd
import matplotlib.pyplot as plt

# 1. Cargar y Preprocesar los Datos
# ---------------------------------

# Cargar los datos
datos = pd.read_csv('score.csv')

def funcion_coste(m, b, points):
    """
    Calcula el error cuadrático medio (MSE) para una regresión lineal.
    Parámetros:
    m (float): Pendiente de la recta.
    b (float): Intersección de la recta.
    points (DataFrame): Conjunto de datos que contiene las características (Hours) y las etiquetas (Scores).
    Retorna:
    float: El error cuadrático medio (MSE) de la regresión lineal.
    """
    error_total = 0
    for i in range(len(points)):
        x = points.iloc[i].Hours
        y = points.iloc[i].Scores
        error_total += (y - (m*x + b))**2
    return error_total / float(len(points))


def descenso_gradiente(m_actual, b_actual, points, learning_rate):
    """
    Realiza una iteración del descenso de gradiente para ajustar los parámetros de una regresión lineal.
    Parámetros:
    m_actual (float): Valor actual de la pendiente de la recta.
    b_actual (float): Valor actual de la intersección de la recta.
    points (DataFrame): Conjunto de datos que contiene las características (Hours) y las etiquetas (Scores).
    learning_rate (float): Tasa de aprendizaje para el ajuste de los parámetros.
    Retorna:
    tuple: Nuevos valores de la pendiente (m_nuevo) y la intersección (b_nuevo) después de una iteración del descenso de gradiente.
    """
    m_gradiente = 0
    b_gradiente = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].Hours
        y = points.iloc[i].Scores

        m_gradiente += -(2/n) * x * (y - (m_actual*x + b_actual))
        b_gradiente += -(2/n) * (y - (m_actual*x + b_actual))

    m_nuevo = m_actual - learning_rate * m_gradiente
    b_nuevo = b_actual - learning_rate * b_gradiente

    return m_nuevo, b_nuevo


# 2. Definir los Hiperparámetros
# ------------------------------
learning_rate = 0.0001
iteraciones = 1000
m = 0
b = 0

# 3. Entrenar el Modelo
# ---------------------

for i in range(iteraciones):
    m, b = descenso_gradiente(m, b, datos, learning_rate)
    if i % 50 == 0:
        print(f'Iteración {i}: Error = {funcion_coste(m, b, datos)}')
        plt.scatter(datos.Hours, datos.Scores)
        plt.plot(datos.Hours, m * datos.Hours + b, color='red')
        plt.xlabel('Hours')
        plt.ylabel('Scores')
        plt.title(f'Iteración {i}: Hours vs Scores')
        plt.pause(0.1)
        plt.clf()

print(f'Pendiente (m) = {m:.4f}, intersección (b) = {b:.4f}')
