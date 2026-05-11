import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Cargar y Preprocesar los Datos
# ---------------------------------

# Cargar los datos desde un archivo CSV
datos = pd.read_csv('score copy.csv')

# Aquellos cuyo score > 50, se les asigna 1, de lo contrario 0
for index, row in datos.iterrows():
    if row['Scores'] > 50:
        datos.at[index, 'Scores'] = 1
    else:
        datos.at[index, 'Scores'] = 0

# Separar las características (x) y las etiquetas (y)
x = datos.Hours
y = datos.Scores

# Función sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Función de coste para la regresión logística
def funcion_coste_logistico(m, b, points):
    """
    Calcula el coste logístico (función de pérdida) para un conjunto de puntos dado.

    Args:
        m (float): Pendiente de la recta de regresión logística.
        b (float): Intersección de la recta de regresión logística.
        points (list of tuples): Lista de puntos (x, y) donde x es la característica y y es la etiqueta.

    Returns:
        float: El coste logístico promedio para el conjunto de puntos dado.
    """
    error_total = 0
    for i in range(len(points)):
        error_total += - y * np.log(sigmoid(m*x + b)) - (1 - y) * np.log(1 - sigmoid(m*x + b))
    return error_total / float(len(points))

# Función de descenso de gradiente para la regresión logística
def descenso_gradiente_logistico(m_actual, b_actual, points, learning_rate):
    """
    Realiza el descenso de gradiente para la regresión logística.
    Args:
        m_actual (float): Valor actual de la pendiente.
        b_actual (float): Valor actual de la intersección.
        points (DataFrame): Conjunto de datos que contiene las características y etiquetas.
        learning_rate (float): Tasa de aprendizaje para el descenso de gradiente.
    Returns:
        tuple: Nuevos valores de la pendiente (m_nuevo) y la intersección (b_nuevo).
    """
    m_gradiente = 0
    b_gradiente = 0

    n = len(points)
    
    for i in range(n):
        x = points.iloc[i].Hours
        y = points.iloc[i].Scores
        
        m_gradiente += -(2/n) * x * (y - sigmoid(m_actual*x + b_actual))
        b_gradiente += -(2/n) * (y - sigmoid(m_actual*x + b_actual))
        
    m_nuevo = m_actual - learning_rate * m_gradiente
    b_nuevo = b_actual - learning_rate * b_gradiente
    
    return m_nuevo, b_nuevo

# 2. Definir los Hiperparámetros
# ------------------------------
learning_rate = 0.01
iteraciones = 1000
m = 0
b = 0

# 3. Entrenar el Modelo
# ---------------------

# Realizar el entrenamiento del modelo usando descenso de gradiente
for i in range(iteraciones):
    m, b = descenso_gradiente_logistico(m, b, datos, learning_rate)
    if i % 50 == 0:
        print(f'Iteración {i}: Error = {funcion_coste_logistico(m, b, datos)}')
        plt.scatter(x, y, label='Datos')
        x_vals = np.linspace(x.min(), x.max(), 100)
        y_vals = sigmoid(m * x_vals + b)
        plt.plot(x_vals, y_vals, color='red', label='Sigmoide')
        plt.xlabel('Hours')
        plt.ylabel('Scores')
        plt.title(f'Iteración {i}: Hours vs Scores')
        plt.legend()
        plt.pause(0.1)
        plt.clf()
            
# 4. Evaluar el Modelo
# -----------------

# Realizar predicciones usando el modelo entrenado
predicciones = sigmoid(m*x + b)
predicciones = [1 if p >= 0.5 else 0 for p in predicciones]

# Calcular y mostrar la precisión del modelo
from sklearn.metrics import accuracy_score
print(f'Precisión: {accuracy_score(y, predicciones)}')


