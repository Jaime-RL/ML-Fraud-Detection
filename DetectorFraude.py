import math
import numpy as np
import pandas as pd
import random

#from sklearn.model_selection import train_test_split
#from imblearn.over_sampling import SMOTE  # Importar la función SMOTE
#from sklearn.preprocessing import StandardScaler  # Para escalar los datos

# Cargar los datos
datos = pd.read_csv('creditcard.csv', header=None)

'''
1. Dividiremos los datos en entrenamiento y prueba manualmente
    Barajamos los datos para evitar sesgos. 
    Separamos el 80% para entrenamiento y el 20% para prueba.
    Extraemos las características (X) y la variable objetivo (y).
'''

print(datos.head())  # Mostrar los primeros datos

# Mezclar los datos aleatoriamente (barajar)
# sample() baraja aleatoriamente las filas, y frac=1.0 asegura que se use el 100% de los datos
datos = datos.sample(frac=1, random_state=42).reset_index(drop=True)

# Separar características (X) y variable objetivo (y) usando operaciones 
X = datos.drop(datos.columns[-1], axis=1)  # Eliminar la última columna para obtener las características
y = datos[datos.columns[-1]]  # Seleccionar solo la última columna para la variable objetivo

print(X.head())  # Mostrar las primeras 5 características
print(y.head())  # Mostrar las primeras 5 etiquetas

# Barajar los datos       100% datos | =barajeo aleatorio | indice anterior no se añade
datos_barajeados = datos.sample(frac=1, random_state=42).reset_index(drop=True)

# Mostrar los primeros 5 datos para verificar
print(datos_barajeados.head())


# Dividir los datos en entrenamiento y prueba
# Calcular el número total de filas
n = len(datos)

# Dividir: 80% entrenamiento, 20% prueba
#  iloc permite seleccionar filas y columnas usando índices enteros, sin importar sus etiquetas
train_data = datos.iloc[:int(0.8 * n)]
test_data = datos.iloc[int(0.8 * n):]

# Separar características (X) y variable objetivo (y)
X_train = train_data.iloc[:, :-1].values  # Todas las columnas excepto la última
y_train = train_data.iloc[:, -1].values   # La última columna
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Mostrar tamaños de conjuntos
print("Tamaño de entrenamiento:", X_train.shape, y_train.shape)
print("Tamaño de prueba:", X_test.shape, y_test.shape)

'''
2. Aplicar técnicas de balanceo (como SMOTE) y escalado en el conjunto de entrenamiento (y transformar el de prueba con esos mismos parámetros).
    SMOTE: Técnica de sobremuestreo que crea muestras sintéticas de la clase minoritaria para equilibrar las clases.
    Escalado: Estandariza los datos para que tengan una media de 0 y una desviación estándar de 1.
'''

# Función para calcular la distancia entre dos vectores
def euclidean_distance(a, b):
    distance = 0
    for ai, bi in zip(a, b):
        distance += (ai - bi) ** 2
    return math.sqrt(distance)

# Función para encontrar el vecino más cercano de 'sample' en el conjunto 'X'
def find_nearest_neighbors(X, sample, k=5):
    min_dist = float('inf')  # valor infinito positivo
    nearest = []
    for neighbor_candidate in X:
        if np.array_equal(neighbor_candidate, sample):  # Compara si los arrays son iguales
            continue
        dist = euclidean_distance(sample, neighbor_candidate)
        if dist < min_dist:
            min_dist = dist
            nearest = [neighbor_candidate]  # reset nearest con el nuevo vecino más cercano
        elif dist == min_dist:  # Si la distancia es igual a la mínima, agrega al vecino
            nearest.append(neighbor_candidate)

    return nearest[:k]  # Devuelve los primeros k vecinos más cercanos

# Función para generar una muestra sintética entre sample y neighbor
def generate_synthetic_sample(sample, neighbor):
    lam = random.random()  # Número aleatorio entre 0 y 1
    synthetic_sample = []
    for i in range(len(sample)):
        value = sample[i] + lam * (neighbor[i] - sample[i])
        synthetic_sample.append(value)
    return synthetic_sample

# Función SMOTE
def smote():
    """
    X_minority: Conjunto de datos de la clase minoritaria (fraudulentas)
    k: Número de vecinos a considerar
    n_samples: Número total de muestras sintéticas a generar (si no se especifica, se generarán tantas como sea necesario para equilibrar las clases)
    """
    
    # Separar las transacciones fraudulentas y no fraudulentas
    X_minority = X_train[y_train == 1]
    X_majority = X_train[y_train == 0]

    synthetic_samples = []
    
    # El número de muestras que se generan igualan las clases
    n_samples = len(X_majority) - len(X_minority)

    for sample in X_minority:
        # Encontrar los k vecinos más cercanos (agregamos k a la llamada)
        neighbors = find_nearest_neighbors(X_minority, sample, k=5)
        
        for neighbor in neighbors:
            # Generar una nueva muestra sintética entre la muestra y su vecino
            synthetic_sample = generate_synthetic_sample(sample, neighbor)
            synthetic_samples.append(synthetic_sample)
        
        if len(synthetic_samples) >= n_samples:
            break
    
    return synthetic_samples

# Aplicar SMOTE para generar muestras sintéticas
synthetic_samples = smote()


# Convertir las muestras sintéticas en un DataFrame y etiquetarlas como fraudulentas
synthetic_samples_df = pd.DataFrame(synthetic_samples)
synthetic_samples_df['Class'] = 1

# Combinar las muestras sintéticas con las no fraudulentas
datos_balanceados = pd.concat([train_data[train_data.iloc[:, -1] == 0], synthetic_samples_df], ignore_index=True)

# Rebarajear los datos balanceados
datos_balanceados = datos_balanceados.sample(frac=1, random_state=42).reset_index(drop=True)

# Verificar el balance de clases después de SMOTE
print(datos_balanceados['Class'].value_counts())











'''
1. División de datos: Primero se hace la división entre los datos de entrenamiento y prueba, y luego se aplica SMOTE solo sobre el conjunto de entrenamiento.
2. SMOTE: Asegura que el conjunto de entrenamiento esté balanceado antes de entrenar el modelo.
3. Escalado: Se usa StandardScaler para estandarizar los datos, ya que el modelo puede verse afectado por las diferencias de escala en las características.
4. Tamaño de los conjuntos: Muestra los tamaños de los conjuntos de entrenamiento y prueba después de realizar estos procesos.
'''

'''
# Cargar los datos
datos = pd.read_csv('creditcard.csv')

# Eliminar la columna objetivo
X = datos.drop(columns=['Class'])  

# Clase objetivo (fraude o no fraude)
y = datos['Class']  

#1. Divide los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#2. Aplica SMOTE solo al conjunto de entrenamiento
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Verifica el balanceo de las clases después de SMOTE
print("Distribución después de SMOTE en el conjunto de entrenamiento:")
print(y_resampled.value_counts())

# Escala los datos en una distribucion normla para mejorar el rendimiento del modelo
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)
X_test_scaled = scaler.transform(X_test)  # Escala también los datos de prueba

#4. Muestra el tamaño del conjunto de entrenamiento y prueba
print("Tamaño del conjunto de entrenamiento:", X_resampled.shape)
print("Tamaño del conjunto de prueba:", X_test_scaled.shape)


# Entrenar un modelo de regresión logística
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 1. Crear e instanciar el modelo de regresión logística
modelo = LogisticRegression(random_state=42)

# 2. Entrenar el modelo con los datos de entrenamiento
modelo.fit(X_train, y_train)

# 3. Hacer predicciones sobre el conjunto de prueba
y_pred = modelo.predict(X_test)

# 4. Evaluar el rendimiento del modelo

# 4.1. Accuracy (Exactitud)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 4.2. Reporte detallado con precisión, recall y F1-score
print("Reporte de clasificación:\n", classification_report(y_test, y_pred))

# 4.3. Matriz de confusión (para ver los verdaderos positivos, falsos positivos, etc.)
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
'''