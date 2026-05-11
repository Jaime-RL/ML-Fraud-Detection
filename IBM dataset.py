import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar y Preprocesar los Datos
# ---------------------------------

# Cargar los datos
datos = pd.read_csv('Telco-Customer-Churn.csv')

# Eliminar los datos faltantes
datos.dropna(inplace=True)

# Eliminar columnas no necesarias
datos.drop(['customerID'], axis=1, inplace=True)
print(datos.head())
print(datos.info())

# Identificar las variables a convertir en variables dummy
columnas = datos.select_dtypes(include='object').columns
columnas = columnas.drop('Churn')
print(columnas)

# Convertir variables categóricas a variables dummy
datos = pd.get_dummies(datos, columns=columnas ,drop_first=True, dtype=int)

# Convertir la variable objetivo a numérica
datos['Churn'] = datos['Churn'].map({'Yes': 1, 'No': 0})

# Mostrar numero de Churn antes del balanceo
print("Antes del balanceo:")
print(datos['Churn'].value_counts())

# Aplicar la técnica de balanceo SMOTE
from imblearn.over_sampling import SMOTE

smote = SMOTE()

X_smote, y_smote = smote.fit_resample(datos.drop('Churn', axis=1), datos['Churn'])

# Convertir los datos balanceados a un DataFrame
datos_balanceados = pd.DataFrame(X_smote, columns=datos.drop('Churn', axis=1).columns)
datos_balanceados['Churn'] = y_smote

# Mostrar numero de Churn después del balanceo
print("Después del balanceo:")
print(datos_balanceados['Churn'].value_counts())


# 2. Dividir los Datos en Entrenamiento y Prueba
# ----------------------------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(datos.drop('Churn', axis=1), datos['Churn'], test_size=0.2)

# 3. Escalar los Datos
# --------------------

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Crear y Entrenar el Modelo
# -----------------------------

from sklearn.linear_model import LogisticRegression

modelo = LogisticRegression()
modelo.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier

modelo_rf = RandomForestClassifier()
modelo_rf.fit(X_train, y_train)

# 5. Predecir y Evaluar el Modelo
# -------------------------------

modelo.pred = modelo.predict(X_test)

modelo_rf.pred = modelo_rf.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score

# Evaluar el Modelo de Regresión Logística
print("Evaluación del Modelo de Regresión Logística")
print(confusion_matrix(y_test, modelo.pred))
print(f'Precisión del modelo de Regresión Logística: {accuracy_score(y_test, modelo.pred)}')

print("=============================================")

# Evaluar el Modelo Random Forest
print("Evaluación del Modelo Random Forest")
print(confusion_matrix(y_test, modelo_rf.pred))
print(f'Precisión del modelo Random Forest: {accuracy_score(y_test, modelo_rf.pred)}')


'''
1 - Limpieza de Datos: Asegúrate de que los datos no tengan valores nulos o inconsistentes.
2 - Análisis Exploratorio de Datos (EDA): Realiza un análisis para entender mejor las relaciones entre las variables.
3 - Selección de Características: Elige las características más relevantes para el modelo.
4 - Ingeniería de Características: Crea nuevas características a partir de las existentes si es posible.
5 - Normalización/Estandarización: Normaliza o estandariza los datos si es necesario.
6 - División de Datos: Divide los datos en conjuntos de entrenamiento y prueba.
7 - Optimización de Hiperparámetros: Utiliza técnicas como Grid Search o Random Search para encontrar los mejores hiperparámetros.
8 - Validación Cruzada: Utiliza validación cruzada para evaluar el rendimiento del modelo.
'''


# 1. Cargar y Preprocesar los Datos
# ---------------------------------

# 2. Dividir los Datos en Entrenamiento y Prueba
# ----------------------------------------------

# 3. Escalar los Datos
# --------------------

# 4. Crear y Entrenar el Modelo
# -----------------------------

# 5. Predecir y Evaluar el Modelo
# -------------------------------