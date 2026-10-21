"""Predicción de abandono de clientes (IBM Telco Customer Churn).

Compara regresión logística y Random Forest sobre un dataset desbalanceado,
aplicando SMOTE únicamente al conjunto de entrenamiento.
"""
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Cargar y preprocesar los datos
# ---------------------------------
datos = pd.read_csv('Telco-Customer-Churn.csv')

# TotalCharges viene como texto y contiene espacios en blanco en clientes nuevos
datos['TotalCharges'] = pd.to_numeric(datos['TotalCharges'], errors='coerce')
datos.dropna(inplace=True)
datos.drop(['customerID'], axis=1, inplace=True)

# Variables categóricas -> variables dummy
columnas = datos.select_dtypes(include='object').columns.drop('Churn')
datos = pd.get_dummies(datos, columns=columnas, drop_first=True, dtype=int)
datos['Churn'] = datos['Churn'].map({'Yes': 1, 'No': 0})

X = datos.drop('Churn', axis=1)
y = datos['Churn']

# 2. Dividir en entrenamiento y prueba (estratificado para mantener la proporción de clases)
# ----------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Balancear SOLO el conjunto de entrenamiento con SMOTE
# --------------------------------------------------------
# Si se balancea antes de dividir, muestras sintéticas derivadas de datos de test
# acaban en entrenamiento y las métricas salen infladas (data leakage).
print("Antes del balanceo:\n", y_train.value_counts())
X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
print("Después del balanceo:\n", y_train.value_counts())

# 4. Escalar los datos
# --------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Entrenar y evaluar los modelos
# ---------------------------------
modelos = {
    "Regresión Logística": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
}

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    print("=" * 45)
    print(f"Evaluación del modelo: {nombre}")
    print(confusion_matrix(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, digits=3))
