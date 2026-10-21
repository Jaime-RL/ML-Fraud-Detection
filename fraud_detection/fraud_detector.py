"""Detección de fraude en transacciones con tarjeta de crédito.

Dataset: "Credit Card Fraud Detection" (Kaggle, ULB) -> creditcard.csv
284.807 transacciones, de las cuales solo ~0,17 % son fraudulentas.

Pipeline:
    1. División estratificada entrenamiento/prueba (80/20).
    2. Escalado con StandardScaler ajustado solo con entrenamiento.
    3. Balanceo con SMOTE implementado desde cero (y comparación con imblearn).
    4. Entrenamiento de regresión logística y Random Forest.
    5. Evaluación con métricas adecuadas para clases desbalanceadas
       (precision, recall, F1 y área bajo la curva precision-recall).
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def smote(X_minority, n_samples, k=5, rng=None):
    """Genera `n_samples` muestras sintéticas de la clase minoritaria (SMOTE).

    Para cada muestra sintética:
        1. Se elige al azar una muestra real x de la clase minoritaria.
        2. Se elige al azar uno de sus k vecinos más cercanos x_nn.
        3. Se interpola: x_new = x + lambda * (x_nn - x), con lambda ~ U(0, 1).

    Args:
        X_minority (ndarray): Muestras de la clase minoritaria, shape (m, n_features).
        n_samples (int): Número de muestras sintéticas a generar.
        k (int): Número de vecinos más cercanos considerados.
        rng (np.random.Generator): Generador aleatorio (para reproducibilidad).

    Returns:
        ndarray: Muestras sintéticas, shape (n_samples, n_features).
    """
    rng = rng or np.random.default_rng(RANDOM_STATE)
    k = min(k, len(X_minority) - 1)

    # k + 1 porque el vecino más cercano de cada punto es él mismo
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_minority)
    _, vecinos = nn.kneighbors(X_minority)
    vecinos = vecinos[:, 1:]

    base = rng.integers(0, len(X_minority), size=n_samples)
    elegido = vecinos[base, rng.integers(0, k, size=n_samples)]
    lam = rng.random((n_samples, 1))

    return X_minority[base] + lam * (X_minority[elegido] - X_minority[base])


def balancear(X, y, k=5):
    """Iguala el número de muestras de ambas clases añadiendo fraudes sintéticos."""
    X_min = X[y == 1]
    n_nuevas = int((y == 0).sum() - (y == 1).sum())
    X_sint = smote(X_min, n_nuevas, k=k)
    X_bal = np.vstack([X, X_sint])
    y_bal = np.concatenate([y, np.ones(n_nuevas, dtype=y.dtype)])
    orden = np.random.default_rng(RANDOM_STATE).permutation(len(y_bal))
    return X_bal[orden], y_bal[orden]


def evaluar(nombre, modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    y_score = modelo.predict_proba(X_test)[:, 1]
    print("=" * 60)
    print(f"Modelo: {nombre}")
    print("Matriz de confusión [[TN FP] [FN TP]]:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=3, target_names=["Legítima", "Fraude"]))
    print(f"Average precision (PR-AUC): {average_precision_score(y_test, y_score):.4f}")


def main(ruta_csv):
    datos = pd.read_csv(ruta_csv)
    X = datos.drop(columns=["Class"]).values
    y = datos["Class"].values
    print(f"Transacciones: {len(y)} | Fraudes: {y.sum()} ({100 * y.mean():.3f} %)")

    # 1. División estratificada: mantiene la proporción de fraudes en ambos conjuntos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # 2. Escalado (SMOTE usa distancias, así que conviene escalar antes)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 3. Balanceo SOLO del conjunto de entrenamiento
    X_bal, y_bal = balancear(X_train, y_train)
    print(f"Tras SMOTE -> legítimas: {(y_bal == 0).sum()} | fraudes: {(y_bal == 1).sum()}")

    # 4 y 5. Entrenamiento y evaluación
    modelos = {
        "Regresión logística + SMOTE": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest + SMOTE": RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE
        ),
    }
    for nombre, modelo in modelos.items():
        modelo.fit(X_bal, y_bal)
        evaluar(nombre, modelo, X_test, y_test)

    # Comparación: nuestra implementación de SMOTE frente a la de imbalanced-learn
    try:
        from imblearn.over_sampling import SMOTE

        X_ref, y_ref = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train, y_train)
        ref = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE).fit(X_ref, y_ref)
        evaluar("Regresión logística + SMOTE (imblearn, referencia)", ref, X_test, y_test)
    except ImportError:
        print("imbalanced-learn no instalado: se omite la comparación con SMOTE de referencia.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="creditcard.csv", help="Ruta a creditcard.csv")
    main(parser.parse_args().data)
