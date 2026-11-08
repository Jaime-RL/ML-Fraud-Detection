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
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
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
    return y_pred, y_score


def guardar_graficas(resultados, y_test, carpeta):
    """Guarda las curvas precision-recall y la matriz de confusión del mejor modelo.

    `resultados` es un dict {nombre_modelo: (y_pred, y_score)}.
    """
    import matplotlib

    matplotlib.use("Agg")  # sin ventana: solo se escriben ficheros
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)

    # Curvas precision-recall: la métrica clave cuando solo el 0,17 % es fraude
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for nombre, (_, y_score) in resultados.items():
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        ap = average_precision_score(y_test, y_score)
        ax.plot(recall, precision, label=f"{nombre} (AP = {ap:.3f})")
    ax.axhline(y_test.mean(), ls="--", c="grey", label=f"Chance (AP = {y_test.mean():.4f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves (test set)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(carpeta / "pr_curves.png", dpi=150)

    # Matriz de confusión del modelo con mayor PR-AUC
    mejor = max(resultados, key=lambda n: average_precision_score(y_test, resultados[n][1]))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, resultados[mejor][0], display_labels=["Legit", "Fraud"], cmap="Blues", ax=ax
    )
    ax.set_title(mejor, fontsize=10)
    fig.tight_layout()
    fig.savefig(carpeta / "confusion_matrix.png", dpi=150)
    print(f"Gráficas guardadas en {carpeta}/")


def main(ruta_csv, carpeta_graficas=None):
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
    resultados = {}
    for nombre, modelo in modelos.items():
        modelo.fit(X_bal, y_bal)
        resultados[nombre] = evaluar(nombre, modelo, X_test, y_test)

    # Comparación: nuestra implementación de SMOTE frente a la de imbalanced-learn
    try:
        from imblearn.over_sampling import SMOTE

        X_ref, y_ref = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train, y_train)
        ref = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE).fit(X_ref, y_ref)
        resultados["Regresión logística + SMOTE (imblearn)"] = evaluar(
            "Regresión logística + SMOTE (imblearn, referencia)", ref, X_test, y_test
        )
    except ImportError:
        print("imbalanced-learn no instalado: se omite la comparación con SMOTE de referencia.")

    if carpeta_graficas:
        guardar_graficas(resultados, y_test, carpeta_graficas)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="creditcard.csv", help="Ruta a creditcard.csv")
    parser.add_argument("--plots", metavar="CARPETA", help="Guarda curvas PR y matriz de confusión")
    args = parser.parse_args()
    main(args.data, args.plots)
