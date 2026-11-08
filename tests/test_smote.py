"""Tests de la implementación de SMOTE desde cero (fraud_detection/fraud_detector.py)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fraud_detection"))
from fraud_detector import balancear, smote  # noqa: E402


@pytest.fixture
def minoritaria():
    return np.random.default_rng(0).normal(size=(30, 4))


def test_numero_y_forma_de_muestras(minoritaria):
    sinteticas = smote(minoritaria, n_samples=100, k=5)
    assert sinteticas.shape == (100, 4)


def test_muestras_dentro_del_rango_de_la_clase(minoritaria):
    # Cada muestra es una interpolación entre dos puntos reales,
    # así que no puede salirse del rango de cada característica.
    sinteticas = smote(minoritaria, n_samples=500, k=5)
    assert np.all(sinteticas >= minoritaria.min(axis=0) - 1e-12)
    assert np.all(sinteticas <= minoritaria.max(axis=0) + 1e-12)


def test_interpolacion_sobre_el_segmento():
    # Con dos puntos y k=1, todo punto sintético debe estar en el segmento que los une.
    a, b = np.array([0.0, 0.0]), np.array([2.0, 4.0])
    sinteticas = smote(np.vstack([a, b]), n_samples=50, k=1)
    lam = sinteticas[:, 0] / 2.0
    assert np.all((lam >= 0) & (lam <= 1))
    np.testing.assert_allclose(sinteticas, np.outer(lam, b))


def test_reproducible_con_la_misma_semilla(minoritaria):
    s1 = smote(minoritaria, 20, rng=np.random.default_rng(7))
    s2 = smote(minoritaria, 20, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(s1, s2)


def test_balancear_iguala_clases_y_conserva_originales():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(110, 3))
    y = np.array([0] * 100 + [1] * 10)
    X_bal, y_bal = balancear(X, y)

    assert (y_bal == 0).sum() == (y_bal == 1).sum() == 100
    # Las 110 filas originales siguen presentes (solo se añaden sintéticas)
    originales = {tuple(f) for f in X}
    assert originales <= {tuple(f) for f in X_bal}
