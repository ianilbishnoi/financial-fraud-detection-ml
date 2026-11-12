"""Fast tests on a small synthetic dataset shaped like the real one (Time, V1..V28, Amount, Class)."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

import fraud_detection as fd


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(
        n_samples=6000, n_features=28, n_informative=8, weights=[0.98, 0.02], flip_y=0, random_state=0
    )
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    rng = np.random.default_rng(0)
    df.insert(0, "Time", np.arange(len(df), dtype=float))
    df["Amount"] = rng.exponential(80, len(df))
    df["Class"] = y
    return df


def test_split_is_stratified_and_disjoint(data):
    X_train, X_val, X_test, y_train, y_val, y_test = fd.split(data)
    assert len(X_train) + len(X_val) + len(X_test) == len(data)
    assert not set(X_train.index) & set(X_test.index)
    rate = data["Class"].mean()
    for y in (y_train, y_val, y_test):
        assert abs(y.mean() - rate) < 0.005


def test_best_threshold_is_a_probability():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.6, 0.7, 0.9])
    t = fd.best_threshold(y, scores)
    assert 0.0 <= t <= 1.0
    assert t == pytest.approx(0.7)  # separates the classes perfectly


def test_evaluate_counts_confusion_matrix():
    m = fd.evaluate(np.array([0, 1, 1, 0]), np.array([0.1, 0.9, 0.4, 0.8]), threshold=0.5)
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert m["precision"] == 0.5 and m["recall"] == 0.5


def test_scaler_is_fit_on_training_data_only(data):
    X_train, *_ = fd.split(data)
    _, _, _, y_train, _, _ = fd.split(data)
    lr = fd.build_models()["logistic_regression"]
    lr.fit(X_train, y_train)
    scaler = lr.named_steps["standardscaler"]
    assert np.allclose(scaler.mean_, X_train.mean().to_numpy())


def test_run_end_to_end(data, tmp_path):
    results = fd.run(data, tmp_path)
    assert (tmp_path / "metrics.json").exists() and (tmp_path / "pr_curve.png").exists()
    assert json.loads((tmp_path / "metrics.json").read_text())["best_model"] in results["models"]
    assert results["baseline_always_legit"]["accuracy"] > 0.97  # accuracy alone looks great...
    for m in results["models"].values():
        assert m["recall"] > 0.5  # ...but the real models actually catch fraud
        assert 0 <= m["pr_auc"] <= 1


def test_load_data_requires_class_column(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        fd.load_data(p)
