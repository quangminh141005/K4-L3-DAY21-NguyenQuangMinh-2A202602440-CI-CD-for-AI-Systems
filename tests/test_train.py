import os
import json
import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score
from src.train import train


FEATURE_NAMES = [
    "age", "workclass", "education_num", "marital_status", "occupation",
    "relationship", "sex", "capital_gain", "capital_loss", "hours_per_week",
]


@pytest.fixture(autouse=True)
def isolated_training(tmp_path, monkeypatch):
    """Keep test artifacts and MLflow runs separate from real experiments."""
    monkeypatch.chdir(tmp_path)
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    try:
        yield
    finally:
        mlflow.set_tracking_uri(previous_uri)


def _make_temp_data(tmp_path):
    """
    Tao dataset nho voi cung schema Adult de su dung trong test.

    pytest cung cap `tmp_path` la mot thu muc tam thoi, tu dong xoa sau khi test ket thuc.
    Ham nay dung du lieu ngau nhien nen khong can ket noi cloud storage hay tai file CSV thuc.
    """
    rng = np.random.default_rng(0)
    n = 200

    X = rng.random((n, len(FEATURE_NAMES)))
    y = rng.integers(0, 2, size=n)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y

    train_path = str(tmp_path / "train.csv")
    eval_path = str(tmp_path / "holdout.csv")
    df.iloc[:160].to_csv(train_path, index=False)
    df.iloc[160:].to_csv(eval_path, index=False)
    return train_path, eval_path


def test_train_returns_float(tmp_path):
    """Kiem tra ham train() tra ve mot so thuc nam trong [0.0, 1.0]."""
    train_path, eval_path = _make_temp_data(tmp_path)
    f1 = train(
        {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2},
        data_path=train_path,
        eval_path=eval_path,
    )
    assert isinstance(f1, float)
    assert 0.0 <= f1 <= 1.0


def test_report_file_created(tmp_path):
    """Kiem tra file outputs/report.json duoc tao sau khi huan luyen."""
    train_path, eval_path = _make_temp_data(tmp_path)
    f1 = train(
        {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert os.path.exists("outputs/report.json")
    with open("outputs/report.json") as f:
        report = json.load(f)
    assert "f1_score" in report
    assert "accuracy" in report
    assert report["f1_score"] == f1
    for metric in ("f1_score", "accuracy"):
        assert isinstance(report[metric], float)
        assert 0.0 <= report[metric] <= 1.0


def test_model_file_created(tmp_path):
    """Kiem tra model da luu du doan duoc va khop voi metrics holdout."""
    train_path, eval_path = _make_temp_data(tmp_path)
    f1 = train(
        {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert os.path.exists("models/model.joblib")
    model = joblib.load("models/model.joblib")
    df_eval = pd.read_csv(eval_path)
    preds = model.predict(df_eval.drop(columns=["target"]))
    assert len(preds) == 40
    assert set(preds).issubset({0, 1})
    assert f1_score(df_eval["target"], preds) == pytest.approx(f1)
    with open("outputs/report.json") as f:
        report = json.load(f)
    assert accuracy_score(df_eval["target"], preds) == pytest.approx(report["accuracy"])
