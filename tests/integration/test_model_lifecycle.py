"""Жизненный цикл модели через HTTP: прогон → диагностика → артефакт → применение.

Проверяется именно та вертикаль, ради которой продукт существует: пользователь обучил
модели, разобрался в ошибках, сохранил модель и применил её к новым данным. Каждый шаг
здесь идёт через настоящее приложение, потому что коды ответов и форма данных — часть
публичного поведения.
"""
from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient

from backend.core.config import get_settings

WAIT_SECONDS = 300
ROWS = 400


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("MARENA_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    get_settings.cache_clear()

    from backend.api.app import create_app
    from backend.arena.runner import get_runner

    get_runner.cache_clear()

    with TestClient(create_app()) as test_client:
        yield test_client

    get_runner.cache_clear()
    get_settings.cache_clear()


def _dataset_bytes() -> bytes:
    generator = np.random.default_rng(23)
    tenure = generator.integers(0, 60, size=ROWS).astype(float)
    spend = generator.normal(100, 30, size=ROWS)
    #сигнал есть, но не детерминированный: иначе ошибок не останется и разбирать нечего
    score = 0.03 * spend - 0.04 * tenure + generator.normal(0, 1.0, size=ROWS)
    frame = pl.DataFrame(
        {
            "tenure": tenure,
            "spend": spend,
            "plan": generator.choice(["basic", "pro"], size=ROWS),
            "churned": score > float(np.mean(score)),
        }
    )
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    return buffer.getvalue()


@pytest.fixture
def finished_run(client: TestClient) -> tuple[str, str]:
    """Датасет и завершённый прогон с двумя участниками."""
    from backend.arena.runner import get_runner

    response = client.post(
        "/api/datasets",
        files={"file": ("customers.parquet", _dataset_bytes(), "application/octet-stream")},
    )
    assert response.status_code == 201, response.text
    dataset_id = response.json()["dataset"]["dataset_id"]

    started = client.post(
        "/api/arena/runs",
        json={
            "dataset_id": dataset_id,
            "target_column": "churned",
            "n_splits": 3,
            "seed": 17,
            "selected_contenders": ["logistic_regression"],
        },
    )
    assert started.status_code == 202, started.text
    run_id = started.json()["run"]["run_id"]
    get_runner().wait(run_id, timeout=WAIT_SECONDS)

    card = client.get(f"/api/arena/runs/{run_id}").json()
    assert card["run"]["state"] == "SUCCEEDED", card["run"]["error_message"]
    return dataset_id, run_id


# ---------------------------------------------------------------- диагностика


def test_diagnostics_measure_the_gap_and_expose_thresholds(
    client: TestClient, finished_run: tuple[str, str]
) -> None:
    _, run_id = finished_run
    response = client.get(
        f"/api/arena/runs/{run_id}/contenders/logistic_regression/diagnostics"
    )

    assert response.status_code == 200, response.text
    diagnostics = response.json()["diagnostics"]

    #разрыв измерен, а не предположен: замер на обучающей части сохраняется прогоном
    assert diagnostics["folds"]
    assert all(row["probe_rows"] > 0 for row in diagnostics["folds"])
    assert diagnostics["mean_gap"] is not None
    #пороги отдаются наружу, а не живут внутри интерфейса
    assert diagnostics["thresholds"]["relative_gap"]["caution"]
    assert diagnostics["classification"]["available"] is True
    assert diagnostics["regression"] is None


def test_error_rows_lead_to_real_observations(
    client: TestClient, finished_run: tuple[str, str]
) -> None:
    _, run_id = finished_run
    response = client.get(
        f"/api/arena/runs/{run_id}/contenders/logistic_regression/errors",
        params={"kind": "false_negative", "limit": 5},
    )

    assert response.status_code == 200, response.text
    errors = response.json()["errors"]
    assert errors["total_matching"] > 0

    row = errors["rows"][0]
    assert set(row["features"]) == {"tenure", "spend", "plan"}
    #цель не подписана словом «признак», а идентификатор строки не стал признаком
    assert "churned" in row["context"]
    assert "__row_id__" not in row["features"]


def test_diagnostics_for_a_contender_that_never_ran_is_a_clear_refusal(
    client: TestClient, finished_run: tuple[str, str]
) -> None:
    _, run_id = finished_run
    response = client.get(f"/api/arena/runs/{run_id}/contenders/catboost/diagnostics")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


# ---------------------------------------------------------------- артефакт


def test_artifact_is_saved_and_describes_itself(
    client: TestClient, finished_run: tuple[str, str]
) -> None:
    dataset_id, run_id = finished_run
    response = client.get(f"/api/models/{run_id}/logistic_regression")

    assert response.status_code == 200, response.text
    body = response.json()
    manifest = body["manifest"]

    assert manifest["model_sha256"]
    assert manifest["feature_schema"]["columns"] == ["tenure", "spend", "plan"]
    assert manifest["task"]["target_column"] == "churned"
    assert manifest["dataset"]["dataset_id"] == dataset_id
    assert manifest["protocol"]["fold_assignment_hash"]
    assert manifest["environment"]["sklearn"]
    #граница доверия названа прямо в ответе
    assert "joblib" in body["trust_boundary"]


def test_corrupted_artifact_is_refused_over_http(
    client: TestClient, finished_run: tuple[str, str]
) -> None:
    from backend.arena.runner import get_runner

    _, run_id = finished_run
    path = get_runner().store.contender_path(run_id, "logistic_regression") / "model.joblib"
    path.write_bytes(b"broken")

    response = client.get(f"/api/models/{run_id}/logistic_regression")

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "artifact_corrupted"


# ---------------------------------------------------------------- применение


def test_single_prediction_uses_the_saved_preprocessing(
    client: TestClient, finished_run: tuple[str, str]
) -> None:
    _, run_id = finished_run
    response = client.post(
        f"/api/models/{run_id}/logistic_regression/predict",
        json={"values": {"tenure": 12.0, "spend": 140.0, "plan": "pro"}},
    )

    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["n_rows"] == 1
    assert result["has_probabilities"] is True
    #вероятности подписаны метками задачи и суммируются к единице
    probabilities = result["predictions"][0]["probabilities"]
    assert set(probabilities) == {"False", "True"}
    assert abs(sum(probabilities.values()) - 1.0) < 1e-6


def test_single_prediction_without_a_required_column_is_refused(
    client: TestClient, finished_run: tuple[str, str]
) -> None:
    _, run_id = finished_run
    response = client.post(
        f"/api/models/{run_id}/logistic_regression/predict",
        json={"values": {"tenure": 12.0, "spend": 140.0}},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "schema_mismatch"
    assert "plan" in body["error"]["message"]


def test_batch_prediction_returns_rows_and_exportable_csv(
    client: TestClient, finished_run: tuple[str, str]
) -> None:
    _, run_id = finished_run
    fresh = pl.DataFrame(
        {
            "tenure": [5.0, 40.0, 20.0],
            "spend": [150.0, 60.0, 100.0],
            "plan": ["pro", "basic", "pro"],
        }
    )
    buffer = io.BytesIO()
    fresh.write_csv(buffer)

    response = client.post(
        f"/api/models/{run_id}/logistic_regression/predict-batch",
        files={"file": ("fresh.csv", buffer.getvalue(), "text/csv")},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["result"]["n_rows"] == 3
    #выгрузка — часть продукта, а не дополнение к нему
    assert body["csv"].splitlines()[0].startswith("index,prediction")
    assert len(body["csv"].splitlines()) == 4


def test_importance_is_measured_on_holdout_without_retraining(
    client: TestClient, finished_run: tuple[str, str]
) -> None:
    _, run_id = finished_run
    response = client.get(
        f"/api/models/{run_id}/logistic_regression/importance", params={"repeats": 3}
    )

    assert response.status_code == 200, response.text
    importance = response.json()["importance"]

    assert importance["available"] is True
    assert importance["measured_on"] == "holdout"
    assert {row["feature"] for row in importance["rows"]} == {"tenure", "spend", "plan"}
    #никаких причинных обещаний
    assert "не утверждение" in importance["interpretation"]
    assert importance["caveats"]


# ---------------------------------------------------------------- эксперименты


def test_experiment_history_carries_conditions_and_champion(
    client: TestClient, finished_run: tuple[str, str]
) -> None:
    dataset_id, run_id = finished_run
    response = client.get("/api/experiments")

    assert response.status_code == 200, response.text
    experiments = response.json()["experiments"]
    entry = next(item for item in experiments if item["run_id"] == run_id)

    assert entry["dataset"]["dataset_id"] == dataset_id
    assert entry["dataset"]["fingerprint"]
    assert entry["protocol"]["fold_assignment_hash"]
    assert entry["champion"]["contender_key"] in {"logistic_regression", "baseline_majority", None}
    assert entry["artifacts"]
    assert entry["runtime_seconds"] is not None


def test_comparing_runs_names_what_differs(client: TestClient, finished_run: tuple[str, str]) -> None:
    """Сравнение начинается с условий, а не с метрик."""
    from backend.arena.runner import get_runner

    dataset_id, first = finished_run
    second = client.post(
        "/api/arena/runs",
        json={
            "dataset_id": dataset_id,
            "target_column": "churned",
            #другой seed — другое фактическое разбиение
            "n_splits": 3,
            "seed": 99,
            "selected_contenders": ["logistic_regression"],
        },
    )
    assert second.status_code == 202, second.text
    second_id = second.json()["run"]["run_id"]
    get_runner().wait(second_id, timeout=WAIT_SECONDS)

    response = client.post("/api/experiments/compare", json={"run_ids": [first, second_id]})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["comparable"] is False
    keys = {item["key"] for item in body["differences"]}
    assert "seed" in keys
    assert "fold_assignment_hash" in keys
    assert "разбиение" in body["verdict"] or "отличаются" in body["verdict"]
