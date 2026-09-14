"""HTTP-слой прогонов: коды статусов, форма ответа, защита от двойного запуска.

Коды и структура ответа — часть публичного поведения, и проверять их в обход HTTP
бессмысленно: ровно на этой границе прошлый раз обнаружилось, что клиент не отправлял
заголовок типа содержимого, а модульные тесты этого не видели.
"""
from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

from backend.core.config import get_settings

WAIT_SECONDS = 180


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("MARENA_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    get_settings.cache_clear()

    from backend.api.app import create_app
    from backend.arena.runner import get_runner

    #реестр прогонов живёт синглтоном на процесс: без сброса тест увидел бы каталог
    #предыдущего и историю чужих прогонов
    get_runner.cache_clear()

    with TestClient(create_app()) as test_client:
        yield test_client

    get_runner.cache_clear()
    get_settings.cache_clear()


@pytest.fixture
def dataset_id(client: TestClient) -> str:
    rows = 240
    frame = pl.DataFrame(
        {
            "amount": [float((value * 41) % 233) for value in range(rows)],
            "plan": [["basic", "pro"][value % 2] for value in range(rows)],
            "churned": [(value * 41) % 233 > 120 for value in range(rows)],
        }
    )
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    response = client.post(
        "/api/datasets",
        files={"file": ("customers.parquet", buffer.getvalue(), "application/octet-stream")},
    )
    assert response.status_code == 201, response.text
    return response.json()["dataset"]["dataset_id"]


def _start(client: TestClient, dataset_id: str, **extra: object) -> dict:
    payload = {
        "dataset_id": dataset_id,
        "target_column": "churned",
        "n_splits": 2,
        "holdout_size": 0.2,
        "seed": 7,
        "selected_contenders": ["logistic_regression"],
    }
    payload.update(extra)
    return client.post("/api/arena/runs", json=payload)


def _finish(client: TestClient, run_id: str) -> dict:
    from backend.arena.runner import get_runner

    get_runner().wait(run_id, timeout=WAIT_SECONDS)
    response = client.get(f"/api/arena/runs/{run_id}")
    assert response.status_code == 200, response.text
    return response.json()


def test_start_returns_accepted_and_the_run_card(client: TestClient, dataset_id: str) -> None:
    response = _start(client, dataset_id)

    assert response.status_code == 202, response.text
    body = response.json()
    assert body["duplicate_of_active_run"] is False
    run = body["run"]
    assert run["state"] in {"PENDING", "RUNNING"}
    #отпечаток разбиения переносится в карточку: честность фолдов проверяема
    assert run["spec"]["fold_assignment_hash"]
    #метка процесса-владельца — внутреннее дело backend
    assert "owner_token" not in run

    _finish(client, run["run_id"])


def test_run_reaches_success_with_metrics_and_progress(
    client: TestClient, dataset_id: str
) -> None:
    run_id = _start(client, dataset_id).json()["run"]["run_id"]
    body = _finish(client, run_id)

    assert body["run"]["state"] == "SUCCEEDED"
    assert body["active"] is False
    assert body["progress"]["completed_fraction"] == 1.0

    contenders = client.get(f"/api/arena/runs/{run_id}/contenders").json()["contenders"]
    assert {item["contender_key"] for item in contenders} == {
        "baseline_majority",
        "logistic_regression",
    }

    for item in contenders:
        assert item["state"] == "SUCCEEDED"
        cross_validated = next(part for part in item["metrics"] if part["split"] == "cv")
        assert any(metric["value"] is not None for metric in cross_validated["metrics"])


def test_baseline_is_present_even_when_not_selected(client: TestClient, dataset_id: str) -> None:
    """Выбор пользователя сужает состав, но точку отсчёта не убирает."""
    run_id = _start(client, dataset_id).json()["run"]["run_id"]
    body = _finish(client, run_id)
    keys = {item["contender_key"] for item in body["run"]["contenders"]}

    assert "baseline_majority" in keys


def test_repeated_submission_does_not_create_a_second_run(
    client: TestClient, dataset_id: str
) -> None:
    first = _start(client, dataset_id)
    second = _start(client, dataset_id)

    assert first.status_code == 202
    #200 вместо 202: приняли к сведению, но нового прогона не создали
    assert second.status_code == 200
    assert second.json()["duplicate_of_active_run"] is True
    assert second.json()["run"]["run_id"] == first.json()["run"]["run_id"]

    _finish(client, first.json()["run"]["run_id"])
    assert len(client.get("/api/arena/runs").json()["runs"]) == 1


def test_events_are_readable_and_pageable(client: TestClient, dataset_id: str) -> None:
    run_id = _start(client, dataset_id).json()["run"]["run_id"]
    _finish(client, run_id)

    everything = client.get(f"/api/arena/runs/{run_id}/events").json()["events"]
    tail = client.get(f"/api/arena/runs/{run_id}/events", params={"since": 2}).json()

    assert everything[0]["kind"] == "run_queued"
    assert everything[-1]["kind"] == "run_finished"
    #прогресс событийный: каждый посчитанный фолд — отдельная запись
    assert any(item["kind"] == "fold_completed" for item in everything)
    assert tail["events"] == everything[2:]


def test_unknown_run_answers_in_the_common_error_shape(client: TestClient) -> None:
    response = client.get("/api/arena/runs/run_0123456789abcdef0123")

    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "not_found"
    assert body["detail"] == body["error"]["message"]


def test_malformed_run_id_is_refused_before_touching_the_filesystem(
    client: TestClient,
) -> None:
    response = client.get("/api/arena/runs/..%2F..%2Fsecrets")

    assert response.status_code in {400, 404}
    assert response.json()["error"]["code"] in {"validation_error", "not_found", "http_error"}


def test_cancelling_a_finished_run_is_refused(client: TestClient, dataset_id: str) -> None:
    run_id = _start(client, dataset_id).json()["run"]["run_id"]
    assert _finish(client, run_id)["run"]["state"] == "SUCCEEDED"

    response = client.post(f"/api/arena/runs/{run_id}/cancel")

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "validation_error"


def test_leaderboard_recomputes_under_a_new_objective_without_retraining(
    client: TestClient, dataset_id: str
) -> None:
    """Смена цели даёт другую таблицу и не запускает ни одного обучения (D-6)."""
    run_id = _start(client, dataset_id).json()["run"]["run_id"]
    assert _finish(client, run_id)["run"]["state"] == "SUCCEEDED"

    default = client.get(f"/api/arena/runs/{run_id}/leaderboard")
    assert default.status_code == 200, default.text
    board = default.json()["leaderboard"]
    assert board["objective"]["metric"] == "roc_auc"
    assert board["rows"]
    #каждая строка объясняет своё место, а не просто несёт число
    assert any(row["versus_baseline"] for row in board["rows"] if not row["is_baseline"])

    custom = client.post(
        f"/api/arena/runs/{run_id}/leaderboard",
        json={
            "metric": "recall",
            "constraints": [
                {
                    "metric": "precision",
                    "operator": "gte",
                    "value": 0.99,
                    "reason": "ложная тревога дорого стоит",
                }
            ],
        },
    )
    assert custom.status_code == 200, custom.text
    changed = custom.json()["leaderboard"]
    assert changed["objective"]["metric"] == "recall"
    assert changed["objective"]["constraints"][0]["metric"] == "precision"
    #история прогонов не пополнилась: обучение не повторялось
    assert len(client.get("/api/arena/runs").json()["runs"]) == 1


def test_leaderboard_refuses_a_metric_the_task_does_not_have(
    client: TestClient, dataset_id: str
) -> None:
    run_id = _start(client, dataset_id).json()["run"]["run_id"]
    _finish(client, run_id)

    response = client.get(f"/api/arena/runs/{run_id}/leaderboard", params={"metric": "rmse"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "validation_error"
