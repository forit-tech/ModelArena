"""HTTP-слой: импорт, карточка, предпросмотр, постановка задачи.

Тесты идут через реальное приложение целиком: контракт ошибок, коды статусов и форма
ответа — часть публичного поведения, и проверять их в обход HTTP бессмысленно.
"""
from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

from backend.core.config import get_settings

CONTRACT_ROOT = Path(__file__).resolve().parents[2] / "contracts" / "dataset-package" / "v1"
GOLDEN_ZIP = CONTRACT_ROOT / "golden" / "customers_golden.dapkg.zip"


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    #каждому тесту свой каталог артефактов: иначе тесты начинают видеть данные друг друга
    monkeypatch.setenv("MARENA_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    get_settings.cache_clear()

    from backend.api.app import create_app

    with TestClient(create_app()) as test_client:
        yield test_client

    get_settings.cache_clear()


@pytest.fixture
def table_bytes() -> bytes:
    frame = pl.DataFrame(
        {
            "spend": [float(value) for value in range(60)],
            "plan": ["basic", "pro"] * 30,
            "churned": [True, False] * 30,
        }
    )
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    return buffer.getvalue()


def _upload(client: TestClient, name: str, payload: bytes) -> dict:
    response = client.post("/api/datasets", files={"file": (name, payload)})
    assert response.status_code == 201, response.text
    return response.json()["dataset"]


# ---------------------------------------------------------------- health


def test_health_exposes_contract_version(client: TestClient) -> None:
    #при расхождении с DataArena причина должна быть видна сразу, а не выясняться
    #по коду отказа при первом импорте
    payload = client.get("/api/health").json()

    assert payload["status"] == "ok"
    assert payload["dataset_package"]["format"] == "dataarena.package"
    assert payload["dataset_package"]["fingerprint_algorithm"] == "dataarena-logical-sha256-v1"


# ---------------------------------------------------------------- импорт


def test_package_import_returns_snapshot(client: TestClient) -> None:
    dataset = _upload(client, "customers_golden.dapkg.zip", GOLDEN_ZIP.read_bytes())

    assert dataset["source"] == "package"
    assert dataset["row_count"] == 12
    assert dataset["row_key"] == ["customer_id"]


def test_broken_package_is_refused_with_contract_code(client: TestClient) -> None:
    #код отказа приходит из матрицы контракта как есть: обе стороны называют
    #одну причину одинаково
    broken = (CONTRACT_ROOT / "golden" / "negative" / "traversal.dapkg.zip").read_bytes()
    response = client.post("/api/datasets", files={"file": ("traversal.dapkg.zip", broken)})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "package_unsafe_path"


def test_unsupported_format_is_named(client: TestClient) -> None:
    response = client.post("/api/datasets", files={"file": ("notes.txt", b"hello")})

    assert response.status_code == 400
    assert "Неподдерживаемый формат" in response.json()["detail"]


def test_same_data_imported_twice_is_one_dataset(client: TestClient, table_bytes: bytes) -> None:
    first = _upload(client, "sales.parquet", table_bytes)
    second = _upload(client, "sales_copy.parquet", table_bytes)

    assert first["dataset_id"] == second["dataset_id"]
    assert client.get("/api/datasets").json()["total"] == 1


# ---------------------------------------------------------------- карточка и страница


def test_card_contains_profile_computed_by_us(client: TestClient) -> None:
    dataset = _upload(client, "customers_golden.dapkg.zip", GOLDEN_ZIP.read_bytes())
    payload = client.get(f"/api/datasets/{dataset['dataset_id']}").json()

    profile = payload["profile"]
    assert profile["row_count"] == 12
    identifier = next(column for column in profile["columns"] if column["name"] == "customer_id")
    assert identifier["is_probable_id"]
    #никакого сводного балла: он скрывает, какой именно сигнал его испортил (D-10)
    assert "quality_score" not in profile


def test_preview_returns_only_the_visible_page(client: TestClient, table_bytes: bytes) -> None:
    dataset = _upload(client, "sales.parquet", table_bytes)
    payload = client.get(
        f"/api/datasets/{dataset['dataset_id']}/preview", params={"offset": 10, "limit": 5}
    ).json()

    assert len(payload["rows"]) == 5
    assert payload["total_rows"] == 60
    assert payload["rows"][0]["spend"] == 10.0


def test_oversized_page_is_refused(client: TestClient, table_bytes: bytes) -> None:
    dataset = _upload(client, "sales.parquet", table_bytes)
    response = client.get(
        f"/api/datasets/{dataset['dataset_id']}/preview", params={"limit": 10_000}
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "request_validation_error"


def test_unknown_dataset_gives_structured_error(client: TestClient) -> None:
    response = client.get("/api/datasets/ds_0000000000000000000000")

    assert response.status_code == 404
    body = response.json()
    #единый формат: и машинный код, и текст для человека
    assert body["error"]["code"] == "not_found"
    assert body["detail"]


def test_dataset_can_be_deleted(client: TestClient, table_bytes: bytes) -> None:
    dataset = _upload(client, "sales.parquet", table_bytes)
    assert client.delete(f"/api/datasets/{dataset['dataset_id']}").status_code == 200
    assert client.get("/api/datasets").json()["total"] == 0


# ---------------------------------------------------------------- постановка задачи


def test_analyze_explains_task_and_protocol(client: TestClient, table_bytes: bytes) -> None:
    dataset = _upload(client, "sales.parquet", table_bytes)
    response = client.post(
        f"/api/datasets/{dataset['dataset_id']}/task/analyze", json={"target_column": "churned"}
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["task"]["task_type"] == "binary"
    assert payload["protocol"]["cv_splitter"] == "stratified_k_fold"
    #объяснение — обязательная часть ответа, а не украшение
    assert payload["task"]["explanation"]
    assert payload["protocol"]["explanation"]


def test_analyze_reports_unselected_structure(client: TestClient) -> None:
    frame = pl.DataFrame(
        {
            "customer_id": [value // 4 for value in range(80)],
            "spend": [float(value) for value in range(80)],
            "churned": [True, False] * 40,
        }
    )
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    dataset = _upload(client, "repeat.parquet", buffer.getvalue())

    payload = client.post(
        f"/api/datasets/{dataset['dataset_id']}/task/analyze", json={"target_column": "churned"}
    ).json()

    assert any("идентификатор сущности" in warning for warning in payload["protocol"]["warnings"])


def test_analyze_refuses_unknown_target(client: TestClient, table_bytes: bytes) -> None:
    dataset = _upload(client, "sales.parquet", table_bytes)
    response = client.post(
        f"/api/datasets/{dataset['dataset_id']}/task/analyze", json={"target_column": "missing"}
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "validation_error"


# ---------------------------------------------------------------- готовность


def test_analyze_returns_readiness_without_a_score(client: TestClient, table_bytes: bytes) -> None:
    dataset = _upload(client, "sales.parquet", table_bytes)
    payload = client.post(
        f"/api/datasets/{dataset['dataset_id']}/task/analyze", json={"target_column": "churned"}
    ).json()

    readiness = payload["readiness"]
    assert readiness["status"] in {"READY", "CAUTION", "HIGH_RISK", "BLOCKED"}
    #балл запрещён: он прячет причину за одним числом (D-22)
    assert "score" not in readiness
    assert readiness["summary"]
    #контекст обязан быть виден, иначе пороги нечем проверить
    assert readiness["context"]["effective_dimension"] >= 1


def test_readiness_reacts_to_structure_left_unhandled(client: TestClient) -> None:
    frame = pl.DataFrame(
        {
            "customer_id": [value // 4 for value in range(200)],
            "spend": [round(value * 1.7, 2) for value in range(200)],
            "churned": [value % 3 == 0 for value in range(200)],
        }
    )
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    dataset = _upload(client, "repeat.parquet", buffer.getvalue())

    payload = client.post(
        f"/api/datasets/{dataset['dataset_id']}/task/analyze", json={"target_column": "churned"}
    ).json()

    codes = {finding["code"] for finding in payload["readiness"]["findings"]}
    assert "group_structure_ignored" in codes
    finding = next(
        item for item in payload["readiness"]["findings"] if item["code"] == "group_structure_ignored"
    )
    #находка обязана объяснить, что именно исказится в сравнении
    assert "узнавать сущность" in finding["consequence"]


# ---------------------------------------------------------------- контендеры


def test_analyze_lists_contenders_with_reasons(client: TestClient, table_bytes: bytes) -> None:
    dataset = _upload(client, "sales.parquet", table_bytes)
    payload = client.post(
        f"/api/datasets/{dataset['dataset_id']}/task/analyze", json={"target_column": "churned"}
    ).json()

    contenders = payload["contenders"]
    assert contenders

    baselines = [item for item in contenders if item["is_baseline"]]
    assert len(baselines) == 1, "baseline обязан быть ровно один"

    #ни один участник не исчезает молча: у всего, кроме ready, есть причина
    for item in contenders:
        if item["status"] != "ready":
            assert item["reason"], f"{item['adapter_key']}: статус {item['status']} без причины"

    assert payload["resources"]["threads_per_fit"] >= 1
