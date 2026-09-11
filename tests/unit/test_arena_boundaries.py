"""Структурные запреты движка: то, чего в нём не должно быть ни при каких условиях.

Эти проверки читают исходный код. Так и задумано: разбиение и параллелизм — те места,
где нарушение не видно по результату. Свой `KFold`, добавленный «для удобства», обесценит
работу групп, времени, скользящего окна и `assignment_hash`, а метрика при этом останется
правдоподобной, и ни один поведенческий тест не покраснеет.

Тест краснеет ровно тогда, когда запрещённая конструкция появляется в коде, — это его
единственная задача (D-20).
"""
from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

ARENA = Path(__file__).resolve().parents[2] / "backend" / "arena"
BACKEND = Path(__file__).resolve().parents[2] / "backend"

#конструкторы разбиения: любой из них внутри движка означает, что фолды перестали
#быть единственным источником границ
SPLIT_CONSTRUCTORS = (
    "KFold",
    "StratifiedKFold",
    "GroupKFold",
    "StratifiedGroupKFold",
    "ShuffleSplit",
    "StratifiedShuffleSplit",
    "GroupShuffleSplit",
    "TimeSeriesSplit",
    "train_test_split",
    "cross_val_score",
    "cross_validate",
    "cross_val_predict",
)


def _sources(root: Path) -> list[tuple[Path, str]]:
    return [(path, path.read_text(encoding="utf-8")) for path in sorted(root.rglob("*.py"))]


def _code_only(source: str) -> str:
    """Исходник без комментариев и строковых литералов.

    Проверять запреты по сырому тексту нельзя: комментарий, объясняющий, **почему**
    конструкция запрещена, сам содержит её имя. Такой тест краснел бы именно на
    документации правила и подталкивал бы удалить объяснение вместо нарушения.
    """
    kept: list[str] = []
    skipped = {tokenize.COMMENT, tokenize.STRING, tokenize.FSTRING_START}

    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type not in skipped and token.string.strip():
            kept.append(token.string)

    return " ".join(kept)


def test_arena_never_builds_its_own_splits() -> None:
    """`FoldPlan` — закон: движок получает индексы готовыми и своих не строит."""
    offenders: list[str] = []

    for path, source in _sources(ARENA):
        code = _code_only(source)
        offenders.extend(
            f"{path.name}: {name}"
            for name in SPLIT_CONSTRUCTORS
            if re.search(rf"\b{name}\b", code)
        )

    assert not offenders, (
        "В Arena engine появилось построение разбиения. Индексы принадлежат FoldPlan: "
        f"{offenders}"
    )


def test_arena_does_not_import_model_selection() -> None:
    offenders = [
        path.name for path, source in _sources(ARENA) if "model_selection" in _code_only(source)
    ]

    assert not offenders, f"Arena engine импортирует модуль разбиений sklearn: {offenders}"


def test_no_adapter_or_engine_asks_for_every_core() -> None:
    """`n_jobs=-1` при параллельном запуске даёт «модели × все ядра» (D-12)."""
    offenders = [
        path.name
        for path, source in _sources(BACKEND)
        if re.search(r"(?:n_jobs|thread_count|nthread)\s*=\s*-\s*1\b", _code_only(source))
    ]

    assert not offenders, f"Запрошены все ядра: {offenders}"


def test_preprocessing_is_never_fitted_before_the_split() -> None:
    """`fit` препроцессора получает только обучающую часть фолда.

    Вызов `fit_transform` по всей таблице до разбиения — это утечка статистик
    в проверочную часть. Внутри движка такому вызову взяться неоткуда.
    """
    offenders = [
        path.name for path, source in _sources(ARENA) if "fit_transform" in _code_only(source)
    ]

    assert not offenders, f"Обучение трансформера вне границ фолда: {offenders}"


def test_arena_does_not_swallow_errors_silently() -> None:
    """`except: pass` прячет отказ и превращает его в пустое место в leaderboard."""
    offenders: list[str] = []

    for path, source in _sources(ARENA):
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.ExceptHandler):
                continue

            #докстрока-объяснение внутри обработчика допустима, а голый pass — нет
            body = [item for item in node.body if not isinstance(item, ast.Expr)]

            if len(body) == 1 and isinstance(body[0], ast.Pass):
                offenders.append(f"{path.name}:{node.lineno}")

    assert not offenders, f"Ошибка проглатывается без следа: {offenders}"


def test_worker_module_imports_only_the_standard_library() -> None:
    """Тяжёлый импорт на уровне модуля загрузил бы BLAS раньше ограничения потоков.

    Дочерний процесс импортирует этот модуль, чтобы найти точку входа. Если numpy или
    sklearn попадут туда, число потоков будет прочитано библиотекой до того, как мы его
    установим, и «2 контендера × 7 потоков» превратятся в два захвата всех ядер.
    """
    heavy = {"numpy", "polars", "pandas", "sklearn", "joblib", "scipy"}
    tree = ast.parse((ARENA / "worker.py").read_text(encoding="utf-8"))
    offenders: list[str] = []

    for node in tree.body:
        if isinstance(node, ast.Import):
            offenders.extend(
                alias.name for alias in node.names if alias.name.split(".")[0] in heavy
            )
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".")[0] in heavy
        ):
            offenders.append(node.module)

    assert not offenders, f"worker.py импортирует тяжёлые библиотеки на уровне модуля: {offenders}"
