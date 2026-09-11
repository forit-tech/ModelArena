# ModelArena — Design

Технический дизайн до реализации. Опирается на зафиксированные решения [DECISIONS.md](DECISIONS.md)
(`D-1`…`D-13`), границы [SCOPE.md](SCOPE.md) и обоснования [ARCHITECTURE_REVIEW.md](ARCHITECTURE_REVIEW.md).

Здесь — **что именно строим**: структура репозитория, доменная модель, интерфейсы, схемы,
раскладка артефактов, границы API, план миграции, объём MVP и Stage 1.

Дата: 2026-09-06. Кода нет.

---

## 0. Единая нумерация этапов

До сих пор существовали два плана: roadmap 0–14 в `ARCHITECTURE_REVIEW.md` §25 и план 0–8
в `SCOPE.md` §6. Они описывали одно и то же разными словами. **Этот документ их заменяет.**

| Этап | Содержание | Тип |
|---|---|---|
| 0 | Границы, решения, контракт, дизайн | документы (закрыт этим документом) |
| **1** | **Скелет, core, dataset layer, чтение package, Task Setup** | **новое + перенос** |
| 2 | Dataset Readiness | новое |
| 3 | Protocol + preprocessing profiles + гарантии честности | перенос + новое |
| 4 | Adapter registry + baseline + ядро моделей | новое (замена `catalog.py`) — **закрыт** |
| 5 | Arena engine + jobs + resource budget | новое (замена `trainer.py`) — **закрыт** |
| 6 | Metrics + CV aggregation + Leaderboard + Champion | перенос + новое |
| 7 | Head-to-Head | новое |
| 8 | Diagnostics: overfitting, learning curves, calibration, error analysis | перенос + новое |
| 9 | Experiment tracking + `experiment.yaml` | перенос + новое |
| 10 | Optuna tuning финалистов | новое |
| 11 | Model export bundle | перенос + новое |
| 12 | Интеграция с DataArena (импорт по URL, deep links) | новое |
| 13 | Drift / сравнение с будущим датасетом | перенос + новое |
| 14 | Failure slices, SHAP (опц.), производительность, README | новое |

Граница MVP проходит **после этапа 9** (см. §9).

---

## 1. Структура репозитория

```
ModelArena/
├── README.md
├── pyproject.toml
├── docs/
│   ├── DECISIONS.md              ADR log
│   ├── SCOPE.md                  границы сервиса
│   ├── ARCHITECTURE_REVIEW.md    обоснования решений
│   ├── DESIGN.md                 этот файл
│   ├── MIGRATION.md              карта переноса из AutoDataAnalysis
│   └── contracts/
│       └── dataset-package-consumer.md
├── contracts/
│   └── dataset-package/v1/       копия JSON Schema + golden package (владелец — DataArena)
├── backend/
│   ├── core/          config.py errors.py logging.py ids.py hashing.py
│   ├── datasets/      io.py registry.py snapshot.py package_reader.py profile.py cache.py
│   ├── tasks/         spec.py inference.py features.py
│   ├── readiness/     report.py checks/{size,target,missing,cardinality,rare,duplicates,
│   │                                    identifiers,signal,feasibility}.py
│   ├── leakage/       report.py checks/{target_copy,identifiers,post_event,near_copy,
│   │                                    datetime,duplicates_across_split,single_feature,
│   │                                    group_crossing,missingness}.py
│   ├── protocol/      plan.py splitters.py feasibility.py recommend.py
│   ├── preprocessing/ profiles.py factory.py transformers.py
│   ├── models/        registry.py context.py applicability.py
│   │                  adapters/{dummy,linear,forest,extra_trees,hist_gbm,
│   │                            svm,knn,xgboost,lightgbm,catboost}.py
│   ├── arena/         runner.py orchestration.py resources.py
│   ├── evaluation/    metrics.py aggregation.py predictions.py bootstrap.py
│   ├── scoring/       objectives.py normalization.py composite.py champion.py
│   ├── diagnostics/   overfitting.py learning_curves.py calibration.py
│   │                  error_analysis.py slices.py head_to_head.py
│   ├── explain/       importance.py permutation.py shap_module.py
│   ├── tuning/        stages.py spaces.py study.py
│   ├── experiments/   records.py store.py reproducibility.py
│   ├── artifacts/     bundle.py export.py loading.py
│   ├── monitoring/    profiles.py drift.py comparison.py novelty.py
│   ├── jobs/          queue.py worker.py events.py state.py
│   └── api/           app.py routes/ schemas/
├── frontend/          React 19 + TS + Vite + Recharts (D-3)
├── tests/             unit/ integration/ fairness/ contracts/
├── examples/          benchmark-датасеты и скрипты
└── .github/workflows/ ci.yml
```

**Правила зависимостей** (проверяются линтером импортов в CI):

1. ни один модуль домена не импортирует `api` и `jobs`;
2. `arena/` вызывает `protocol`, `preprocessing`, `models`, `evaluation` — но сам не считает
   ни метрики, ни split;
3. `scoring/` и `diagnostics/` читают только сохранённые результаты и предсказания,
   к моделям и данным для обучения не обращаются (следствие D-6);
4. `models/adapters/*` не импортируют друг друга;
5. никакого импорта кода DataArena — нигде и никогда.

---

## 2. Experiment domain model

### 2.1 Сущности

```python
# datasets/
@dataclass(frozen=True)
class DatasetSnapshot:
    dataset_id: str                 # ds_<ulid>
    name: str
    source: Literal["package", "file", "url"]
    package_ref: PackageRef | None  # format, format_version, package_id, links
    fingerprint: str                # sha256(schema || content) — идентичность данных
    row_count: int
    column_count: int
    schema: list[ColumnSpec]        # имя, physical/logical/semantic type, nullable
    created_at: str
    parent_dataset_id: str | None   # для снимков, полученных из другого снимка

# tasks/
@dataclass(frozen=True)
class TaskSpec:
    target_column: str
    task_type: Literal["binary", "multiclass", "regression"]
    positive_label: str | None      # обязателен для binary
    class_labels: list[str] | None  # ФИКСИРОВАННЫЙ порядок на весь run
    feature_columns: list[str]
    excluded_columns: list[ExclusionRecord]   # (column, reason, source)
    group_column: str | None
    time_column: str | None

@dataclass(frozen=True)
class ExclusionRecord:
    column: str
    reason: str
    source: Literal["user", "leakage_guard", "readiness", "package_hint"]

# protocol/
@dataclass(frozen=True)
class EvaluationProtocol:
    holdout: HoldoutSpec | None
    cv: CrossValidationSpec
    seed: int
    derived_seeds: dict[str, int]   # splitter, bootstrap, per-adapter
    fold_assignment_hash: str       # доказательство одинаковости split (D-7)
    strategy_explanation: str
    warnings: list[str]
    feasibility_notes: list[str]

# preprocessing/
@dataclass(frozen=True)
class PreprocessingSpec:
    profile_key: str                # scaled_onehot | tree_ordinal | native_missing | ...
    numeric_columns: list[str]
    datetime_columns: list[str]
    categorical_columns: list[str]
    params: dict[str, Any]          # min_frequency, max_categories, impute strategies
    effective_dim_estimate: int     # размерность ПОСЛЕ кодирования

# models/
@dataclass(frozen=True)
class Contender:
    contender_key: str              # уникален в пределах run: "hist_gbm", "hist_gbm__tuned"
    adapter_key: str
    label: str
    family: str
    params: dict[str, Any]
    origin: Literal["default", "suggested", "tuned"]
    tuning_trials: int | None
    preprocessing_profile: str
    is_baseline: bool

# experiments/
@dataclass
class ArenaRun:
    run_id: str                     # run_<ulid>
    created_at: str
    name: str
    dataset_id: str
    dataset_fingerprint: str
    task: TaskSpec
    protocol: EvaluationProtocol
    preprocessing: dict[str, PreprocessingSpec]   # contender_key → spec
    contenders: list[Contender]
    resource_budget: ResourceBudget
    status: RunStatus
    progress: RunProgress
    environment: Environment
    readiness_ref: str              # путь к сохранённому отчёту
    leakage_ref: str
    timings: dict[str, float]

@dataclass(frozen=True)
class ContenderResult:
    contender_key: str
    status: Literal["ok", "failed", "skipped", "unavailable", "cancelled"]
    reason: str | None              # обязателен для skipped / unavailable / failed
    fold_metrics: list[dict[str, float]]
    cv_mean: dict[str, float]
    cv_std: dict[str, float]
    holdout_metrics: dict[str, float]
    train_metrics: dict[str, float]           # только для диагностики overfitting
    holdout_intervals: dict[str, Interval]    # bootstrap
    timings: Timings                          # fit_seconds, predict_seconds, per_1k_rows
    artifact_size_bytes: int | None
```

### 2.2 Вычисляемые проекции (не хранятся как истина, D-6)

```python
Objective(preset, weights: dict[str, float], primary_metric: str)
LeaderboardRow(contender_key, cv_mean, cv_std, holdout, gain_over_baseline,
               normalized: dict[str, float], composite_score, rank)
Champion(contender_key, objective, composite_score, breakdown, justification,
         caveats: list[str])
```

`Leaderboard` и `Champion` — **функция** от `list[ContenderResult] × Objective`.
Они пересчитываются при каждом запросе и никогда не пишутся в `run.json` как факт.
Если бы писались — смена весов задним числом рассинхронизировала бы карточку с реальностью.

### 2.3 Инварианты

| # | Инвариант | Где проверяется |
|---|---|---|
| I-1 | `dataset_fingerprint` в run равен фактическому fingerprint снимка | старт run |
| I-2 | `target_column ∉ feature_columns` | сборка `TaskSpec` |
| I-3 | `class_labels` фиксированы на весь run; вероятности всех моделей выровнены по этому порядку | `evaluation/predictions.py` |
| I-4 | `fold_assignment_hash` одинаков у всех контендеров | тест fairness |
| I-5 | Трансформеры fit только на train-части фолда | тест leakage |
| I-6 | Holdout не читается во время tuning | счётчик доступа + тест |
| I-7 | Невычислимая метрика отсутствует в словаре, а не равна 0 | `evaluation/metrics.py` + тест |
| I-8 | `__row_id__` никогда не входит в матрицу признаков | сборка признаков + тест |
| I-9 | Снимок датасета неизменяем; правка = новый `dataset_id` | `datasets/registry.py` |

### 2.4 Жизненный цикл

```
DRAFT ──── выбран target, task type ────► CONFIGURED
                                              │  readiness + leakage посчитаны,
                                              │  протокол построен, контендеры выбраны
                                              ▼
                                          QUEUED ──► RUNNING ──┬──► COMPLETED
                                                       │        ├──► PARTIAL   (часть контендеров упала)
                                                       │        └──► FAILED
                                                       └── cancel ──► CANCELLED
```

`PARTIAL` — отдельное состояние сознательно: если 5 из 7 контендеров обучились, run полезен,
и прятать его за `FAILED` нечестно. В карточке видно, кто и почему не дошёл.

---

## 3. ModelAdapter protocol

### 3.1 Контекст и бюджет

```python
@dataclass(frozen=True)
class DatasetContext:
    n_rows: int
    n_train_rows: int              # строк в одном train-фолде — на этом реально учат
    n_features: int
    effective_dim: int             # размерность ПОСЛЕ кодирования выбранным профилем
    n_numeric: int
    n_categorical: int
    n_datetime: int
    max_cardinality: int
    missing_ratio: float
    task_type: str
    n_classes: int | None
    min_class_count: int | None
    has_native_missing_need: bool  # есть ли пропуски вообще

@dataclass(frozen=True)
class ResourceBudget:
    max_parallel_fits: int
    threads_per_fit: int           # передаётся в n_jobs / nthread / thread_count
    memory_budget_mb: int
    time_budget_seconds: int | None
```

### 3.2 Интерфейс

```python
class Availability(NamedTuple):
    available: bool
    reason: str          # "lightgbm не установлен: pip install modelarena[boosters]"

class Applicability(NamedTuple):
    verdict: Literal["ok", "caution", "skip"]
    reason: str          # обязателен, если verdict != "ok"

class ModelAdapter(Protocol):
    key: str                                  # "hist_gbm"
    label: str                                # "HistGradientBoosting"
    family: str                               # dummy|linear|tree_ensemble|gbdt|svm|neighbors
    supported_tasks: frozenset[str]
    preprocessing_profile: str
    supports_proba: bool
    is_baseline: bool

    def availability(self) -> Availability: ...
    def applicability(self, ctx: DatasetContext) -> Applicability: ...
    def default_params(self, ctx: DatasetContext) -> dict[str, Any]: ...
    def build(self, params: dict[str, Any], budget: ResourceBudget, seed: int) -> BaseEstimator: ...
    def search_space(self, ctx: DatasetContext) -> dict[str, ParamSpec]: ...
    def native_importance(self, fitted: BaseEstimator) -> dict[str, float] | None: ...
    def estimate_fit_cost(self, ctx: DatasetContext) -> CostEstimate: ...
```

Правила реализации:

- `build` **обязан** взять число потоков из `budget`. `n_jobs=-1` запрещён — линтер-правило в CI;
- `build` **обязан** принять `seed` и прокинуть его во все стохастические параметры;
- адаптер не знает про фолды, метрики, leaderboard — только про то, как собрать estimator;
- `estimate_fit_cost` — грубая оценка (порядок величины) для планирования очереди и для
  предупреждения «этот контендер, скорее всего, займёт десятки минут».

### 3.3 Applicability: конкретные правила MVP

| Адаптер | Условие | Вердикт и причина |
|---|---|---|
| `knn` | `n_rows > 200_000` | skip — «расстояния до 200k+ объектов на каждое предсказание; inference будет непригоден» |
| `knn` | `effective_dim > 50` | caution — «в высокой размерности расстояния теряют различимость» |
| `svm_rbf` | `n_train_rows > 20_000` | skip — «время обучения растёт как O(n²)…O(n³)» |
| `logistic` | `effective_dim > 5 · n_train_rows` | caution — «признаков кратно больше наблюдений» |
| любой one-hot | `effective_dim > 10_000` | caution — «взрыв размерности после кодирования» |
| любой one-hot | `effective_dim · n_rows · 8 > memory_budget` | skip — «оценка матрицы превышает бюджет памяти» |
| `random_forest` | `n_rows > 1_000_000` | caution — «ожидаемое время обучения — десятки минут» |
| любой | `availability().available is False` | статус `unavailable`, contender остаётся в списке |

`skip` не убирает контендера из leaderboard: строка остаётся со статусом и причиной.
Пользователь может форсировать запуск вручную — тогда причина сохраняется в run как «forced by user».

### 3.4 Состав MVP (D-2)

| Ядро (sklearn) | classification | regression |
|---|---|---|
| baseline | `DummyClassifier(most_frequent)` + `DummyClassifier(stratified)` | `DummyRegressor(median)` |
| линейная | LogisticRegression | Ridge |
| деревья | RandomForest, ExtraTrees | RandomForest, ExtraTrees |
| бустинг | HistGradientBoosting | HistGradientBoosting |

Extras `[boosters]`: XGBoost, LightGBM, CatBoost.
Guarded (только по applicability): SVM (RBF), KNN.

Baseline **не отключается**. В leaderboard всегда есть колонка «прирост над baseline».

---

## 4. EvaluationProtocol

### 4.1 Схема

```jsonc
{
  "seed": 42,
  "holdout": {
    "enabled": true,
    "strategy": "stratified",          // random | stratified | temporal | group
    "test_size": 0.2,
    "row_count": 2568
  },
  "cv": {
    "splitter": "stratified_k_fold",   // k_fold | stratified_k_fold | group_k_fold
                                       // | stratified_group_k_fold | forward_chaining
    "n_splits": 5,
    "shuffle": true,
    "n_repeats": null,
    "group_column": null,
    "time_column": null
  },
  "derived_seeds": { "splitter": 42, "bootstrap": 1042, "adapters": 2042 },
  "fold_assignment_hash": "sha256:7c1e...",
  "strategy_explanation": "Задача классификации, минорный класс — 214 объектов, что не меньше числа фолдов. Стратификация сохраняет доли классов в каждом фолде.",
  "warnings": [],
  "feasibility_notes": ["Запрошено 10 фолдов, применено 5: минорный класс содержит 214 объектов."]
}
```

### 4.2 Правило выбора (auto, пользователь может переопределить)

```
если задан group_column:
    classification → stratified_group_k_fold (при выполнимости) иначе group_k_fold
    regression     → group_k_fold
    holdout        → group split
иначе если задан time_column:
    cv      → forward_chaining (растущее окно)
    holdout → хвост по времени
иначе если classification и min_class_count >= n_splits:
    cv → stratified_k_fold ; holdout → stratified
иначе если classification:
    cv → k_fold + warning «стратификация невозможна»
иначе:
    cv → k_fold ; holdout → random
```

Кандидаты в `time_column` и `group_column` детектируются переносимым кодом
(`splits.find_time_columns`, `find_group_columns`) и **предлагаются**, а не выбираются молча.
Каждое авторешение сопровождается `strategy_explanation`.

### 4.3 Лестница выполнимости

Проверяется **до** постановки job — иначе sklearn падает внутри воркера, и пользователь
видит `FAILED` без причины.

| Проверка | Действие |
|---|---|
| `min_class_count < n_splits` | понизить `n_splits` до `min_class_count`, записать в `feasibility_notes` |
| `min_class_count < 2` | отказ: стратификация и CV невозможны; предложить убрать класс или сменить target |
| `n_groups < n_splits` | понизить `n_splits`; при `n_groups < 2` — отказ |
| `n_rows < 50` | предупреждение «на валидацию приходится ~10 строк, различия будут в пределах шума» |
| holdout оставляет `< 30` строк | bootstrap-интервалы отключаются с явной причиной |
| forward chaining: `< 3` различимых периодов | откат на `k_fold` + warning |

### 4.4 Гарантии честности (D-7)

- `fold_assignment_hash = sha256(конкатенация отсортированных test-индексов по фолдам)`;
- вычисляется **один раз** при построении протокола, копируется в каждый `ContenderResult`;
- тест `test_same_split_for_all_contenders` сравнивает хеши;
- holdout-индексы лежат в отдельном объекте с обёрткой доступа, считающей обращения;
  во время Stage 3 tuning счётчик обязан остаться нулевым (D-1, тест I-6).

---

## 5. Раскладка артефактов

```
artifacts/
├── datasets/
│   └── ds_01J8.../
│       ├── data.parquet            неизменяемый снимок (D-8)
│       ├── meta.json               DatasetSnapshot
│       ├── profile.json            профиль, посчитанный НАМИ (D-9)
│       ├── row_index.parquet       __row_id__ ↔ исходный ключ строки
│       └── package/                если source == "package": manifest, schema, lineage — как есть
└── runs/
    └── run_01J8.../
        ├── run.json                RunSpec + состояние + list[ContenderRecord]
        ├── analysis.json           readiness и leakage на момент запуска
        ├── folds.npz               утверждённое разбиение + row_ids снимка
        ├── events.jsonl            журнал наблюдённых фактов
        ├── contenders/             ПУБЛИКУЕТСЯ ЦЕЛИКОМ, по каталогу на участника
        │   ├── logistic_regression/
        │   │   ├── predictions.parquet
        │   │   └── result.json
        │   └── hist_gradient_boosting/
        │       ├── predictions.parquet
        │       └── result.json
        ├── .staging/               сборка результата до публикации
        ├── models/                 (этап 11)
        └── diagnostics/            (этап 8)
```

**Почему каталог на участника, а не `predictions/<contender>.parquet` рядом с общим
`run.json`.** Обучение, запись предсказаний и запись метаданных — три операции. Падение
между ними не имеет права оставить «успешный» полуартефакт: карточка с метриками, но без
предсказаний, выглядит достоверно и не проверяется ничем. Результат собирается в `.staging`
и переносится на место **одним** переименованием каталога, поэтому предсказания и метаданные
становятся видимы одновременно.

`run.json` тоже пишется атомарно: временный файл рядом и `os.replace`. Дописывание по месту
при обрыве оставило бы нечитаемый JSON вместе со ссылками на уже посчитанные предсказания.

### 5.1 Схема `predictions/<contender>.parquet`

| Колонка | Тип | Смысл |
|---|---|---|
| `__row_id__` | `int64` | позиция строки в снимке датасета |
| `split` | `str` | `cv` \| `holdout` \| `train_probe` |
| `fold` | `int8` | номер фолда; `-1` для holdout |
| `y_true` | по задаче | факт |
| `y_pred` | по задаче | предсказание |
| `proba__<label>` | `float32` | по колонке на каждый класс, порядок = `TaskSpec.class_labels` |

Одна строка датасета встречается в `cv` **ровно один раз** (out-of-fold) и в `holdout`
ровно один раз, если попала в holdout. Это делает join двух контендеров по `__row_id__`
однозначным — на нём стоит весь Head-to-Head (D-6).

`train_probe` пишется только для диагностики overfitting и по подвыборке train, чтобы файл
не рос вдвое.

### 5.2 Правила хранения

- `models/*.joblib` сопровождаются `sha256` в `run.json`; загрузка только после сверки
  (joblib исполняет код при десериализации);
- пользовательские `.joblib` не загружаются никогда;
- retention: снимок датасета не удаляется, пока на него ссылается хотя бы один run;
- `predictions/` — самое ценное после `run.json`: при нехватке места первыми чистятся
  `models/` проигравших контендеров, а не предсказания.

---

## 6. Dataset Package: сторона потребителя

Контракт `dataarena.package/1` принадлежит DataArena (D-11). Полная спецификация потребителя —
[docs/contracts/dataset-package-consumer.md](contracts/dataset-package-consumer.md).

Кратко, что делает ModelArena при импорте:

1. читает `manifest.json`, проверяет `format == "dataarena.package"` и **мажорную** часть
   `format_version`; неизвестная мажорная версия → отказ с внятным текстом, без попыток угадать;
2. сверяет `sha256` каждой используемой части до чтения;
3. читает `data/*.parquet` → собирает снимок → считает **свой** fingerprint;
4. читает `schema.json` — `semantic_type` используется как стартовая гипотеза о типе колонки,
   но подтверждается фактическим dtype в parquet;
5. `profile.json` и `hints` показывает с пометкой источника; **на решения не влияют** (D-9);
   при `computed_on.sampled: true` в UI стоит явная плашка;
6. `lineage.json` / `recipe.json` сохраняет как есть для карточки датасета;
7. незнакомые поля игнорирует.

Запреты: никаких pickle/joblib внутри пакета, никакого исполнения выражений из пакета,
лимит на размер и на число частей.

---

## 7. Граница API

### 7.1 DataArena → ModelArena

Единственная точка — импорт пакета. **Pull-модель**: ModelArena сама забирает данные.

```
POST /api/datasets/import
  multipart:  file=<name>.dapkg.zip
  или JSON:   { "package_url": "http://dataarena.local/api/workspaces/ws1/packages/pkg_.../download" }
  или JSON:   { "package_path": "D:/exports/customers.dapkg" }
→ 201 { "dataset_id": "ds_...", "warnings": [...] }
```

Deep link из DataArena («Open in ModelArena»):

```
http://modelarena.local/#/import?package_url=<urlencoded>
```

Фронт ModelArena дёргает тот же endpoint и ведёт пользователя сразу в Task Setup.

### 7.2 ModelArena → DataArena

Только **навигационная**, без обмена данными: если `manifest.links.inspect_url_template`
присутствует, в интерфейсе появляется `Inspect in DataArena`, подставляющая имя колонки.
Никаких вызовов API DataArena из backend ModelArena нет.

### 7.3 Поведение при недоступности

| Ситуация | Поведение |
|---|---|
| DataArena не запущена | ModelArena работает; импорт по URL падает с понятной ошибкой; импорт файлом работает |
| ModelArena не запущена | забота DataArena; её кнопка сообщает о недоступности |
| Пакет неизвестной мажорной версии | отказ с текстом «ModelArena понимает dataarena.package/1, получено 2.0» |
| `sha256` не совпал | отказ; данные не читаются |

### 7.4 Внутренний API ModelArena (набросок для Stage 1+)

```
GET    /api/health
POST   /api/datasets/import                     пакет | файл | url            (Stage 1)
GET    /api/datasets                            список снимков                (Stage 1)
GET    /api/datasets/{id}                       карточка + схема + профиль    (Stage 1)
GET    /api/datasets/{id}/preview               страница строк                (Stage 1)
POST   /api/datasets/{id}/task/analyze          тип задачи + рекомендация протокола (Stage 1)
POST   /api/datasets/{id}/readiness             отчёт готовности              (Stage 2)
POST   /api/datasets/{id}/leakage               кандидаты в утечку            (Stage 3)
POST   /api/arena/runs                          создать run → 202 + run_id    (Stage 5, готово)
                                                200, если такой эксперимент уже идёт (D-27)
GET    /api/arena/runs                          история                       (Stage 5, готово)
GET    /api/arena/runs/{id}                     карточка + прогресс           (Stage 5, готово)
GET    /api/arena/runs/{id}/contenders          участники со статусами        (Stage 5, готово)
GET    /api/arena/runs/{id}/events?since=N      журнал событий                (Stage 5, готово)
POST   /api/arena/runs/{id}/cancel              остановка                     (Stage 5, готово)
GET    /api/runs/{id}/leaderboard?objective=…   пересчёт без обучения (D-6)   (Stage 6)
GET    /api/runs/{id}/head-to-head?a=&b=                                      (Stage 7)
GET    /api/runs/{id}/rows?bucket=…             строки под сегментом          (Stage 7)
GET    /api/runs/{id}/contenders/{key}/diagnostics                            (Stage 8)
POST   /api/runs/{id}/champion/export                                         (Stage 11)
POST   /api/predictions                         inference новым датасетом     (после MVP)
POST   /api/monitoring/compare                  drift                         (Stage 13)
```

Формат ошибок переносится из AutoDataAnalysis без изменений:
`{"error": {"code", "message", "details"}, "detail": "..."}`.

---

## 8. Карта миграции

Полная потабличная карта с LOC и вердиктами — [MIGRATION.md](MIGRATION.md).

Сводка: `backend/ml/` — **3 959 строк**, из них **3 212 переносятся, 747 переписываются** (D-12).

| Вердикт | LOC | Что |
|---|---|---|
| перенос как есть | 2 155 | leakage (487), splits (460), error_analysis (366), importance (318), experiments/store (229), metrics (196), bootstrap (99) |
| перенос с адаптацией | 1 057 | predictor (438), verdict (218), preprocessing (166), config (115), experiments/models (113) |
| переписывается | 747 | `trainer.py` (604) → `arena/` + `protocol/` + `evaluation/`; `catalog.py` (143) → `models/adapters/` |
| не переносится | — | `etl/`, `dataview/`, `profiling/quality.py` (Quality Score остаётся в DataArena, D-10) |

Плюс к этому: 447 строк HTTP-слоя (`routes/{training,experiments,inference}.py` + `schemas/ml.py`),
3 064 строки frontend-разделов и 1 065 строк ML-тестов.

---

## 9. Объём MVP

**MVP = этапы 1–9.** Формулировка того, что MVP умеет:

> Пользователь берёт подготовленный датасет, выбирает target, видит честную оценку
> готовности данных и кандидатов в утечку, запускает турнир моделей на одном протоколе,
> получает leaderboard с приростом над baseline и CV-дисперсией, задаёт objective,
> получает Champion с объяснением выбора, сравнивает две модели построчно,
> разбирает ошибки и видит, переобучилась ли модель. Прогон сохраняется и воспроизводим.

### Входит в MVP

| Область | Объём |
|---|---|
| Импорт | Dataset Package + прямой файл (D-13); снимок в Parquet; fingerprint |
| Task Setup | автоопределение типа задачи, выбор target, исключение признаков с причиной |
| Readiness | все проверки §11 ARCHITECTURE_REVIEW, статусы READY/CAUTION/HIGH RISK |
| Leakage | перенесённые 8 проверок + group crossing + missingness |
| Protocol | holdout + CV, 5 сплиттеров, лестница выполнимости, `fold_assignment_hash` |
| Preprocessing | 4 профиля под семейства моделей |
| Contenders | baseline + 4 модели ядра на задачу; extras-адаптеры при наличии зависимостей |
| Arena | job-система, прогресс, cancel, resource budget |
| Metrics | полный набор classification/regression, CV mean±std, bootstrap на holdout |
| Leaderboard | пересчёт по objective без обучения; 5 пресетов + произвольные веса |
| Champion | composite score с разложением + оговорка «разница в пределах шума» |
| Head-to-Head | квадранты, клик до строк |
| Diagnostics | overfitting, learning curves, calibration, error analysis, importance |
| Experiments | история, карточка, `experiment.yaml`, environment |
| Frontend | Datasets, Task Setup, Arena, Leaderboard, Head-to-Head, Diagnostics, Experiments |

### Сознательно НЕ входит в MVP

| Не входит | Почему | Когда |
|---|---|---|
| **Optuna tuning** | Сначала должен быть честный турнир на дефолтах. Tuning без защищённого протокола — это переобучение с красивым интерфейсом | Этап 10 |
| **Export bundle модели** | Требует стабильной схемы признаков и решения по `refit_scope`; до этого экспортировать нечего | Этап 11 |
| **Импорт по URL из DataArena** | DataArena ещё не производит пакеты; в MVP импорт файлом и golden package | Этап 12 |
| **Drift / сравнение с будущим датасетом** | Нужен стабильный baseline-профиль модели, который появляется только после этапа 11 | Этап 13 |
| **Прогноз деградации** | Требует истории снимков, которой в MVP физически нет. Делать «прогноз» без истории — это то, что ТЗ §23 прямо запрещает | после накопления истории |
| **Failure slices** | Требует статистики множественных сравнений; сырая версия даёт мусорные выводы по трём объектам | Этап 14 |
| **SHAP** | Permutation importance закрывает потребность в MVP; SHAP — тяжёлая зависимость ради второго графика | Этап 14, optional extra |
| **Inference / Test Drive** | Полезно, но не отвечает на главный вопрос ModelArena («какая модель лучше и почему»). Код перенесён, включается после экспорта | после этапа 11 |
| **Multilabel, ranking, time series forecasting, anomaly detection** | ТЗ §2 прямо говорит не пытаться сделать всё в v1 | не раньше v2 |
| **Distributed workers (Celery/Redis)** | Локальной job-системы достаточно; абстракция очереди оставляет путь наверх | по потребности |
| **AI-слой пересказа** | Остаётся в архиве AutoDataAnalysis; к ML-решениям отношения не имеет | не планируется |
| **Редактирование данных, SQL, join** | Это DataArena (SCOPE §1, D-13) | никогда |

---

## 10. Stage 1 — что делается

**Предусловие:** Dataset Package v1 заморожен — producer- и consumer-тесты прошли на одном
golden package ([D-15](DECISIONS.md#d-15-критерий-freeze-контракта-и-разблокировки-stage-1)).
До этого этап не начинается.

**Цель этапа:** довести до рабочего состояния путь «датасет попал внутрь → пользователь выбрал
target → система сказала, какая это задача и каким протоколом её проверять». Без обучения.

### Состав

| # | Работа | Источник |
|---|---|---|
| 1.1 | Скелет репозитория, `pyproject.toml` с extras, CI (ruff + mypy + pytest + frontend build) | новое |
| 1.2 | `core/`: config, errors, единый формат ошибок, фабрика FastAPI-приложения | перенос как есть |
| 1.3 | `datasets/io.py`: чтение csv/parquet/json/jsonl/xlsx, детект кодировки и разделителя, `to_pandas`, fingerprint | перенос как есть |
| 1.4 | `datasets/registry.py` + `snapshot.py`: неизменяемый снимок в Parquet, `meta.json`, `__row_id__`, LRU-кеш | новое (кеш — перенос) |
| 1.5 | `datasets/package_reader.py`: чтение `dataarena.package/1`, проверка версии и `sha256`, отказ на major-mismatch | новое |
| 1.6 | `datasets/profile.py`: профиль колонок для карточки датасета | перенос с адаптацией (без Quality Score) |
| 1.7 | `tasks/`: `TaskSpec`, автоопределение типа задачи, разделение binary/multiclass, отбор признаков | перенос с адаптацией |
| 1.8 | `protocol/recommend.py`: детекторы time/group колонок + рекомендация стратегии с объяснением | перенос как есть |
| 1.9 | API: `health`, `datasets/import`, `datasets`, `datasets/{id}`, `datasets/{id}/preview`, `task/analyze` | новое + перенос контракта ошибок |
| 1.10 | Frontend: каркас, тёмная тема, разделы Datasets и Task Setup, транспортный слой, `ErrorBoundary` | перенос токенов + новое |
| 1.11 | Golden package в `contracts/` + тест чтения | новое |

### Definition of Done этапа 1

- `pytest` зелёный, `ruff` и `mypy` чистые, `npm run build` проходит;
- тест: golden package читается, схема и профиль доступны, **ни один модуль DataArena не импортирован**
  (проверяется анализом импортов, а не на глаз);
- тест: пакет с `format_version: "2.0"` отвергается с внятным сообщением;
- тест: `.csv` и `.parquet` с одинаковым содержимым дают снимки с одинаковым fingerprint;
- тест: снимок неизменяем — повторный импорт того же файла не создаёт второй снимок;
- ручной smoke: импорт файла → карточка датасета → выбор target → «binary classification,
  рекомендован stratified k-fold, потому что …»;
- коммит + чекпоинт.

### Чего в Stage 1 намеренно нет

Обучения, метрик, моделей, readiness-отчёта, job-системы. Этап проверяет только вход
и постановку задачи. Если вход спроектирован неверно, всё остальное придётся переделывать —
поэтому он идёт первым и отдельно.

---

## 11. Технические риски

ML-риски (leakage, unfair comparison, winner's curse и т.д.) перечислены в
[ARCHITECTURE_REVIEW.md](ARCHITECTURE_REVIEW.md) §26 — R1…R17. Здесь — риски реализации
и переноса, которых там нет.

| # | Риск | Вероятность | Митигация |
|---|---|---|---|
| T-1 | **Контракт пакета не утверждён**, обе стороны реализуют импорт/экспорт и расходятся | снят процедурой | Stage 1 **не начинается ни одной стороной**, пока не выполнен критерий freeze: producer- и consumer-тесты проходят на одном golden package (D-15). Дальше импорт работает против него же, лежащего в обоих репозиториях |
| T-2 | **Перенос ML-кода тянет за собой связи**: `trainer.py` импортирует 12 модулей, включая `profiling.safe_ratio` из DataArena-домена | высокая | Порядок переноса — снизу вверх: сначала листовые модули без зависимостей, затем те, кто их использует. `safe_ratio` дублируется в `core/` (4 строки, дублирование дешевле связи) |
| T-3 | **Тесты переносятся вместе с кодом, но зависят от `conftest.py` и фикстур** старого проекта | высокая | Фикстуры переносятся первыми, отдельным коммитом; тест считается перенесённым, только когда зелёный в новом репозитории |
| T-4 | **Windows-специфика**: `ProcessPoolExecutor` использует spawn, воркер переимпортирует модуль; тяжёлые импорты sklearn в каждом процессе | средняя | Воркер импортирует адаптер лениво, по `adapter_key`; замер накладных расходов на старте этапа 5, до массового распараллеливания |
| T-5 | **Измерение памяти ненадёжно** | средняя | Метрика памяти помечена опциональной с самого начала; если не удаётся мерить стабильно — поле не показывается вообще, а не показывается неверным |
| T-6 | **Parquet-round-trip меняет dtype** (категории, nullable-целые, timezone) | средняя | Тест round-trip на benchmark-датасетах в Stage 1; расхождение типов — ошибка импорта, а не «мелочь» |
| T-7 | **`predictions/*.parquet` растут** на больших датасетах с многоклассовой задачей (`proba__*` на класс) | средняя | `float32` вместо `float64`; при `n_classes > 30` вероятности пишутся только для top-k классов + сумма остальных, и это отражено в схеме |
| T-8 | **Пересчёт leaderboard на каждый запрос** становится узким местом при 20 контендерах × 10 фолдов | низкая | Считается по агрегатам из `run.json`, а не по `predictions/`; предсказания читаются только для Head-to-Head и диагностики конкретной пары |
| T-9 | **SSE-прогресс не переживает reverse proxy** | низкая | Fallback на polling `GET /runs/{id}` заложен с самого начала; SSE — оптимизация, а не обязательный транспорт |
| T-10 | **Расхождение frontend-типов с backend-схемами** | средняя | Типы генерируются из OpenAPI, не пишутся руками; проверка в CI |
| T-11 | **Два roadmap в двух документах** (уже случилось: `ARCHITECTURE_REVIEW` §25 и `SCOPE` §6) | реализовался | §0 этого документа — единственная нумерация; при расхождении прав он |
| T-12 | **Дублирование работы с параллельной сессией DataArena** (уже случилось: контракт спроектирован дважды) | реализовался | Владение по артефактам: контракт и golden package — у DataArena, ML-домен и артефакты экспериментов — у ModelArena |

---

## 12. Что нужно от DataArena

Замечания к контракту, которые ModelArena просит внести владельцу (D-11). Все они уже
приняты в [RECONCILIATION.md](../../DataArena/docs/contracts/RECONCILIATION.md) §2 —
здесь они собраны как чеклист:

1. блок `version: { dataset_id, version, parent_version }` в манифесте — нужен, чтобы
   отличать версии одного датасета и связывать эксперименты между собой;
2. `links.inspect_url_template` — для кнопки `Inspect in DataArena`;
3. pull-транспорт: URL пакета, а не push данных;
4. явный запрет исполняемой сериализации внутри пакета;
5. формулировка «`hints` и `profile` — advisory» в тексте контракта, а не только в сверке;
6. golden package в обоих репозиториях как источник истины.
