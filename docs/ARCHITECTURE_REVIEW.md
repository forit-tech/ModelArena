# ModelArena — Architecture & Design Review

Статус: **дизайн до реализации**. Код не написан. Документ нужен для ревью и правок до старта.

Дата: 2026-09-06

---

## 0. Что было изучено

| Что | Результат |
|---|---|
| `C:\...\MyDevs\DataArena` | Пустая папка. Кода нет. |
| `C:\...\MyDevs\ModelArena` | Пустая папка. Кода нет. |
| Параллельная сессия DataArena | Идёт, находится в той же фазе (исследование/дизайн). Формат Dataset Package там ещё не зафиксирован. |
| Старый проект | `C:\...\MyDevs\data_analysis_automatisation` = **AutomatedDataAnalysis**, git remote `forit-tech/AutoDataAnalysis`. 61 Python-файл, ~10 400 строк вместе с тестами, FastAPI + Polars + sklearn, React 19 + TS + Vite. |

**Главный вывод по контракту:** DataArena сейчас нечего импортировать. Значит контракт Dataset Package проектируется здесь как **предложение**, фиксируется в `contracts/` отдельным версионированным JSON Schema и должен быть согласован с DataArena до реализации импорта. До согласования ModelArena работает standalone — это и так обязательное требование (§1 ТЗ).

---

## 1. Старый проект: что это на самом деле

AutoDataAnalysis — это уже **объединение DataArena и ModelArena в одном приложении**:

```
Overview   → паспорт датасета, Quality Score           → DataArena
Explore    → просмотр, поиск, SQL                      → DataArena
Prepare    → merge, поколоночные правила, ETL-рецепт   → DataArena
Model      → split + baseline + сравнение кандидатов   → ModelArena
Evaluate   → error analysis, importance, эксперименты  → ModelArena
Test Drive → inference на новых данных                 → ModelArena
```

Разделение на два сервиса — не переписывание с нуля, а **разрез по уже существующему шву**. Это сильно снижает риск: ML-слой уже написан, оттестирован и написан в честной парадигме (baseline обязателен, split объясняет себя, leakage — «сигналы», а не «доказано», метрики с bootstrap-интервалами). Это ровно принципы §31 ТЗ.

При этом старый ML-слой — **это MVP-уровень**, а не ModelArena:

- 4 кандидата вместо адаптерной системы;
- один holdout, никакого CV, никакой CV-дисперсии;
- нет learning curves, calibration, head-to-head, tuning, drift;
- нет job-системы: обучение висит в HTTP-запросе;
- нет objective/composite score — «лучшая модель» выбирается по одной метрике (`balanced_accuracy` / `rmse`), что прямо запрещено §10 ТЗ.

---

## 2. Reusable code: конкретный аудит

Переиспользуем **логику и формулировки**, а не импортируем пакет (сервисы не должны зависеть друг от друга и от старого репозитория). Код копируется в ModelArena и дорабатывается.

| Модуль старого проекта | Строк | Вердикт |
|---|---|---|
| `ml/validation/splits.py` | 460 | **Взять почти целиком.** `recommend_split` + объяснения + детекторы time/group колонок — готовый фундамент §5. Расширить: KFold/StratifiedKFold/GroupKFold/forward-chaining, feasibility-проверки. |
| `ml/validation/leakage.py` | 487 | **Взять целиком.** 8 проверок + осторожные формулировки + DISCLAIMER — это готовый Leakage Guard §4. Добавить: missingness-паттерн, group-утечку относительно фактического протокола. |
| `ml/evaluation/metrics.py` | 196 | **Взять как основу.** Правильные вещи: `HIGHER_IS_BETTER`, `describe_metric`, отсутствующая метрика не пишется как 0. Дополнить: log loss, Brier, MCC, MAPE с проверкой применимости, per-class. |
| `ml/evaluation/bootstrap.py` | 99 | **Взять целиком.** Перцентильный bootstrap + честный отказ на малой выборке. |
| `ml/evaluation/error_analysis.py` | 366 | **Взять как основу §12–13.** Уже есть FP/FN, худшие срезы с минимальным размером среза, остатки, гистограммы. Усилить статистикой (см. риск R5). |
| `ml/explainability/importance.py` | 318 | **Взять целиком.** Permutation importance + коэффициенты со знаком + честно названная окклюзия вместо фейкового «SHAP». |
| `ml/training/preprocessing.py` | 166 | **Взять как основу.** `DateTimeFeatures`, `split_feature_types` — рабочие. Переделать в **несколько профилей препроцессинга** под семейства моделей (раздел 10). |
| `ml/experiments/store.py` | 229 | **Взять целиком.** Protocol + FileSystem-реализация + SHA-256 проверка артефакта + защита пути. Готовая база §19. |
| `ml/experiments/models.py` | 113 | **Взять как основу**, но домен переработать (раздел 7 — нужно больше сущностей). |
| `datasets/io.py` | 254 | **Взять целиком.** csv/parquet/json/jsonl/xlsx, детект разделителя и кодировки, `to_pandas` без pyarrow, fingerprint. |
| `datasets/cache.py` | 76 | **Взять целиком.** LRU по fingerprint с лимитом байт. |
| `ml/inference/predictor.py` | 438 | **Взять частично.** `validate_inference_schema` и `assess_novelty` прямо нужны для §22 (сравнение с будущим датасетом). |
| `profiling/profiler.py` | 373 | **Взять частично.** Профиль колонок нужен как вход Readiness и как baseline-профиль для drift. Но Quality Score (`profiling/quality.py`) — **не брать**: это домен DataArena, и §3 запрещает сводить readiness к одному числу. |
| `profiling/comparison.py` | 206 | **Взять как основу §22.** Структура «метрика / before / after / delta / direction» — ровно то, что нужно для drift-отчёта. |
| `core/errors.py`, `core/config.py`, `api/app.py` | ~280 | **Взять целиком.** Единый формат ошибок, фабрика приложения, настройки из env с рабочими дефолтами. |
| `etl/*`, `dataview/*` | ~700 | **Не брать.** Это DataArena. |
| `ml/training/trainer.py` | 604 | **Не брать как есть.** Это ровно тот «один trainer.py», против которого §26. Функция `train_models` делает десять разных дел. Разбирается на `arena/runner`, `protocol`, `evaluation`, `diagnostics`, `experiments`. |
| `ml/training/catalog.py` | 143 | **Не брать.** Хардкод четырёх моделей заменяется адаптерной системой. |
| `tests/*` | ~1 600 | **Взять как образец.** Особенно `test_leakage.py` и `test_splits.py`. |

### Три конкретных дефекта старого кода, которые нельзя переносить

1. **`n_jobs=-1` в каталоге моделей.** `RandomForestClassifier(n_jobs=-1)` при последовательном обучении безобиден, но при параллельном запуске нескольких кандидатов даёт «N моделей × все ядра» — ровно сценарий «убить компьютер» из §28. В ModelArena число потоков модели задаёт оркестратор из бюджета ресурсов, `-1` запрещён.
2. **`class_weight="balanced"` по умолчанию** для LogisticRegression и RandomForest. Это тихо меняет калибровку вероятностей и делает несопоставимыми log loss, Brier и PR-AUC с моделями, у которых флага нет (GradientBoosting его не получает). При наличии раздела Calibration (§16) это прямая ложь. Решение: балансировка — **явное решение эксперимента**, применяется ко всем контендерам или ни к одному, и записывается в конфиг.
3. **Выбор лучшей модели по одной метрике** (`max(balanced_accuracy)`). Прямо запрещено §10.

---

## 3. Границы продукта: что ModelArena НЕ делает

ModelArena показывает данные только в объёме, нужном для ML-решения:

- превью N строк и схему — да;
- распределение target и отдельного признака — да;
- строки, на которых модель ошиблась — да (это Error Analysis, а не редактор);
- SQL IDE, merge, редактирование ячеек, экспорт в пять форматов — **нет**, это DataArena.

Из ModelArena можно уйти обратно в DataArena (`Inspect in DataArena`) только если датасет пришёл из неё и в манифесте есть `links.inspect_url_template`.

---

## 4. Архитектура: слои и поток

```
                    ┌──────────────────────────────────────────┐
  upload / package  │  datasets   snapshot + registry +         │
        ──────────► │             fingerprint (Parquet, immut.) │
                    └───────────────┬──────────────────────────┘
                                    ▼
                    ┌──────────────────────────────────────────┐
                    │  tasks      TaskSpec: target, task type,  │
                    │             features, excluded + reasons  │
                    └───────────────┬──────────────────────────┘
                                    ▼
             ┌──────────────────────┴──────────────────────┐
             ▼                                             ▼
    ┌─────────────────┐                          ┌──────────────────┐
    │  readiness      │                          │  leakage guard   │
    │  READY/CAUTION/ │                          │  candidates,     │
    │  HIGH RISK      │                          │  не «доказано»   │
    └────────┬────────┘                          └────────┬─────────┘
             └──────────────────┬─────────────────────────┘
                                ▼
                    ┌──────────────────────────────────────────┐
                    │  protocol   splitter + folds + seeds      │
                    │             ОДИН план на всех контендеров │
                    └───────────────┬──────────────────────────┘
                                    ▼
                    ┌──────────────────────────────────────────┐
                    │  arena/runner  ◄── jobs (QUEUED/RUNNING/  │
                    │                     COMPLETED/FAILED/     │
                    │                     CANCELLED) + budget   │
                    │   для каждого контендера:                 │
                    │     preprocessing spec → Pipeline         │
                    │     fit по фолдам (трансформеры fit       │
                    │     ТОЛЬКО на train-фолде)                │
                    │     → predictions.parquet + timings       │
                    └───────────────┬──────────────────────────┘
                                    ▼
        ┌───────────────────────────┴──────────────────────────┐
        ▼                    ▼                  ▼              ▼
  ┌───────────┐      ┌─────────────┐    ┌────────────┐  ┌───────────┐
  │evaluation │      │ diagnostics │    │  scoring   │  │  explain  │
  │ metrics + │      │ overfit,    │    │ objective  │  │ importance│
  │ CV agg +  │      │ curves,     │    │ → composite│  │ permutat. │
  │ bootstrap │      │ calibration,│    │ → Champion │  │ (SHAP opt)│
  └───────────┘      │ H2H, slices │    └─────┬──────┘  └───────────┘
                     └─────────────┘          ▼
                                       ┌────────────┐
                                       │ artifacts  │──► experiment.yaml
                                       │ + monitor  │──► drift vs new data
                                       └────────────┘
```

**Ключевое архитектурное решение:** оркестратор сохраняет **предсказания всех контендеров** (out-of-fold + holdout) в `predictions/<contender>.parquet`. Всё, что идёт после — метрики, leaderboard, composite score, head-to-head, error analysis, calibration, confusion matrix, failure slices, смена objective и весов — **считается post-hoc без переобучения**. Это делает §10 (смена objective) и §11 (head-to-head) мгновенными, а не отдельными прогонами.

Второе решение: **никакого кода, общего с DataArena**. Общий только версионированный контракт-файл, лежащий копией в обоих репозиториях и проверяемый golden-фикстурами с обеих сторон.

---

## 5. Canonical format и Dataset Registry

**Внутренний канонический формат — Parquet на диске, Polars DataFrame в памяти, pandas только на границе sklearn.**

Почему Parquet, а не CSV:

- сохраняет dtypes. Через CSV дата становится строкой, и `split_feature_types` начинает считать её категорией — это молча ломает и препроцессинг, и выбор temporal split;
- сохраняет NULL отдельно от пустой строки;
- быстрее и компактнее при повторных чтениях (кеш эксперимента, learning curves и drift-сравнения читают датасет многократно).

```
artifacts/
  datasets/<dataset_id>/
    data.parquet          # неизменяемый снимок
    meta.json             # источник, fingerprint, схема, время, ссылка на DataArena
    profile.json          # профиль колонок, пересчитанный НАМИ
    row_index.parquet     # __row_id__ ↔ исходный ключ строки (если был)
```

- `dataset_id` — случайный, `fingerprint` — SHA-256 (схема + содержимое). Два одинаковых файла с разными именами дают один fingerprint; переименование колонки меняет его.
- Снимок **неизменяем**. Любая правка = новый `dataset_id` со ссылкой на родителя. Именно это делает эксперименты воспроизводимыми и сравнимыми между собой.
- `__row_id__` = позиция в снимке. Он **никогда не признак**: живёт вне матрицы признаков и нужен только чтобы Head-to-Head и Error Analysis могли показать конкретные строки.

---

## 6. Dataset Package: контракт DataArena → ModelArena

**Предложение v1.** Пакет — директория или `.zip`:

```
package.json      обязателен  — манифест
data.parquet      обязателен  — одна материализованная таблица
profile.json      опционален  — профиль колонок DataArena
lineage.json      опционален  — рецепт, исходные файлы, версии
```

`package.json`:

```jsonc
{
  "format_version": 1,
  "package_id": "9f3c...",
  "name": "customers_clean",
  "created_at": "2026-09-06T18:20:00Z",
  "producer": { "name": "DataArena", "version": "0.4.1" },
  "dataset": {
    "file": "data.parquet",
    "format": "parquet",
    "row_count": 12840,
    "column_count": 31,
    "fingerprint_sha256": "e1b0..."      // ОБЯЗАТЕЛЕН, ModelArena его проверяет
  },
  "version": { "dataset_id": "cust", "version": 7, "parent_version": 6 },
  "schema": [
    { "name": "customer_id", "dtype": "int64", "semantic_type": "identifier",
      "nullable": false, "missing_ratio": 0.0, "unique_count": 12840,
      "is_probable_id": true }
  ],
  "hints": {                              // ТОЛЬКО подсказки, не обязательства
    "suggested_target": "churned",
    "suggested_group_column": "customer_id",
    "suggested_time_column": "signup_date",
    "excluded_candidates": [
      { "column": "churn_reason", "reason": "заполняется после события" }
    ]
  },
  "links": {
    "dataarena_base_url": "http://localhost:5174",
    "inspect_url_template": "/datasets/cust/v7?focus={column}"
  }
}
```

Правила, которые делают контракт безопасным:

1. **`hints` и `profile.json` — advisory.** ModelArena показывает их как «DataArena предполагает…», но **все числа, на которых строятся Readiness, Leakage и решения, пересчитывает сама** по `data.parquet`. Чужая статистика может устареть относительно данных.
2. **Fingerprint обязателен и проверяется.** Несовпадение = отказ импорта с внятным сообщением.
3. **Пакет — недоверенный вход.** Валидация по JSON Schema, никаких pickle/joblib внутри, никакого eval выражений, лимит размера.
4. **Обратная совместимость по `format_version`.** Неизвестные поля игнорируются, неизвестная major-версия — отказ с понятным текстом.

Транспорт:

- offline: пользователь выгружает `.zip` из DataArena → `POST /api/datasets/import` (multipart);
- online: `POST /api/datasets/import` с `{ "package_url": "..." }` — **ModelArena сама скачивает** (pull, а не push: так DataArena не нужно знать про очереди и авторизацию ModelArena);
- кнопка DataArena `Open in ModelArena` = deep link на фронт ModelArena `/#/import?package_url=...`, который дёргает тот же endpoint и ведёт пользователя сразу в Task Setup (§24, §30).

Ответственность DataArena по этому контракту — только выдать пакет и знать base URL ModelArena. Импорта кода нет ни в одну сторону.

---

## 7. Experiment domain model

```python
Dataset(dataset_id, fingerprint, row_count, column_count, schema, source,
        package_ref?, created_at)

TaskSpec(
    target_column, task_type,                    # binary | multiclass | regression
    positive_label?,                             # для binary — что считать «1»
    feature_columns, excluded_columns[(col, reason)],
    group_column?, time_column?,
    class_labels?,                               # фиксированный порядок меток на весь run
)

EvaluationProtocol(
    holdout: { enabled, strategy, test_size } | None,
    cv:      { splitter, n_splits, shuffle, n_repeats? },
    seed, derived_fold_seeds,
    fold_assignment_hash,                        # доказательство одинаковости split
    explanation, warnings, feasibility_notes
)

PreprocessingSpec(profile_key, numeric, datetime, categorical, ...)   # декларация, не объекты

Contender(contender_key, adapter_key, label, family, params, origin)
    # origin ∈ {default, suggested, tuned} — фиксирует, откуда взялись гиперпараметры

ArenaRun(
    run_id, created_at, name,
    dataset_id, dataset_fingerprint,
    task_spec, protocol, preprocessing_specs, contenders,
    resource_budget, status, progress, timings,
    environment { python, packages, os, cpu_count, threads }
)

ContenderResult(
    contender_key, status,                       # ok | failed | skipped(reason)
    fold_metrics[], cv_mean{}, cv_std{},
    holdout_metrics{}, train_metrics{},          # train — только для overfit-диагностики
    timings { fit_seconds, predict_seconds, predict_per_1k_rows },
    artifact_size_bytes, error?
)

Objective(preset, weights{quality, speed, stability, size, ...}, primary_metric)
Leaderboard      # вычисляемая проекция: results × objective
Champion(contender_key, objective, composite_score, breakdown[], justification, caveats[])
ExportedModel(bundle_path, feature_schema, metrics, dataset_fingerprint, refit_scope, version)
```

Хранение:

```
artifacts/runs/<run_id>/
  run.json                          # всё выше, кроме предсказаний и моделей
  experiment.yaml                   # воспроизводимый конфиг (§20)
  environment.json
  predictions/<contender>.parquet   # __row_id__, fold, split, y_true, y_pred, proba_*
  models/<contender>.joblib         # + sha256 в run.json
  events.jsonl                      # лог прогресса job
```

**Инвариант:** leaderboard, champion и composite score **не хранятся как истина** — они пересчитываются из `ContenderResult` + `Objective`. Иначе смена весов objective задним числом рассинхронизирует карточку с реальностью.

---

## 8. Model Adapter interface

```python
class Availability(NamedTuple):
    available: bool
    reason: str            # "lightgbm не установлен: pip install modelarena[boosters]"

class Applicability(NamedTuple):
    verdict: str           # "ok" | "caution" | "skip"
    reason: str            # обязателен, если не ok

class ModelAdapter(Protocol):
    key: str                       # "hist_gradient_boosting"
    label: str                     # "HistGradientBoosting"
    family: str                    # dummy|linear|tree_ensemble|gbdt|svm|neighbors
    supported_tasks: frozenset[str]
    preprocessing_profile: str     # scaled_onehot | tree_ordinal | native_missing | ...
    supports_proba: bool
    is_baseline: bool

    def availability(self) -> Availability: ...
    def applicability(self, ctx: DatasetContext) -> Applicability: ...
    def default_params(self, ctx: DatasetContext) -> dict: ...
    def build(self, params: dict, budget: ResourceBudget) -> BaseEstimator: ...
    def search_space(self, ctx: DatasetContext) -> dict: ...     # для Optuna
    def native_importance(self, fitted) -> dict | None: ...
```

`DatasetContext` — то, из чего адаптер принимает решения: `n_rows`, `n_features`, `effective_dim` (размерность **после** кодирования), кардинальности, доли пропусков, число классов, размер минорного класса, бюджет ресурсов.

`applicability` — это и есть §7 «ModelArena должна понимать applicability»:

| Адаптер | Правило | Вердикт |
|---|---|---|
| KNN | `n_rows > 200_000` | skip: «расстояния до 200k+ объектов на каждое предсказание — inference будет непригодным» |
| KNN | `effective_dim > 50` | caution: «в высокой размерности расстояния теряют смысл» |
| SVM (RBF) | `n_rows > 20_000` | skip: «обучение растёт как O(n²)…O(n³)» |
| LogisticRegression | `effective_dim > 5 · n_rows` | caution |
| любая one-hot модель | `effective_dim > 10_000` | caution: «взрыв размерности после кодирования» |
| CatBoost / LightGBM | не установлен | недоступен, **видимо в UI**, а не молча отсутствует |

Правило: адаптер **никогда не исчезает молча**. Он либо в списке, либо в списке с пометкой «пропущен, потому что …». Это часть честности §31.

**Регистрация:** явный реестр в `models/registry.py` + опциональные модули, которые не импортируются при отсутствии зависимости. Тяжёлые импорты ленивые (в старом проекте так уже сделано и отражено в ruff-конфиге).

---

## 9. Evaluation Protocol

**Схема по умолчанию: CV на train pool + нетронутый holdout.**

```
весь датасет
   ├── train_pool (80%)  ──► K-fold CV  ──► ранжирование, tuning, выбор Champion
   └── holdout    (20%)  ──► ОДИН замер каждым контендером в конце
```

- **Ранжирование идёт по CV mean, а не по holdout.** Так holdout остаётся защищённым от переобучения на выборе модели (§18): он не участвует ни в сравнении, ни в подборе гиперпараметров. Он показывается рядом как подтверждение — и если `holdout` заметно ниже `cv_mean`, интерфейс пишет об этом прямо.
- `cv_std` показывается всегда и участвует в оценке стабильности (§10).
- Все контендеры получают **одни и те же массивы индексов фолдов**. В `run.json` пишется `fold_assignment_hash`; тест проверяет его совпадение по всем контендерам. Это техническая гарантия §5 «одинаковый split для всех».

Выбор сплиттера (автоматически с объяснением, пользователь может переопределить):

| Условие | Сплиттер |
|---|---|
| есть group column | `StratifiedGroupKFold` (classification) / `GroupKFold` |
| есть time column | forward chaining (растущее окно), holdout = хвост по времени |
| classification, `min_class_count >= n_splits` | `StratifiedKFold` |
| иначе classification | `KFold` + предупреждение, что стратификация невозможна |
| regression | `KFold` |

Feasibility-проверки до запуска (иначе sklearn падает внутри job, и пользователь видит «FAILED» без причины):

- `min_class_count < n_splits` → уменьшить `n_splits` и **сказать об этом**;
- `n_groups < n_splits` → то же;
- `n_rows < 50` → CV с пятью фолдами даёт по 10 строк на валидацию — отдельное предупреждение;
- пустой фолд или отсутствие класса в фолде → метрика для этого фолда не считается, а не считается нулём.

Seeds: один `seed` эксперимента → детерминированные производные для сплиттера, каждой модели и bootstrap. Все записываются.

Оговорка про воспроизводимость: **обещаем воспроизводимость конфигурации, а не бит-в-бит идентичность чисел**. Порядок редукции float в многопоточном GBDT зависит от числа потоков, поэтому в `environment.json` пишется число потоков, а в README честно сказано, где возможны расхождения в последних знаках.

---

## 10. Preprocessing

Правило §6 соблюдается **структурно, а не дисциплиной**: все трансформеры лежат внутри `Pipeline`, и `fit` вызывается только на train-части каждого фолда. Никакой глобальной подготовки до split не существует в принципе — нет функции, которая могла бы это сделать.

Профили препроцессинга (адаптер выбирает свой):

| Профиль | Numeric | Datetime | Categorical | Для кого |
|---|---|---|---|---|
| `scaled_onehot` | median impute + StandardScaler | календарные части + impute + scale | most_frequent + OneHot(min_frequency, max_categories) | linear, SVM, KNN |
| `tree_ordinal` | median impute, без scaling | календарные части | OrdinalEncoder(handle_unknown) | RandomForest, ExtraTrees |
| `native_missing` | без impute | календарные части | ordinal / native | HistGradientBoosting, LightGBM, XGBoost |
| `native_categorical` | без impute | календарные части | передаётся как есть | CatBoost |
| `passthrough_dummy` | — | — | — | baseline |

Это осознанное отступление от «одинакового препроцессинга для всех», и его нужно оговорить явно: **честность сравнения определяется одинаковым split, одинаковым target, одинаковым набором признаков и одинаковым seed**, а не одинаковым кодированием. Заставлять GBDT работать через StandardScaler + OneHot — это не «честно», это искусственно ухудшать одного участника. В карточке эксперимента для каждого контендера записывается его профиль, чтобы это было видно.

Что **запрещено** в v1: target encoding и любые статистики, зависящие от target, вне фолда. Если добавим позже — только внутри `Pipeline` с fold-wise fit и с отдельным тестом на утечку.

---

## 11. Dataset Readiness

Отвечает на вопрос «насколько этот датасет готов **для этой задачи**», поэтому считается **после** выбора target и task type и **с учётом выбранного протокола**.

Никакого 0–100 score. Каждая проверка возвращает finding:

```python
Finding(key, severity, title, columns[], evidence{}, what_it_means, suggested_action)
severity ∈ {info, caution, high_risk}
```

Итог — худший severity: `READY` / `CAUTION` / `HIGH RISK` плюс список причин.

Проверки (все пороги **относительные**, не «>10 000 строк = хорошо»):

| Проверка | Контекстный порог |
|---|---|
| Размер выборки vs размерность | `n_rows / effective_dim`. 900×4 → ok; 900×500 → HIGH RISK: «на признак приходится меньше двух наблюдений» |
| Минорный класс в штуках | `min_class_count`. 14 объектов при 5-fold → ~2.8 на фолд → HIGH RISK с расчётом, а не с абстрактной формулировкой |
| Дисбаланс | доля мажорного класса + список метрик, которые становятся обманчивыми |
| Target | константный / почти константный / пропуски в target / регрессионный target с шестью уникальными значениями |
| Пропуски | по колонкам; отдельно колонки с >50%; отдельно **пропуски, коррелирующие с target** (это и readiness-сигнал, и кандидат в leakage) |
| Константные и near-constant признаки | доля мажорного значения ≥ 0.99 |
| Дубликаты колонок | точные и почти точные копии |
| Высокая кардинальность | вклад в `effective_dim` |
| Редкие категории | уровни, которых меньше `n_splits` — они не переживут разбиение по фолдам |
| Дубли строк | всего и пересекающие фолды |
| Кандидаты в идентификаторы | почти уникальные, монотонные счётчики |
| Связь признак↔target | дешёвый одно-признаковый пробник: ловит и «сигнала нет вообще», и «сигнала подозрительно много» |
| Feasibility протокола | можно ли вообще построить выбранный split |

Формулировка результата всегда включает **что это значит для обучения**, а не только факт. Пример: не «minority class = 14», а «минорный класс — 14 объектов; при 5-fold в валидационном фолде окажется около трёх объектов, поэтому recall по нему будет прыгать на десятки процентов между фолдами, и сравнение моделей по нему ненадёжно».

---

## 12. Leakage Guard

Берётся целиком из старого `leakage.py` (восемь проверок, severity, DISCLAIMER, формулировки «сигнал, не доказательство») и расширяется:

- `+` group leakage: одна и та же сущность в train и validation при **фактически выбранном** протоколе;
- `+` missingness-паттерн, почти идеально разделяющий классы;
- `+` дубли строк, пересекающие фолды (не только holdout);
- `+` признак, доступный только после события — остаётся эвристикой по имени с severity `info`;
- `−` preprocessing leakage не проверяется эвристикой: он **исключён структурно** и покрыт тестом.

Действие пользователя: исключить признак → он попадает в `TaskSpec.excluded_columns` с причиной и виден в карточке эксперимента.

Формулировка везде: «Potential leakage candidate because …». Никогда «это утечка».

---

## 13. MVP contenders

**Ядро без тяжёлых зависимостей** (только sklearn) — установка остаётся лёгкой:

| Classification | Regression |
|---|---|
| `DummyClassifier` (most_frequent + stratified) — **всегда** | `DummyRegressor` (median) — **всегда** |
| LogisticRegression | Ridge |
| RandomForest | RandomForest |
| ExtraTrees | ExtraTrees |
| HistGradientBoosting | HistGradientBoosting |

**Опциональные адаптеры** (extra `modelarena[boosters]`): LightGBM, XGBoost, CatBoost.
**Guarded** (включаются только по applicability): SVM, KNN.

Baseline не отключается (§8). Leaderboard всегда показывает колонку «прирост над baseline» — без неё F1 = 0.88 ничего не значит.

Auto Suggest для §30 предлагает подмножество по `DatasetContext`: на 800×12 имеет смысл всё ядро; на 2 000 000×40 — только HistGBM/LightGBM/линейная, а RandomForest помечается caution по времени.

---

## 14. Метрики

**Binary classification:** accuracy, balanced_accuracy, precision, recall, F1, ROC-AUC, PR-AUC, log loss, Brier, MCC.
**Multiclass:** accuracy, balanced_accuracy, precision/recall/F1 в macro и weighted, per-class, ROC-AUC (ovr macro, когда вычислим), log loss.
**Regression:** MAE, RMSE, R², MedAE; MAPE — **только если** `min(|y|)` не близок к нулю (иначе метрика отсутствует с указанием причины); опционально sMAPE.

Правила:

- у каждой метрики есть `direction`, описание и условие применимости; всё это отдаётся в API, чтобы фронт не переизобретал «меньше — лучше»;
- **невычислимая метрика отсутствует, а не равна нулю** (иначе она молча топит модель в ранжировании);
- primary metric выбирается по task type и дисбалансу, но **пользователь может сменить**;
- для CV каждая метрика идёт парой `mean ± std` по фолдам, для holdout — с bootstrap-интервалом.

**Операционные метрики:** `fit_seconds`, `predict_seconds`, `predict_per_1k_rows`, `artifact_size_bytes` — измеряются надёжно. Память — только best-effort (peak RSS дочернего процесса) и **помечается как приблизительная**; если измерение окажется нестабильным на Windows, поле не показывается вообще, а не показывается неверным (§9: «если измерение реализовано надёжно»).

---

## 15. Objective, composite score, Champion

Пресеты: `Quality`, `Speed`, `Lightweight`, `Stable`, `Balanced` + произвольные веса.

Нормализация внутри одного run (сравниваются только участники этого прогона):

- **quality**: `(score − baseline_score) / (best_score − baseline_score)`, обрезано в [0, 1]. Это нормализация **относительно baseline**, а не min-max по контендерам: она напрямую отвечает на вопрос «сколько пользы сверх тривиального предсказания», и модель хуже baseline получает 0, а не «просто последнее место»;
- **speed / latency / size**: min-max по `log10(value)` с инверсией — иначе одна очень медленная модель сплющивает всех остальных в один пиксель;
- **stability**: `1 − min(1, cv_std / разброс_между_контендерами)`. Смысл: «дисперсия этой модели по фолдам мала по сравнению с различиями между моделями».

```
composite = Σ wᵢ · nᵢ ,  Σ wᵢ = 1
```

UI обязан показывать **разложение**: какой критерий сколько баллов дал (в старом проекте так сделан Quality Score — хороший прецедент, паттерн переносим).

Правило честности: если разрыв между первым и вторым **меньше `cv_std` лидера**, Champion объявляется с явной оговоркой «разница в пределах шума; при равном качестве выбран более быстрый/простой» — и это записывается в `Champion.caveats`.

Смена objective **не требует переобучения**: пересчёт по сохранённым результатам.

---

## 16. Head-to-Head и диагностика (на сохранённых предсказаниях)

Так как `predictions/<contender>.parquet` содержит `__row_id__, fold, y_true, y_pred, proba_*`, всё это считается join'ом двух таблиц:

- квадрант 2×2: оба правы / оба ошиблись / A прав, B ошибся / B прав, A ошибся;
- клик по квадранту → список `__row_id__` → подтягиваем строки из снимка датасета для показа. **Target здесь показывается как факт (`y_true`)**; это не утечка, потому что показ идёт после обучения и не влияет на модель;
- разница метрик, CV-стабильность, время, размер, калибровка;
- сравнение confusion matrix и распределений вероятностей.

Диагностика (§12, §14, §15, §16):

- confusion matrix, per-class метрики, списки FP/FN, распределение вероятностей, «трудные» объекты;
- регрессия: остатки, крупнейшие промахи, распределение ошибки, predicted vs actual;
- **overfitting**: train vs CV vs holdout + gap + `cv_std`; вывод только при согласии нескольких сигналов, никогда по одному числу;
- **learning curves**: доли train (10…100%), тот же протокол, кривые train/validation с доверительной полосой. Интерпретация только качественная: «кривая валидации ещё растёт», «вышла на плато», «большой разрыв — high variance». Никаких «добавьте 20 000 строк → F1 0.95»;
- **calibration**: reliability diagram, Brier, распределение уверенности; текстом отделяем «модель классифицирует правильно» от «вероятности откалиброваны»;
- **failure slices** — этап 8+, с обязательным `support`, минимальным размером среза, поправкой на множественные сравнения и пометкой «exploratory».

---

## 17. Tuning (§18)

```
Stage 1  дешёвый турнир: default params, CV, все применимые адаптеры
Stage 2  выбор top-N (по умолчанию 3) по composite score
Stage 3  Optuna на финалистах, ТОЛЬКО на train_pool, во внутреннем CV
Stage 4  честная переоценка: тот же внешний протокол, обновлённый leaderboard
```

Holdout не участвует в Stage 3 вообще. В карточке контендера `origin = tuned` и записывается число trials: тюненная модель получила больше вычислительного бюджета, чем остальные, это само по себе делает сравнение неравным, и об этом нужно писать прямо.

---

## 18. Jobs и ресурсы

MVP-архитектура без Celery/Redis, но с возможностью замены:

- `JobQueue` (Protocol) + `LocalJobQueue`: поток-оркестратор + **`ProcessPoolExecutor` для самих обучений**. Процессы, а не потоки, потому что (а) настоящий cancel, (б) изоляция по памяти, (в) GIL;
- состояния `QUEUED → RUNNING → COMPLETED | FAILED | CANCELLED`, персистятся на диск, чтобы после рестарта не было «вечного RUNNING»;
- прогресс: `GET /api/runs/{id}/events` (SSE) с fallback на polling; события пишутся в `events.jsonl`;
- cancel: терминирование воркеров + пометка незавершённых контендеров как `cancelled`;
- **ResourceBudget**: `max_parallel_fits`, `threads_per_fit`, `memory_budget_mb`. Оркестратор задаёт `n_jobs` / `nthread` / `thread_count` каждому estimator явно; `n_jobs=-1` в коде запрещён. По умолчанию `max_parallel_fits × threads_per_fit ≤ cpu_count − 1`;
- перед стартом контендера — оценка `effective_dim × n_rows × 8 байт` против бюджета; при превышении контендер помечается `skipped(reason)`, а не убивает машину.

---

## 19. Артефакты, экспорт, воспроизводимость

Экспорт Champion — **bundle, а не estimator**:

```
champion_bundle/
  model.joblib          # весь Pipeline: препроцессинг + модель
  manifest.json         # feature schema, task, метки классов, метрики, protocol,
                        # dataset fingerprint, refit_scope, версии пакетов, sha256
  experiment.yaml       # конфиг для повторения прогона
  README.md             # как загрузить и как вызвать predict
```

- `refit_scope` явно говорит, обучен ли экспортируемый артефакт на train или переобучен на 100% данных. Старый проект переобучал на всех данных и честно это писал — поведение хорошее, но должно быть **осознанным выбором в UI**, а не невидимым дефолтом;
- загрузка артефактов только из собственного каталога и только после проверки SHA-256 (joblib = произвольный код при десериализации). Пользовательские `.joblib` не загружаются никогда;
- ONNX — **не обещаем**. Для части моделей корректный экспорт невозможен; добавим точечно там, где он реально работает, и напишем список поддерживаемых.

---

## 20. Drift и деградация (§22–23)

При обучении сохраняется baseline-профиль: схема, распределения признаков, доля пропусков, распределение категорий, распределение предсказаний на holdout.

Загрузка нового датасета → отчёт, разделённый на **два независимых блока**:

```
DATA DRIFT                          MODEL PERFORMANCE
schema drift                        доступно ТОЛЬКО при наличии labels
feature drift (PSI / KS)            метрики на новых данных
missingness drift                   калибровка на новых данных
categorical drift                   сравнение с holdout
prediction drift
```

Если labels нет — интерфейс прямо пишет: **«качество модели по этим данным неизвестно; изменились только входные распределения»**. Никаких «модель деградировала».

Degradation forecasting — только при накопленной истории снимков (t1, t2, t3, …), с показом неопределённости и допущений. Без истории — только текущие сигналы риска. Никаких «модель деградирует через 14 дней».

Здесь же переиспользуется `assess_novelty` из старого проекта: если новые строки совпадают с обучающими, отчёт говорит об этом прямо.

---

## 21. Структура репозитория

```
ModelArena/
├── README.md                     # русский, §34
├── pyproject.toml                # extras: boosters, tuning, shap, dev
├── contracts/
│   └── dataset-package/v1/schema.json + fixtures/
├── backend/
│   ├── core/            config errors logging ids hashing
│   ├── datasets/        io registry snapshot package_import fingerprint cache
│   ├── tasks/           spec target_inference feature_selection
│   ├── readiness/       checks/ report
│   ├── leakage/         checks/ report
│   ├── protocol/        splitters plan feasibility
│   ├── preprocessing/   profiles factory transformers
│   ├── models/          registry applicability adapters/
│   ├── arena/           runner orchestration resources
│   ├── evaluation/      metrics aggregation predictions bootstrap
│   ├── scoring/         objectives normalization composite champion
│   ├── diagnostics/     overfitting learning_curves calibration
│   │                    error_analysis slices head_to_head
│   ├── explain/         importance permutation shap(optional)
│   ├── tuning/          study spaces stages
│   ├── experiments/     records store reproducibility
│   ├── artifacts/       bundle export loading
│   ├── monitoring/      profiles drift comparison
│   ├── jobs/            queue worker events
│   └── api/             routes/ schemas/ app
├── frontend/            React 19 + TS + Vite
├── tests/               unit/ integration/ leakage/ contracts/
├── examples/            benchmark datasets + скрипты
└── docs/
```

Правило: ни один модуль домена не импортирует `api`. `arena/runner` оркестрирует, но сам метрики не считает.

---

## 22. Frontend

React 19 + TypeScript + Vite. Разделы: `Datasets · Task Setup · Arena · Leaderboard · Head-to-Head · Diagnostics · Experiments · Models`.

Тёмная база, violet / wine-red акценты. Метафора арены — в словах (`Champion`, `Contenders`, `Head-to-Head`), не в визуале. Никаких HP-баров.

Beginner flow (§30) — линейный, шесть шагов, на каждом одно решение. Advanced settings — сворачиваемые панели: не мешают, но дают доступ к метрикам, split, препроцессингу, моделям, гиперпараметрам, весам objective и лимитам ресурсов.

Реально нужные графики: learning curves, reliability diagram, PR/ROC, распределение остатков, confusion matrix, гистограммы, bar-charts важности. Хардкодить всё это на голом SVG — заметный объём работы; выбор библиотеки вынесен в открытые решения.

---

## 23. Тесты

**Unit:** splitters (включая feasibility-откаты), профили препроцессинга, каждый адаптер (build + applicability), метрики (включая отсутствие невычислимых), нормализация и composite score, ранжирование, readiness-проверки, leakage-проверки, сериализация артефактов, парсер и валидатор Dataset Package.

**Integration:** dataset → task → readiness → protocol → arena → leaderboard → champion → export, отдельно для classification (binary и multiclass) и regression.

**Специальные тесты на честность** — это то, что отличает проект:

1. `test_no_preprocessing_leakage` — подмена значений в validation-фолде не должна менять обученные статистики трансформеров;
2. `test_same_split_for_all_contenders` — `fold_assignment_hash` совпадает у всех;
3. `test_holdout_never_used_for_ranking` — при tuning holdout не читается (проверяется обёрткой над доступом);
4. `test_group_never_crosses_folds`;
5. `test_temporal_train_before_test`;
6. `test_absent_metric_is_absent_not_zero`;
7. `test_objective_change_does_not_retrain`;
8. `test_leaked_column_is_flagged` — синтетический датасет с колонкой, вычисленной из target.

**Contract tests:** golden-фикстуры Dataset Package v1; те же файлы должны лежать в DataArena.

---

## 24. Benchmark datasets

Нужны маленькие публичные наборы под разные ситуации: сбалансированная классификация, сильный дисбаланс, multiclass, регрессия, набор с пропусками, набор с категориальными признаками, набор, склонный к переобучению (мало строк, много признаков), набор с групповой структурой и набор со временем.

Правило §33: **их результаты не попадают в README, пока эксперимент реально не воспроизведён** запущенным кодом.

---

## 25. Roadmap

| Этап | Содержание | Готово, когда |
|---|---|---|
| 0 | Скелет репозитория, конфиг, ошибки, контракт Dataset Package v1 + фикстуры | schema валидируется, CI зелёный |
| 1 | Dataset import (upload + package), snapshot registry, Task Setup, автоопределение task type | датасет импортируется, схема видна, target выбирается |
| 2 | Dataset Readiness | реальные findings на benchmark-датасетах, тесты порогов |
| 3 | Protocol + профили препроцессинга + гарантии честности | fold hash одинаков, тесты на утечку препроцессинга зелёные |
| 4 | Adapter registry + baseline + четыре модели ядра | сравнение на одном split реально работает |
| 5 | Arena engine + jobs + resource budget + cancel + progress | длинный прогон не вешает HTTP, cancel работает |
| 6 | Metrics + CV aggregation + Leaderboard + objective/composite/Champion | смена весов пересчитывает leaderboard без переобучения |
| 7 | Head-to-Head на сохранённых предсказаниях | квадранты кликабельны, строки открываются |
| 8 | Diagnostics: overfitting, learning curves, calibration, error analysis | кривые реальные, интерпретации осторожные |
| 9 | Experiment tracking + experiment.yaml + environment | старый прогон открывается и понятен |
| 10 | Optuna tuning финалистов | holdout доказуемо не участвует |
| 11 | Model export bundle | выгруженный bundle реально предсказывает из внешнего скрипта |
| 12 | DataArena integration: импорт по URL + Open in ModelArena + Inspect in DataArena | end-to-end с реальной DataArena |
| 13 | Drift / future dataset comparison | два блока разделены, без labels — «неизвестно» |
| 14 | Failure slices, SHAP (опционально), производительность, polish, README со скриншотами | реальный воспроизведённый эксперимент в README |

После каждого этапа: tests, ruff, mypy, frontend build, ручной smoke, commit, checkpoint.

---

## 26. Основные ML-риски

| # | Риск | Митигация |
|---|---|---|
| R1 | **Preprocessing leakage** — статистики видят validation | Всё в `Pipeline`, fit только на train-фолде; нет API, который мог бы сделать иначе; тест |
| R2 | **Несопоставимое сравнение** — разные split / признаки / seed | `fold_assignment_hash` + тест; протокол строится один раз и передаётся всем |
| R3 | **Переобучение на выборе модели** (winner's curse) | Ранжирование по CV, holdout один раз; вывод «разница в пределах шума» при разрыве < `cv_std` |
| R4 | **Test set как цель оптимизации** | Tuning только на train_pool во внутреннем CV; отдельный тест |
| R5 | **Множественные сравнения в failure slices** | Минимальный support, поправка на число проверенных срезов, ярлык exploratory |
| R6 | **Скрытая балансировка ломает калибровку** | `class_weight` — явное решение run, одинаковое для всех, записано в конфиг |
| R7 | **Refit на 100% ≠ измеренная модель** | `refit_scope` в манифесте, явный выбор в UI |
| R8 | **Игнорирование group / time структуры** | Детекторы + предупреждения + leakage-проверка на пересечение групп |
| R9 | **Drift принимают за деградацию** | Два раздельных блока; без labels — прямо «неизвестно» |
| R10 | **Псевдопрогноз деградации** | Только при истории снимков, с неопределённостью и допущениями |
| R11 | **Исчерпание ресурсов** | Бюджет процессов/потоков/памяти, оценка `effective_dim` до fit, `n_jobs=-1` запрещён |
| R12 | **Недетерминизм** | Seeds всюду, число потоков в environment; обещаем воспроизводимость конфига, не бит-в-бит |
| R13 | **Невычислимые метрики как нули** | Отсутствие вместо нуля + тест |
| R14 | **joblib = произвольный код** | Загрузка только из своего каталога и только после SHA-256; пользовательские артефакты не грузим |
| R15 | **Доверие чужой статистике из пакета DataArena** | `hints` / `profile` advisory; все решающие числа пересчитываются; fingerprint проверяется |
| R16 | **MAPE при y≈0, ROC-AUC при отсутствующем классе в фолде** | Проверки применимости с указанием причины отсутствия |
| R17 | **Тюненная модель против нетюненных** | `origin=tuned` и число trials видны в leaderboard |

---

## 27. Открытые решения — нужен ответ до реализации

1. **База ранжирования.** Рекомендую CV mean + holdout как подтверждение (раздел 9). Альтернатива — holdout как основа, CV как проверка стабильности. Влияет на весь Arena engine.
2. **Бустеры в MVP.** Рекомендую опциональные extras (ядро = только sklearn, лёгкая установка, адаптеры видны как «не установлено»). Альтернатива — LightGBM/CatBoost в основных зависимостях.
3. **Графики на фронте.** Хардкод SVG (ноль зависимостей, но заметный объём работы) против лёгкой библиотеки (uPlot / Recharts). Рекомендую лёгкую библиотеку — диагностика ModelArena живёт графиками.
4. **GitHub-репозитории.** `gh` авторизован под `xenonim-ctrl`, но переменная окружения `GH_TOKEN` содержит невалидный токен и перебивает keyring — создание репозитория сейчас упадёт. Нужно решить: аккаунт (`xenonim-ctrl`) или организация (`forit-tech`, где лежит AutoDataAnalysis), public или private, и починить `GH_TOKEN`.
5. **Судьба старого репозитория `forit-tech/AutoDataAnalysis`.** Архивировать с пометкой «разделён на DataArena и ModelArena» или оставить как есть.
