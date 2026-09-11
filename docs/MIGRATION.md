# Migration map: AutoDataAnalysis → ModelArena

Потабличная карта переноса. Источник: `forit-tech/AutoDataAnalysis`, ветка `main`,
коммит `7dd8049`. Счётчики строк получены `wc -l` на этом коммите, не по памяти.

Правило переноса — [D-12](DECISIONS.md#d-12-исключение-из-правила-переносить-ml-код-без-изменений):
логика не меняется, кроме двух модулей, которые переписываются.

---

## 0. Сводка

| Слой | LOC | Переносится | Переписывается |
|---|---|---|---|
| `backend/ml/` | 3 959 | 3 212 | 747 |
| `backend/api/routes/{training,experiments,inference}.py` + `schemas/ml.py` | 447 | адаптируется целиком | — |
| `backend/{core,datasets}/` + `api/{app,uploads}.py` + `profiling/` без quality | 1 395 | 1 395 | — |
| frontend: `features/{models,evaluate,test-drive}`, `LeakagePanel`, `types/ml.ts` | 3 064 | адаптируется | — |
| тесты ML (`test_{training,leakage,splits,experiments,inference}.py`) | 1 065 | переносятся вместе с кодом | — |

Уточнение к прежним оценкам: в `SCOPE.md` §2 и `DataArena/docs/MIGRATION_MAP.md` §1.7
фигурировали «≈3 400» и «2 984» строк `backend/ml/`. Фактическое число — **3 959**.
Расхождение не меняет решений, но здесь используется проверенное значение.

---

## 1. `backend/ml/` — потабличный вердикт

Обозначения: 🟢 перенос как есть · 🟡 перенос с адаптацией · 🔴 переписывается

### 1.1 Перенос как есть — 2 155 LOC

| Файл | LOC | Куда | Что меняется |
|---|---|---|---|
| 🟢 `ml/validation/leakage.py` | 487 | `backend/leakage/` | Разбивается на `checks/*.py` по одной проверке в файле; логика и тексты не трогаются. Добавляются две новые проверки отдельными файлами (group crossing, missingness) |
| 🟢 `ml/validation/splits.py` | 460 | `backend/protocol/` | `find_time_columns`, `find_group_columns`, `recommend_split` — без изменений. `build_split_plan` становится частью `plan.py` и дополняется CV (сам holdout-код сохраняется) |
| 🟢 `ml/evaluation/error_analysis.py` | 366 | `backend/diagnostics/error_analysis.py` | Вход меняется с «предсказания лучшей модели» на «предсказания любого контендера из `predictions/*.parquet`» — сигнатура, не логика |
| 🟢 `ml/explainability/importance.py` | 318 | `backend/explain/` | Без изменений. `explain_single_prediction` (окклюзия) сохраняет честное имя — это не SHAP и так и называется |
| 🟢 `ml/experiments/store.py` | 229 | `backend/experiments/store.py` | `Protocol` + файловая реализация + SHA-256-проверка артефакта переносятся полностью. Добавляется запись `predictions/` и `diagnostics/` |
| 🟢 `ml/evaluation/metrics.py` | 196 | `backend/evaluation/metrics.py` | Дополняется log loss, Brier, MCC, per-class, MAPE с проверкой применимости. Существующие функции и правило «нет метрики → нет ключа» не трогаются |
| 🟢 `ml/evaluation/bootstrap.py` | 99 | `backend/evaluation/bootstrap.py` | Без изменений. Применяется к holdout-предсказаниям |

### 1.2 Перенос с адаптацией — 1 057 LOC

| Файл | LOC | Куда | Что меняется |
|---|---|---|---|
| 🟡 `ml/inference/predictor.py` | 438 | `backend/monitoring/novelty.py` + `backend/artifacts/loading.py` | `validate_inference_schema` и `assess_novelty` идут в monitoring (нужны для сравнения с будущим датасетом). Предсказание на новых данных — после этапа 11, вне MVP |
| 🟡 `ml/evaluation/verdict.py` | 218 | `backend/scoring/champion.py` | Формулировки вердиктов сохраняются, но «лучшая модель» больше не выбирается по одной метрике: вход — composite score и objective (D-1, ТЗ §10) |
| 🟡 `ml/training/preprocessing.py` | 166 | `backend/preprocessing/` | `DateTimeFeatures` и `split_feature_types` без изменений. `build_preprocessor` превращается в фабрику четырёх профилей вместо одного (D-7) |
| 🟡 `ml/training/config.py` | 115 | `backend/tasks/spec.py` + `backend/protocol/plan.py` | Валидация настроек сохраняется; `TrainingConfig` разделяется на `TaskSpec` и `EvaluationProtocol` — это два разных решения с разным временем жизни |
| 🟡 `ml/experiments/models.py` | 113 | `backend/experiments/records.py` | `FeatureSchema` без изменений. `ExperimentRecord` заменяется доменной моделью `ArenaRun` + `ContenderResult` (DESIGN §2) |
| 🟡 `ml/*/__init__.py` | 7 | — | — |

### 1.3 Переписывается — 747 LOC

| Файл | LOC | Во что | Почему не переносится |
|---|---|---|---|
| 🔴 `ml/training/trainer.py` | 604 | `backend/arena/runner.py`, `orchestration.py`, `protocol/plan.py`, `evaluation/aggregation.py` | Функция `train_models` делает десять дел: готовит фрейм, определяет тип задачи, строит split, зовёт leakage, строит препроцессор, обучает кандидатов, выбирает лучшего, считает bootstrap, собирает объяснимость, пишет эксперимент. ТЗ §26 прямо против такого модуля. Внутри есть куски, которые переезжают почти дословно: `_prepare_frame`, `_resolve_task_type`, `_predict_probabilities` (выравнивание вероятностей по общему порядку меток), `_build_feature_schema`, `build_row_hashes` |
| 🔴 `ml/training/catalog.py` | 143 | `backend/models/adapters/*` + `registry.py` | Хардкод четырёх моделей вместо adapter system (D-2). Содержит два дефекта, которые нельзя переносить: `n_jobs=-1` (при параллельном обучении = «N моделей × все ядра», ТЗ §28) и `class_weight="balanced"` только у части моделей (ломает сопоставимость log loss, Brier, PR-AUC). `ModelDefinition.parameters` — идея сохранения только явно заданных гиперпараметров — переносится в адаптеры |

### 1.4 Не переносится

| Файл | LOC | Почему |
|---|---|---|
| `profiling/quality.py` | 102 | Quality Score остаётся в DataArena. В ModelArena readiness не сводится к числу (D-10) |
| `etl/*` | ~865 | DataArena |
| `dataview/*` | ~242 | DataArena |
| `ai/*`, `routes/ai.py` | ~266 | Слой пересказа чисел; к выбору модели отношения не имеет. Остаётся в архиве AutoDataAnalysis |

---

## 2. Общая инфраструктура и профилирование — 1 395 LOC

| Файл | LOC | Вердикт |
|---|---|---|
| 🟢 `core/errors.py` | 70 | Как есть. Единый формат `{"error": {...}, "detail": "..."}` сохраняется — фронт уже умеет его разбирать |
| 🟢 `core/config.py` | 81 | Как есть, с заменой префикса переменных `ADA_` → `MARENA_` |
| 🟢 `api/app.py` | 127 | Как есть: фабрика приложения, обработчики ошибок, CORS |
| 🟢 `datasets/io.py` | 254 | Как есть. Единственная точка чтения датасета; поддержка форматов нужна для standalone-импорта (D-13) |
| 🟢 `datasets/cache.py` | 76 | Как есть. LRU по fingerprint с двумя границами (число таблиц и байты) |
| 🟡 `api/uploads.py` | 92 | Адаптируется: загрузка `.dapkg.zip` вдобавок к табличным файлам |
| 🟡 `profiling/profiler.py` | 373 | Профиль колонок нужен для карточки датасета и как baseline для drift. Переносится **без** вызова Quality Score |
| 🟡 `profiling/models.py` | 114 | То же, без `QualityScore` и `QualityPenalty` |
| 🟡 `profiling/comparison.py` | 206 | Основа drift-отчёта: структура «метрика / before / after / delta / direction» переиспользуется для сравнения датасетов во времени |
| ❗ `profiling.safe_ratio` | 4 | Дублируется в `core/`. Тянуть весь `profiling` ради одной функции — лишняя связь (риск T-2) |

---

## 3. HTTP-слой и схемы — 447 LOC

| Файл | LOC | Вердикт |
|---|---|---|
| 🟡 `api/routes/training.py` | 56 | Логика сохраняется, но `POST /models/train` с `UploadFile` заменяется на `POST /runs` с `dataset_id` и постановкой job (обучение больше не висит в HTTP-запросе, ТЗ §27) |
| 🟡 `api/routes/experiments.py` | 52 | Почти как есть: список, карточка, удаление |
| 🟡 `api/routes/inference.py` | 113 | После этапа 11 |
| 🟡 `schemas/ml.py` | 226 | Переписывается под новую доменную модель, но как **справочник полей**, которые уже проверены практикой |

---

## 4. Frontend — 3 064 LOC

| Что | LOC | Вердикт |
|---|---|---|
| `features/models/*` | 1 166 | Основа разделов Task Setup и Arena. `targetSelection.ts` с тестами переносится как есть |
| `features/evaluate/*` | 890 | Основа Diagnostics и Experiments |
| `features/test-drive/*` | 563 | После этапа 11 |
| `components/LeakagePanel.tsx` | 54 | Как есть |
| `types/ml.ts` | 391 | Переписывается: типы генерируются из OpenAPI, а не пишутся руками (риск T-10) |

Дополнительно из общей части переносятся: транспортный слой `api/client.ts`, `ErrorBoundary`,
дизайн-токены. Charting — новый код на Recharts (D-3), в старом проекте графиков нет.

---

## 5. Тесты — 1 065 LOC

| Файл | LOC | Вердикт |
|---|---|---|
| `test_leakage.py` | 188 | Переносится целиком. Эталон того, как надо тестировать эвристики |
| `test_splits.py` | 232 | Переносится, дополняется CV-сплиттерами и лестницей выполнимости |
| `test_training.py` | 249 | Переписывается вместе с `trainer.py`, но сценарии сохраняются как чеклист |
| `test_experiments.py` | 181 | Переносится, дополняется `predictions/` |
| `test_inference.py` | 215 | После этапа 11 |
| `conftest.py` | 104 | Переносится **первым** (риск T-3) |

Новые тесты, которых в старом проекте нет и которые обязательны (DESIGN §2.3, §4.4):
`test_same_split_for_all_contenders`, `test_holdout_never_used_for_ranking`,
`test_no_preprocessing_leakage`, `test_objective_change_does_not_retrain`,
`test_absent_metric_is_absent_not_zero`, `test_row_id_never_a_feature`,
`test_golden_package_reads`, `test_unknown_major_version_rejected`.

---

## 6. Порядок переноса

Снизу вверх, чтобы ни один перенесённый модуль не ждал ещё не перенесённого (риск T-2):

```
1. conftest.py + фикстуры
2. core/{errors,config} → datasets/{io,cache} → api/app
3. profiling/{profiler,models} без quality
4. protocol/  ← splits.py
5. leakage/   ← leakage.py
6. preprocessing/ ← preprocessing.py (сразу в виде профилей)
7. evaluation/ ← metrics, bootstrap
8. models/    ← НОВОЕ (адаптеры вместо catalog.py)
9. arena/     ← НОВОЕ (вместо trainer.py)
10. scoring/  ← verdict.py + новое
11. diagnostics/ ← error_analysis.py + новое
12. explain/  ← importance.py
13. experiments/ ← store.py + records
14. monitoring/ ← predictor.py + comparison.py
```

Модуль считается перенесённым, только когда его тесты зелёные **в новом репозитории**.
Перенос без тестов не засчитывается.

---

## 7. Чеклист закрытия миграции (условие архивации AutoDataAnalysis, D-5)

- [ ] все 🟢 и 🟡 модули перенесены, тесты зелёные в ModelArena;
- [ ] `trainer.py` и `catalog.py` заменены и покрыты новыми тестами;
- [ ] frontend-разделы Model / Evaluate работают на новом API;
- [ ] ни один импорт из `AutoDataAnalysis` или `DataArena` не остался (проверка анализом импортов);
- [ ] сценарий из README старого проекта воспроизводится в паре DataArena + ModelArena;
- [ ] README `AutoDataAnalysis` дополнен строкой «Project split into DataArena and ModelArena» со ссылками;
- [ ] только после этого — архивация.
