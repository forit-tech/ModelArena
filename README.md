# ModelArena

**Подбор, обучение и разбор моделей на подготовленном датасете.**

Проект FORIT TECH.

---

> ## Текущее состояние: Этап 1 — вход и постановка задачи
>
> Этап 0 закрыт: границы, решения и дизайн зафиксированы, контракт Dataset Package v1
> заморожен (тег `contract/dataset-package-v1.0.0`) и проверен независимой реализацией
> потребителя — 14 фикстур из 14.
>
> Сейчас: скелет, чтение пакета, реестр снимков, постановка ML-задачи. Обучения пока нет.

---

## Зачем

ModelArena начинается там, где заканчивается [DataArena](../DataArena): датасет уже собран,
почищен и выгружен. Задача ModelArena — подобрать под него модель, честно показать метрики,
объяснить, где модель ошибается, и подсказать, что делать с артефактами дальше.

**ModelArena не читает сырые файлы, не редактирует данные и не строит join.**
Если данные плохие — их чинят в DataArena.

## Что планируется

- чтение Dataset Package из DataArena;
- обучение и сравнение кандидатов, включая обязательный baseline;
- стратегии валидации: random, stratified, temporal, group + автовыбор;
- кросс-валидация;
- инспекция утечек данных;
- разбор ошибок и объяснимость;
- диагностика переобучения и наполненности датасета;
- прогноз деградации;
- предсказание на новых данных с проверкой схемы и новизны строк;
- артефакты модели: что сохранять, как загружать, как применять.

Основа переносится из [AutoDataAnalysis](https://github.com/forit-tech/AutoDataAnalysis) —
около 3 400 строк работающего и покрытого тестами ML-кода. Подробности —
в [DataArena/docs/MIGRATION_MAP.md](../DataArena/docs/MIGRATION_MAP.md), раздел 1.7.

## Документы

| Документ | О чём |
|---|---|
| [docs/DECISIONS.md](docs/DECISIONS.md) | Принятые решения (ADR log). Ссылки на них — по номеру `D-N` |
| [docs/DESIGN.md](docs/DESIGN.md) | Структура репозитория, доменная модель, интерфейсы, схемы, границы API, объём MVP, Stage 1 |
| [docs/MIGRATION.md](docs/MIGRATION.md) | Потабличная карта переноса из AutoDataAnalysis с вердиктом по каждому модулю |
| [docs/ARCHITECTURE_REVIEW.md](docs/ARCHITECTURE_REVIEW.md) | Обоснования решений и ML-риски |
| [docs/contracts/dataset-package-consumer.md](docs/contracts/dataset-package-consumer.md) | Как ModelArena читает Dataset Package |
| [docs/SCOPE.md](docs/SCOPE.md) | Границы сервиса, правило разделения с DataArena, план переноса |
| [DataArena/AUDIT.md](../DataArena/AUDIT.md) | Аудит исходного проекта |
| [Dataset Package v1](../DataArena/docs/contracts/dataset-package-v1.md) | Единственная точка соприкосновения с DataArena |

## Независимость

ModelArena работает без DataArena — package можно передать файлом.
DataArena работает без ModelArena. Обе стороны покрываются тестом.

## Лицензия

Будет определена до первого публичного релиза.
