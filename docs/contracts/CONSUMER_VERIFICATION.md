# Независимая проверка контракта потребителем

Документ принадлежит **ModelArena** и в зеркалирование каталога не входит — как и
[dataset-package-consumer.md](dataset-package-consumer.md).

Дата: 2026-09-06. Версия контракта: `dataarena.package/1`, `1.0.0`.
Окружение: Python 3.12.10, **polars 1.43.0**, numpy 2.4.4
(спецификация замерена на polars 1.44.1 — проверка получилась ещё и межверсионной).

---

## 1. Что делалось

Алгоритм `dataarena.fingerprint/1` реализован **заново, по тексту спецификации** —
первой версией [tests/contracts/consumer_check.py](../../tests/contracts/consumer_check.py)
(файл с тех пор переписан под смену алгоритма, см. §7).
Эталонная `reference/fingerprint.py` до написания не открывалась и не импортируется.

Смысл: проверить утверждение спецификации «независимый потребитель может реализовать
алгоритм без импорта кода DataArena». Копирование эталона это утверждение не проверяет.

## 2. Результат

**Отпечаток golden package воспроизведён независимой реализацией на другой версии polars.**

```
объявлено:   6c1e69106eff37c118e110789aa51cc026804bf358940097a6a61adcbd40dae2
пересчитано: 6c1e69106eff37c118e110789aa51cc026804bf358940097a6a61adcbd40dae2
```

Поведение фикстур совпало с таблицей `README.md` там, где отпечаток применим:

| Фикстура | Ожидание | Независимая реализация |
|---|---|---|
| `customers_golden.dapkg` | ПРИНЯТ | отпечаток сошёлся |
| `negative/fingerprint-mismatch.dapkg` | ОТВЕРГНУТ | отпечаток не сошёлся — поймано |
| `negative/corrupted-part.dapkg` | ОТВЕРГНУТ по `sha256` | отпечаток сошёлся, что и ожидается: подмена в байтах файла ловится другим хешем |
| `negative/sampled-profile.dapkg` | ПРИНЯТ + warning | отпечаток сошёлся |
| `negative/minimal.dapkg` | ПРИНЯТ | **проверить не удалось** — см. F-1 |

Алгоритм по сути хороший: значение-ориентированная канонизация, переживает round-trip
через Parquet, не зависит от версии библиотеки. Замечания ниже — не к идее, а к тексту.

---

## 3. Находки

### F-1 — блокирующая. Спека и эталон берут порядок колонок и типы из разных мест

Нормативный текст §D: «Вход — таблица со столбцами **в порядке схемы**», а `logical_type`
подаётся как значение из словаря `schema.json`. Читается однозначно: источник — `schema.json`.

`reference/fingerprint.py` берёт **и порядок, и типы из Parquet**: `for name, dtype in
frame.schema.items()`, `logical_type(dtype)`. `schema.json` в расчёте не участвует вообще.

На golden package эти два правила совпадают, поэтому расхождение не видно. Но
`reference/validate_package.py` (строка 233) явно допускает случай, когда они расходятся:

> `schema.json` расходится с фактическими колонками parquet; выигрывает parquet

В этот момент две добросовестные реализации посчитают **разные отпечатки**, а правило
§D «мягкая реакция запрещена» превратит это в отказ у одной стороны и приём у другой.

**Что нужно:** в §D написать нормативно, что вход отпечатка — таблица, прочитанная из
`data/*.parquet` (порядок частей по `data.files`, порядок колонок — parquet), и что
`schema.json` **не является входом** отпечатка.

Именно на это правило я и напоролся: `negative/minimal.dapkg` объявляет `content_fingerprint`,
но не содержит `schema.json`, и реализация «по букве спеки» не может его проверить,
хотя фикстура обязана быть ПРИНЯТА.

### F-2 — блокирующая. Нет нормативного отображения physical dtype → `logical_type`

Если вход отпечатка — Parquet (см. F-1), то потребителю нужна таблица соответствия.
В спецификации её нет; в эталоне она есть в коде. Реализатор по документу вынужден угадывать.

**Что нужно:** таблицу в §D. Минимум: `Int8/16/32/64`, `UInt8/16/32/64` → `integer`;
`Float32/64` → `float`; `Boolean` → `boolean`; `Date` → `date`; `Datetime(любая единица,
любая tz)` → `datetime` (нормализация в UTC и микросекунды); `Time` → `time`;
`Duration` → `duration`; `Binary` → `binary`; `String` → `string`.

### F-3 — корректность. Молчаливый `return "string"` для неописанных типов

`logical_type()` заканчивается `return "string"`, то есть **любой** неописанный тип —
`Decimal`, `Categorical`, `Enum`, `List`, `Struct`, `Null` — попадает в строковую ветку,
где значения берутся `series.to_list()` и приводятся к тексту.

Для `Decimal` это означает хеширование текстового представления: `1.50` и `1.5` — одно
и то же число, но разные отпечатки. Для `List`/`Struct` — хеширование Python-репрезентации,
которая между версиями polars не гарантирована.

Decimal — не экзотика: он появляется при чтении Excel и CSV с денежными суммами,
то есть на типичном пути DataArena.

**Что нужно:** либо добавить `decimal` в словарь с каноническим кодированием
(беззнаковое целое + масштаб, оба в хеш), либо сделать неописанный dtype **явной ошибкой
сборки пакета** вместо тихого приведения к строке. Второе дешевле и честнее.

### F-4 — безопасность. Незнакомый `algorithm` отключает проверку идентичности

§D: `content_fingerprint.algorithm` незнаком → «читать данные можно, отпечаток не
проверяется, warning». Это путь понижения: пакет, объявивший `algorithm: "чтоугодно/9"`,
проходит **без** проверки содержимого, притом что несовпадение известного алгоритма —
жёсткий отказ.

**Поведение ModelArena** (решение потребителя, менять контракт не требуется):
незнакомый алгоритм — отказ по умолчанию. Если пользователь явно продавливает импорт,
снимок получает `fingerprint_verified: false`, и этот флаг виден на каждом эксперименте,
построенном на нём. Тихого приёма не будет.

### F-5 — процесс. «Проверено обеими сторонами» пока означает одну реализацию

`README.md`: «Проверено обеими сторонами: все семь результатов совпадают с таблицей».
У ModelArena на момент фиксации кода не было — значит проверка выполнена эталонным
валидатором, то есть одной реализацией, запущенной дважды.

Эта проверка — первая действительно независимая, и она сразу дала F-1 и F-2.
Формулировку в README стоит поправить, а критерий freeze
([D-15](../DECISIONS.md#d-15-критерий-freeze-контракта-и-разблокировки-stage-1))
понимать буквально: PASS засчитывается только от реализации, написанной по документу.

### F-6 — полнота фикстур. Пять негативных из десяти

Владелец перечислил десять негативных случаев. В `golden/negative/` шесть каталогов,
и один из них (`minimal`) ожидается ПРИНЯТЫМ, то есть негативных пять.

| Требуется | Есть |
|---|---|
| corrupted artifact SHA-256 | ✅ `corrupted-part` |
| corrupted logical fingerprint | ✅ `fingerprint-mismatch` |
| unsupported major version | ✅ `future-major` |
| newer compatible minor version | ✅ `future-minor` |
| sampled profile | ✅ `sampled-profile` |
| malformed schema | ❌ |
| missing required artifact | ❌ |
| duplicate / conflicting manifest entries | ❌ |
| archive traversal attempt | ❌ |
| decompression-limit violation | ❌ |

Две последние особенно важны: §H перечисляет для них тесты `test_zip_bomb_rejected`
и `test_path_traversal_rejected`, но самих фикстур в каталоге нет, и потребителю
нечем их прогнать у себя.

---

## 4. Полный прогон consumer-проверки — PASS

После разбора F-1 проверка доведена до полной:
[tests/contracts/consumer_check.py](../../tests/contracts/consumer_check.py) —
версия формата, целостность частей, отпечаток содержимого, флаг выборочного профиля.
Эталонные `reference/*.py` не импортируются.

```
[OK ] golden/customers_golden.dapkg                ПРИНЯТ
[OK ] golden/negative/minimal.dapkg                ПРИНЯТ
[OK ] golden/negative/future-minor.dapkg           ПРИНЯТ | warning: более новая minor-версия 1.7.0
[OK ] golden/negative/sampled-profile.dapkg        ПРИНЯТ | warning: профиль по выборке
[OK ] golden/negative/future-major.dapkg           ОТВЕРГНУТ package_version_unsupported
[OK ] golden/negative/corrupted-part.dapkg         ОТВЕРГНУТ package_integrity_failed
[OK ] golden/negative/fingerprint-mismatch.dapkg   ОТВЕРГНУТ package_fingerprint_mismatch

ВСЕ СЕМЬ СОВПАЛИ
```

**Критерий freeze [D-15](../DECISIONS.md#d-15-критерий-freeze-контракта-и-разблокировки-stage-1)
выполнен:** producer-тесты владельца и consumer-тесты ModelArena дают совпадающие результаты
на одном golden package, причём consumer-реализация написана по документу и работает
на другой версии polars.

## 5. Состояние находок

| # | Статус | Что дальше |
|---|---|---|
| F-1 | **разрешена в поведении, дефект текста остался** | Выбрана трактовка эталона: вход отпечатка — таблица из `data/*.parquet`, `schema.json` входом не является. Выбор зафиксирован комментарием в `consumer_check.py`. Просьба к владельцу — одна нормативная фраза в §D, иначе следующий реализатор повторит мою ошибку |
| F-2 | **обойдена, дефект текста остался** | Таблица `physical dtype → logical_type` выведена самостоятельно и совпала. Пока её нет в §D, это остаётся источником расхождений |
| F-3 | **открыта** | `Decimal`, `Categorical`, `List`, `Struct` в эталоне молча уходят в `string`. У себя сделала обратное: неописанный dtype — явная ошибка. Не блокирует Stage 1, потому что в golden таких колонок нет, но всплывёт на первом же денежном датасете из Excel |
| F-4 | **закрыта у потребителя** | Незнакомый `algorithm` — отказ `package_fingerprint_algorithm_unsupported`, строже спецификации. Идентичность, которую нельзя проверить, не должна выглядеть проверенной. На фикстурах расхождений не даёт |
| F-5 | **снята** | Теперь проверок действительно две, и они независимы |
| F-6 | **открыта** | Пяти негативных фикстур нет. Отсутствующие `archive traversal` и `decompression-limit` собираются на стороне ModelArena локально при реализации reader (Stage 1) — ждать владельца для них не нужно |

## 6. Вывод

Контракт принимается. Ни одна из оставшихся находок не блокирует Stage 1:
F-1 и F-2 — дефекты формулировок при совпадающем поведении, F-3 и F-6 закрываются
на нашей стороне и в следующей ревизии спецификации.

Правило `-text` для каталога контракта в `.gitattributes` не трогать: без него checkout
на другой платформе конвертирует переводы строк и `sha256` в манифесте станут невалидными.
Проверено — правило на месте.

---

## 7. Повторный прогон на release candidate — 13 из 13

Владелец закрыл все шесть находок, перенёс каталог в `contracts/dataset-package/v1/`,
добавил пять недостающих фикстур и **изменил алгоритм по существу**:
`dataarena.fingerprint/1` → `dataarena-logical-sha256-v1` (число колонок в заголовке,
Unicode NFC, `decimal`, нормализация datetime). Прежний отпечаток golden недействителен.

Независимая реализация переписана под новый §D — [consumer_check.py](../../tests/contracts/consumer_check.py),
`reference/*.py` по-прежнему не импортируются. Добавлены проверки §H целиком
и лимиты архива.

```
[OK ] golden/customers_golden.dapkg                  ПРИНЯТ
[OK ] golden/customers_golden.dapkg.zip              ПРИНЯТ
[OK ] golden/negative/minimal.dapkg                  ПРИНЯТ
[OK ] golden/negative/future-minor.dapkg             ПРИНЯТ | более новая minor 1.7.0
[OK ] golden/negative/sampled-profile.dapkg          ПРИНЯТ | профиль по выборке
[OK ] golden/negative/future-major.dapkg             ОТВЕРГНУТ package_version_unsupported
[OK ] golden/negative/corrupted-part.dapkg           ОТВЕРГНУТ package_integrity_failed
[OK ] golden/negative/fingerprint-mismatch.dapkg     ОТВЕРГНУТ package_fingerprint_mismatch
[OK ] golden/negative/malformed-schema.dapkg         ОТВЕРГНУТ package_part_invalid
[OK ] golden/negative/missing-artifact.dapkg         ОТВЕРГНУТ package_part_missing
[OK ] golden/negative/duplicate-entry.dapkg          ОТВЕРГНУТ package_duplicate_entry
[OK ] golden/negative/traversal.dapkg.zip            ОТВЕРГНУТ package_unsafe_path
[OK ] golden/negative/zip-bomb.dapkg.zip             ОТВЕРГНУТ package_limit_exceeded

СОВПАЛО ВСЁ (13)
```

Отпечаток golden воспроизведён независимо: `457184a0a0dc0377…`, polars 1.43.0.

### 7.1 Статус находок

| # | Статус |
|---|---|
| F-1 | **закрыта в поведении**, но остаётся расхождение с решением владельца — см. 7.2 |
| F-2 | **закрыта**: нормативная таблица типов в §D, реализовано по ней |
| F-3 | **закрыта**: неописанный тип — ошибка сборки; `decimal` получил каноническое кодирование с нормализацией масштаба |
| F-4 | **закрыта в контракте**: незнакомый алгоритм — отказ, приём только явным действием с `fingerprint_verified: false` |
| F-5 | **закрыта**: формулировка убрана из README |
| F-6 | **закрыта**: тринадцать фикстур, включая traversal и zip-bomb |

### 7.2 Открыто: расхождение с решением владельца по F-1

Решение владельца от 2026-09-06:

> если `schema.json` присутствует — он не может расходиться с physical schema;
> любое расхождение = **invalid package**; формулировку «Parquet wins» убрать.

В release candidate это правило **не отражено**:

| Место | Что написано |
|---|---|
| `dataset-package-v1.md:510` (§H) | `schema.json расходится с фактическим dtype` → чтение; выигрывает Parquet, warning |
| `dataset-package-v1.md:715` | «в Parquet **выигрывает Parquet**, расхождение попадает в warnings» |
| `reference/validate_package.py:303` | `report.warn("schema.json расходится … выигрывает parquet")` |

Спор не про вход отпечатка: выбор «вход — только Parquet» правильный и снимает
недетерминированность. Спор про то, **валиден ли пакет**, в котором `schema.json`
описывает не те данные, что лежат в Parquet. Владелец решил, что нет.

Эти два решения совместимы и вместе дают лучший результат: вход отпечатка — Parquet,
а расхождение схемы с данными делает пакет невалидным, поэтому в валидном пакете
источники физически не могут разойтись.

### 7.3 Мелкие расхождения текста

Найдены при реализации, каждое — тот же класс, что и исходная F-1:

1. `dataset-package-v1.md:260` — псевдокод §D по-прежнему говорит «для каждой колонки,
   **в порядке схемы**», тогда как нормативный абзац выше требует порядок из Parquet.
   Реализатор, читающий только псевдокод, повторит мою первую ошибку;
2. `dataset-package-v1.md:270` — перечень «`logical_type` — одно из: boolean, integer,
   float, date, datetime, duration, time, string, binary» **не содержит `decimal`**,
   хотя таблица отображения и таблица кодирования его содержат.

---

## 8. Финальный прогон — 14 из 14, все условия закрыты

Владелец внёс правку по F-1 в редакции владельца проекта: расхождение `schema.json`
с физической схемой Parquet теперь **отказ** `package_schema_mismatch`, а не «выигрывает
Parquet с предупреждением». Добавлена четырнадцатая фикстура.

Независимая реализация дополнена проверкой согласованности схемы — **по тексту решения,
до появления фикстуры в каталоге**. Когда фикстура появилась, она была опознана
и отвергнута с тем же кодом, без правок под неё.

```
[OK ] golden/customers_golden.dapkg                  ПРИНЯТ
[OK ] golden/customers_golden.dapkg.zip              ПРИНЯТ
[OK ] golden/negative/minimal.dapkg                  ПРИНЯТ
[OK ] golden/negative/future-minor.dapkg             ПРИНЯТ | более новая minor 1.7.0
[OK ] golden/negative/sampled-profile.dapkg          ПРИНЯТ | профиль по выборке
[OK ] golden/negative/future-major.dapkg             ОТВЕРГНУТ package_version_unsupported
[OK ] golden/negative/corrupted-part.dapkg           ОТВЕРГНУТ package_integrity_failed
[OK ] golden/negative/fingerprint-mismatch.dapkg     ОТВЕРГНУТ package_fingerprint_mismatch
[OK ] golden/negative/malformed-schema.dapkg         ОТВЕРГНУТ package_part_invalid
[OK ] golden/negative/missing-artifact.dapkg         ОТВЕРГНУТ package_part_missing
[OK ] golden/negative/duplicate-entry.dapkg          ОТВЕРГНУТ package_duplicate_entry
[OK ] golden/negative/traversal.dapkg.zip            ОТВЕРГНУТ package_unsafe_path
[OK ] golden/negative/zip-bomb.dapkg.zip             ОТВЕРГНУТ package_limit_exceeded
[OK ] golden/negative/schema-mismatch.dapkg          ОТВЕРГНУТ package_schema_mismatch

СОВПАЛО ВСЁ (14)
```

Проверка не молчит о новом: фикстура, появившаяся в каталоге и не описанная в матрице
ожиданий, показывается как FAIL, а не игнорируется.

### 8.1 Текстовые огрехи закрыты

- псевдокод §D больше не говорит «в порядке схемы»; формулировка осталась только
  в журнале решений как описание того, что было исправлено;
- `decimal` присутствует в перечне `logical_type`;
- «выигрывает Parquet» в нормативном тексте и в валидаторе отсутствует.

### 8.2 Условие «оба репозитория указывают на одну версию» — проверено

Сверка каталога `contracts/dataset-package/v1` с каталогом владельца по SHA-256 каждого файла:

```
файлов у меня 91, у владельца 91
только у меня: нет      только у владельца: нет      различаются: нет
ЗЕРКАЛО ПОБАЙТОВО ИДЕНТИЧНО
```

### 8.3 Состояние по финальному критерию

| # | Условие | Статус |
|---|---|---|
| 1 | producer/reference-набор DataArena PASS | на стороне владельца |
| 2 | независимая consumer-проверка ModelArena PASS | ✅ 14/14 |
| 3 | полная матрица фикстур, включая schema mismatch | ✅ |
| 4 | оба репозитория на одной версии спецификации | ✅ побайтово |
| 5 | одновременный тег `contract/dataset-package-v1.0.0` | ждёт подтверждения владельца |

Со стороны ModelArena к freeze препятствий нет.

### 8.4 Перекрёстный прогон: две реализации, одинаковые вердикты

По предложению владельца контракта проверки запущены крест-накрест на одном каталоге,
вместо обмена отчётами о состояниях, которые успевают устареть.

Эталонный `reference/validate_package.py` и независимый `consumer_check.py` дали
**совпадающие вердикты и совпадающие коды ошибок на всех четырнадцати фикстурах**:

```
ПРИНЯТ     customers_golden (каталог и zip), minimal,
           future-minor (+warning), sampled-profile (+warning)
ОТВЕРГНУТ  future-major        package_version_unsupported
           corrupted-part      package_integrity_failed
           fingerprint-mismatch package_fingerprint_mismatch
           malformed-schema    package_part_invalid
           missing-artifact    package_part_missing
           duplicate-entry     package_duplicate_entry
           schema-mismatch     package_schema_mismatch
           traversal.zip       package_unsafe_path
           zip-bomb.zip        package_limit_exceeded
```

Дополнено сравнение `position`: сверяются имя, порядок, `position` и `logical_type`.
`physical_type` информативен и не сверяется — `Int32` и `Int64` дают один `logical_type`,
а ширина целого в отпечаток не входит.

Это сильнее, чем «обе стороны прогнали у себя»: реализации написаны независимо,
одна по документу, и сходятся не только в вердиктах, но и в кодах.
