/**
 * Применение сохранённой модели к новым данным.
 *
 * Здесь заканчивается путь от датасета к результату: модель, выбранная и разобранная
 * на предыдущих экранах, применяется к строкам, которых она не видела.
 *
 * Экран обязан показывать карточку артефакта до предсказания. Модель, про которую
 * неизвестно, каких колонок она ждёт и что означает её выход, применять вслепую нельзя —
 * именно так получаются правдоподобные неверные ответы.
 */
import { useState } from 'react'

import { ApiFailure } from '../../api/client'
import { getModelCard, predictBatch, predictOne } from '../../api/diagnostics-endpoints'
import type { PredictionResult } from '../../api/diagnostics-types'
import { FailureNotice } from '../../components/FailureNotice'
import { Notice } from '../../components/Notice'
import { useRequest } from '../../hooks/useRequest'

interface TestDriveViewProps {
  runId: string | null
  contenderKey: string | null
}

export function TestDriveView({ runId, contenderKey }: TestDriveViewProps) {
  const card = useRequest(
    runId && contenderKey ? () => getModelCard(runId, contenderKey) : null,
    [runId, contenderKey],
  )
  const [values, setValues] = useState<Record<string, string>>({})
  const [single, setSingle] = useState<PredictionResult | null>(null)
  const [batch, setBatch] = useState<{ result: PredictionResult; csv: string } | null>(null)
  const [failure, setFailure] = useState<ApiFailure | null>(null)
  const [busy, setBusy] = useState(false)

  if (!runId || !contenderKey) {
    return (
      <section className="panel">
        <h2>Применение модели</h2>
        <p className="hint">
          Сначала завершите прогон и выберите участника в разделе «Прогон» — применять пока
          нечего.
        </p>
      </section>
    )
  }

  const manifest = card.data?.manifest
  const schema = manifest?.feature_schema

  const fail = (error: unknown) =>
    setFailure(
      error instanceof ApiFailure
        ? error
        : new ApiFailure('frontend_error', 'unexpected', String(error)),
    )

  const onSingle = async () => {
    setBusy(true)
    setFailure(null)
    setSingle(null)

    try {
      const parsed = Object.fromEntries(
        Object.entries(values).map(([key, raw]) => [key, coerce(raw, schema?.numeric ?? [], key)]),
      )
      setSingle((await predictOne(runId, contenderKey, parsed)).result)
    } catch (error: unknown) {
      fail(error)
    } finally {
      setBusy(false)
    }
  }

  const onBatch = async (file: File) => {
    setBusy(true)
    setFailure(null)
    setBatch(null)

    try {
      setBatch(await predictBatch(runId, contenderKey, file))
    } catch (error: unknown) {
      fail(error)
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <section className="panel">
        <h2>Сохранённая модель</h2>

        {card.failure ? <FailureNotice failure={card.failure} onRetry={card.reload} /> : null}

        {manifest ? (
          <>
            <dl className="facts">
              <div>
                <dt>Модель</dt>
                <dd>{manifest.label}</dd>
              </div>
              <div>
                <dt>Обучена на</dt>
                <dd>{manifest.training_rows} строках</dd>
              </div>
              <div>
                <dt>Задача</dt>
                <dd>
                  {manifest.task.task_type} → {manifest.task.target_column}
                </dd>
              </div>
              <div>
                <dt>Контрольная сумма</dt>
                <dd>
                  <code>{manifest.model_sha256.slice(0, 12)}</code>
                </dd>
              </div>
            </dl>

            <p className="hint">
              Ждёт колонки: <strong>{schema?.columns.join(', ')}</strong>. Порядок в файле
              может быть любым — он восстанавливается по манифесту.
            </p>
            <p className="hint">{card.data?.trust_boundary}</p>

            {card.data?.environment_warnings.map((note) => (
              <Notice
                key={note}
                level="caution"
                title="Версии библиотек отличаются"
                why={note}
                action="Если предсказания выглядят странно, переобучите модель в текущем окружении."
              />
            ))}
          </>
        ) : null}
      </section>

      {schema ? (
        <section className="panel">
          <h2>Одна строка</h2>
          <p className="hint">
            Введите значения признаков. Препроцессинг берётся из артефакта и заново
            не обучается — иначе предсказание перестало бы соответствовать модели.
          </p>

          <div className="form-grid">
            {schema.columns.map((name) => (
              <label key={name}>
                {name}
                <span className="muted"> ({schema.dtypes[name] ?? 'значение'})</span>
                <input
                  value={values[name] ?? ''}
                  onChange={(event) =>
                    setValues((current) => ({ ...current, [name]: event.target.value }))
                  }
                />
              </label>
            ))}
          </div>

          <div className="row">
            <button type="button" className="primary" disabled={busy} onClick={onSingle}>
              {busy ? 'Считаем…' : 'Предсказать'}
            </button>
          </div>

          {single ? <PredictionBlock result={single} /> : null}
        </section>
      ) : null}

      <section className="panel">
        <h2>Файл</h2>
        <p className="hint">
          csv, tsv, parquet, json или xlsx. Схема проверяется до предсказания: недостающая
          колонка — отказ, лишняя — предупреждение.
        </p>

        <div className="row">
          <input
            type="file"
            aria-label="Файл для предсказания"
            disabled={busy}
            onChange={(event) => {
              const file = event.target.files?.[0]
              if (file) {
                void onBatch(file)
              }
            }}
          />
        </div>

        {failure ? <FailureNotice failure={failure} onRetry={() => setFailure(null)} /> : null}

        {batch ? (
          <>
            <PredictionBlock result={batch.result} />
            <div className="row">
              <button
                type="button"
                onClick={() => download(batch.csv, `predictions-${contenderKey}.csv`)}
              >
                Скачать предсказания
              </button>
            </div>
          </>
        ) : null}
      </section>
    </>
  )
}

function PredictionBlock({ result }: { result: PredictionResult }) {
  return (
    <>
      {result.schema.notes.map((note) => (
        <p key={note} className="hint">
          {note}
        </p>
      ))}

      <div className="scroll">
        <table className="leaderboard">
          <thead>
            <tr>
              <th>#</th>
              <th>Предсказание</th>
              {result.has_probabilities
                ? result.class_labels.map((label) => <th key={label}>{label}</th>)
                : null}
            </tr>
          </thead>
          <tbody>
            {result.predictions.slice(0, 50).map((row) => (
              <tr key={row.index}>
                <td className="muted">{row.index}</td>
                <td>
                  {typeof row.prediction === 'number'
                    ? row.prediction.toFixed(4)
                    : row.prediction}
                </td>
                {result.has_probabilities
                  ? result.class_labels.map((label) => (
                      <td key={label} className="muted">
                        {row.probabilities?.[label]?.toFixed(4) ?? '—'}
                      </td>
                    ))
                  : null}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {result.n_rows > 50 ? (
        <p className="hint">Показаны первые 50 строк из {result.n_rows}. Выгрузка содержит все.</p>
      ) : null}
    </>
  )
}

function coerce(raw: string, numeric: string[], name: string): unknown {
  if (raw === '') {
    //пустое поле — это пропуск, а не пустая строка: модель обучалась с пропусками
    return null
  }

  if (numeric.includes(name)) {
    const parsed = Number(raw)
    return Number.isNaN(parsed) ? raw : parsed
  }

  return raw
}

function download(content: string, filename: string): void {
  const blob = new Blob([content], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}
