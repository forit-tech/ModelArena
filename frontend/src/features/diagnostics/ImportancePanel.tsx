/**
 * Влияние признаков.
 *
 * Экран обязан удержаться от одной фразы: «признак X влияет на целевую переменную».
 * Измерено другое — насколько падает метрика, если значения признака перемешать.
 * Это утверждение о модели, а не о мире, и формулировка на экране должна это сохранять.
 *
 * Вычисление не мгновенное, поэтому запускается по кнопке: считать его при каждом
 * открытии диагностики значило бы тратить время пользователя без спроса.
 */
import { useState } from 'react'

import { getImportance } from '../../api/diagnostics-endpoints'
import type { Importance } from '../../api/diagnostics-types'
import { ApiFailure } from '../../api/client'
import { FailureNotice } from '../../components/FailureNotice'
import { Notice } from '../../components/Notice'

interface ImportancePanelProps {
  runId: string
  contenderKey: string
}

export function ImportancePanel({ runId, contenderKey }: ImportancePanelProps) {
  const [data, setData] = useState<Importance | null>(null)
  const [failure, setFailure] = useState<ApiFailure | null>(null)
  const [loading, setLoading] = useState(false)

  const measure = async () => {
    setLoading(true)
    setFailure(null)

    try {
      const response = await getImportance(runId, contenderKey, 5)
      setData(response.importance)
    } catch (error: unknown) {
      setFailure(
        error instanceof ApiFailure
          ? error
          : new ApiFailure('frontend_error', 'unexpected', String(error)),
      )
    } finally {
      setLoading(false)
    }
  }

  const scale = data?.rows?.length
    ? Math.max(...data.rows.map((row) => Math.abs(row.mean_drop)), 1e-9)
    : 1

  return (
    <section className="panel">
      <h2>Влияние признаков</h2>
      <p className="hint">
        Перестановочная важность: насколько падает метрика, если значения признака
        перемешать между строками. Модель при этом не переобучается — меняется только вход.
      </p>

      <div className="row">
        <button type="button" className="primary" disabled={loading} onClick={measure}>
          {loading ? 'Измеряем…' : 'Измерить влияние'}
        </button>
        {loading ? <span className="muted">модель предсказывает несколько раз подряд</span> : null}
      </div>

      {failure ? <FailureNotice failure={failure} onRetry={measure} /> : null}

      {data && !data.available ? (
        <Notice
          level="caution"
          title="Важность не измерена"
          why={data.reason ?? ''}
          action="Увеличьте долю holdout при следующем прогоне или возьмите больше данных."
        />
      ) : null}

      {data?.available ? (
        <>
          <p className="hint">
            Измерено на <strong>{data.measured_on}</strong> — части, которой сохранённая модель
            не видела ни разу. {data.n_rows} строк, {data.repeats} повторов,{' '}
            {data.metric_label}.
          </p>

          <div className="scroll">
            <table className="leaderboard">
              <thead>
                <tr>
                  <th>Признак</th>
                  <th>Падение метрики</th>
                  <th>Разброс</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {data.rows?.map((row) => (
                  <tr key={row.feature}>
                    <td>{row.feature}</td>
                    <td>{row.mean_drop.toFixed(4)}</td>
                    <td className="muted">± {row.std_drop.toFixed(4)}</td>
                    <td className="bar-cell">
                      <span
                        className={row.mean_drop >= 0 ? 'bar' : 'bar negative'}
                        style={{ width: `${Math.min(100, (Math.abs(row.mean_drop) / scale) * 100)}%` }}
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {data.correlated_groups?.length ? (
            <Notice
              level="caution"
              title="Есть сильно связанные признаки"
              why={
                'Важность между ними размазывается: модель может опереться на один и почти ' +
                'не тронуть второй. ' +
                data.correlated_groups
                  .map((item) => `${item.features.join(' ↔ ')} (${item.correlation})`)
                  .join('; ')
              }
              action="Нулевую важность у такого признака не читайте как «он не нужен»."
            />
          ) : null}

          <p className="hint">{data.interpretation}</p>
          {data.caveats?.map((note) => (
            <p key={note} className="hint">
              {note}
            </p>
          ))}
        </>
      ) : null}
    </section>
  )
}
