/**
 * Разбор ошибок: переход от «модель ошибается» к конкретным наблюдениям.
 *
 * Значения признаков и остальные колонки снимка показаны раздельно. Подписать целевую
 * колонку словом «признак» на экране разбора ошибок значило бы показать как вход то,
 * чего у модели на входе не было.
 */
import { useState } from 'react'

import { getErrorRows } from '../../api/diagnostics-endpoints'
import { FailureNotice } from '../../components/FailureNotice'
import { useRequest } from '../../hooks/useRequest'

const CLASSIFICATION_KINDS = [
  { key: 'all_mistakes', label: 'все ошибки' },
  { key: 'false_negative', label: 'пропуски' },
  { key: 'false_positive', label: 'ложные срабатывания' },
  { key: 'confident_mistakes', label: 'уверенные ошибки' },
]

const REGRESSION_KINDS = [{ key: 'largest_error', label: 'крупнейшие ошибки' }]

interface ErrorRowsPanelProps {
  runId: string
  contenderKey: string
  taskType: string
}

export function ErrorRowsPanel({ runId, contenderKey, taskType }: ErrorRowsPanelProps) {
  const kinds = taskType === 'regression' ? REGRESSION_KINDS : CLASSIFICATION_KINDS
  const [kind, setKind] = useState(kinds[0]?.key ?? 'all_mistakes')
  const errors = useRequest(
    () => getErrorRows(runId, contenderKey, kind, 25),
    [runId, contenderKey, kind],
  )
  const data = errors.data?.errors
  const first = data?.rows[0]
  const columns = first ? Object.keys(first.features) : []

  return (
    <section className="panel">
      <h2>Строки, на которых модель ошиблась</h2>

      <div className="row">
        <label>
          Показать{' '}
          <select
            aria-label="Вид ошибок"
            value={kind}
            onChange={(event) => setKind(event.target.value)}
          >
            {kinds.map((item) => (
              <option key={item.key} value={item.key}>
                {item.label}
              </option>
            ))}
          </select>
        </label>
        {errors.loading ? <span className="muted">Считаем…</span> : null}
      </div>

      {errors.failure ? <FailureNotice failure={errors.failure} onRetry={errors.reload} /> : null}

      {data ? (
        <>
          <p className="hint">
            {data.note} Подходящих строк: {data.total_matching}, показано {data.shown}.
          </p>

          {data.rows.length ? (
            <div className="scroll">
              <table className="leaderboard">
                <thead>
                  <tr>
                    <th>Строка</th>
                    <th>Факт</th>
                    <th>Предсказано</th>
                    {columns.map((name) => (
                      <th key={name}>{name}</th>
                    ))}
                    {first?.probabilities ? <th>Уверенность</th> : null}
                  </tr>
                </thead>
                <tbody>
                  {data.rows.map((row) => (
                    <tr key={row.row_id}>
                      <td className="muted">{row.row_id}</td>
                      <td>{String(row.y_true)}</td>
                      <td className="indistinguishable">{String(row.y_pred)}</td>
                      {columns.map((name) => (
                        <td key={name}>{format(row.features[name])}</td>
                      ))}
                      {row.probabilities ? (
                        <td className="muted">
                          {Math.max(...Object.values(row.probabilities)).toFixed(3)}
                        </td>
                      ) : null}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="hint">Таких ошибок нет.</p>
          )}

          <p className="hint">
            Показаны только те колонки, которые модель действительно видела. Целевая колонка
            и исключённые из признаков идут отдельно и в таблицу входа не попадают.
          </p>
        </>
      ) : null}
    </section>
  )
}

function format(value: unknown): string {
  if (value === null || value === undefined) {
    //пропуск — это не пустая строка: он должен быть виден как пропуск
    return '—'
  }

  if (typeof value === 'number') {
    return Number.isInteger(value) ? String(value) : value.toFixed(4)
  }

  return String(value)
}
