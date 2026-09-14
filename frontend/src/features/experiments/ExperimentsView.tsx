/**
 * История экспериментов и их сравнение.
 *
 * Сравнение начинается не с метрик, а с перечня того, чем отличаются условия. Две метрики
 * рядом без упоминания разного разбиения — самый удобный способ обмануть себя, и экран
 * устроен так, чтобы этого не позволить: несопоставимые прогоны помечены до того,
 * как пользователь увидит их числа.
 */
import { useState } from 'react'

import { ApiFailure } from '../../api/client'
import { compareExperiments, listExperiments } from '../../api/diagnostics-endpoints'
import type { ExperimentCard, ExperimentComparison } from '../../api/diagnostics-types'
import { FailureNotice } from '../../components/FailureNotice'
import { Notice } from '../../components/Notice'
import { useRequest } from '../../hooks/useRequest'

interface ExperimentsViewProps {
  onOpenRun: (runId: string) => void
}

export function ExperimentsView({ onOpenRun }: ExperimentsViewProps) {
  const history = useRequest(() => listExperiments(), [])
  const [selected, setSelected] = useState<string[]>([])
  const [comparison, setComparison] = useState<ExperimentComparison | null>(null)
  const [failure, setFailure] = useState<ApiFailure | null>(null)

  const toggle = (runId: string) => {
    setComparison(null)
    setSelected((current) =>
      current.includes(runId) ? current.filter((item) => item !== runId) : [...current, runId],
    )
  }

  const run = async () => {
    setFailure(null)

    try {
      setComparison(await compareExperiments(selected))
    } catch (error: unknown) {
      setFailure(
        error instanceof ApiFailure
          ? error
          : new ApiFailure('frontend_error', 'unexpected', String(error)),
      )
    }
  }

  const experiments = history.data?.experiments ?? []

  return (
    <>
      <section className="panel">
        <h2>Эксперименты</h2>
        <p className="hint">
          Каждая строка — прогон вместе с условиями, при которых он выполнялся. Внутри одного
          прогона участники сравнимы по построению; между прогонами — только настолько,
          насколько совпадают условия.
        </p>

        {history.failure ? (
          <FailureNotice failure={history.failure} onRetry={history.reload} />
        ) : null}
        {history.loading && !experiments.length ? <p className="hint">Загружаем…</p> : null}

        {experiments.length ? (
          <>
            <div className="row">
              <button
                type="button"
                className="primary"
                disabled={selected.length < 2}
                onClick={run}
              >
                Сравнить выбранные ({selected.length})
              </button>
              {selected.length === 1 ? (
                <span className="muted">выберите ещё один прогон</span>
              ) : null}
            </div>

            {failure ? <FailureNotice failure={failure} onRetry={run} /> : null}

            <div className="scroll">
              <table className="leaderboard">
                <thead>
                  <tr>
                    <th />
                    <th>Когда</th>
                    <th>Цель</th>
                    <th>Состояние</th>
                    <th>Чемпион</th>
                    <th>Участники</th>
                    <th>Протокол</th>
                    <th>Модели</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {experiments.map((item) => (
                    <tr key={item.run_id}>
                      <td>
                        <input
                          type="checkbox"
                          aria-label={`Выбрать ${item.run_id}`}
                          checked={selected.includes(item.run_id)}
                          onChange={() => toggle(item.run_id)}
                        />
                      </td>
                      <td className="muted">{item.created_at.slice(0, 19).replace('T', ' ')}</td>
                      <td>
                        {item.task.target_column}
                        <span className="muted"> · {item.task.task_type}</span>
                      </td>
                      <td>{item.state}</td>
                      <td>
                        {item.champion?.contender_key ? (
                          item.champion.label
                        ) : (
                          //«не выбран» — это вывод прогона, а не пустая ячейка
                          <span className="muted">не выбран</span>
                        )}
                      </td>
                      <td className="muted">
                        {item.contenders.succeeded}/{item.contenders.planned}
                        {item.contenders.failed ? ` · ${item.contenders.failed} отказ` : ''}
                      </td>
                      <td className="muted">
                        {item.protocol.cv_splitter} · {item.protocol.n_splits} фолдов · seed{' '}
                        {item.protocol.seed}
                      </td>
                      <td className="muted">{item.artifacts.length}</td>
                      <td>
                        <button type="button" onClick={() => onOpenRun(item.run_id)}>
                          Открыть
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <p className="hint">Прогонов пока нет.</p>
        )}
      </section>

      {comparison ? <ComparisonPanel comparison={comparison} /> : null}
    </>
  )
}

function ComparisonPanel({ comparison }: { comparison: ExperimentComparison }) {
  return (
    <section className="panel">
      <h2>Сравнение</h2>

      <Notice
        level={comparison.comparable ? 'info' : 'caution'}
        title={comparison.comparable ? 'Условия совпадают' : 'Условия отличаются'}
        why={comparison.verdict}
        action={
          comparison.comparable
            ? 'Метрики этих прогонов можно сравнивать напрямую.'
            : 'Смотрите на список различий ниже, прежде чем сравнивать числа.'
        }
      />

      {comparison.differences.length ? (
        <>
          <h3 className="subhead">Чем отличаются условия</h3>
          <ul className="contenders">
            {comparison.differences.map((difference) => (
              <li key={difference.key} className="contender cancelled">
                <div className="contender-head">
                  <strong>{difference.label}</strong>
                </div>
                <p className="contender-reason">{difference.consequence}</p>
                <table className="metrics">
                  <tbody>
                    {difference.values.map((entry) => (
                      <tr key={entry.run_id}>
                        <th scope="row">{entry.run_id.slice(0, 16)}</th>
                        <td>{describe(entry.value)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </li>
            ))}
          </ul>
        </>
      ) : null}

      <h3 className="subhead">Результаты</h3>
      <div className="scroll">
        <table className="leaderboard">
          <thead>
            <tr>
              <th>Прогон</th>
              <th>Чемпион</th>
              <th>Результат</th>
              <th>Цель</th>
              <th>Время</th>
            </tr>
          </thead>
          <tbody>
            {comparison.runs.map((item) => (
              <tr key={item.run_id}>
                <td className="muted">{item.run_id.slice(0, 16)}</td>
                <td>{item.champion?.label || <span className="muted">не выбран</span>}</td>
                <td>
                  {item.champion?.score !== null && item.champion?.score !== undefined
                    ? item.champion.score.toFixed(4)
                    : '—'}
                </td>
                <td className="muted">{item.champion?.objective ?? '—'}</td>
                <td className="muted">
                  {item.runtime_seconds === null ? '—' : `${item.runtime_seconds} с`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

function describe(value: unknown): string {
  if (Array.isArray(value)) {
    return value.join(', ')
  }

  return String(value)
}

export type { ExperimentCard }
