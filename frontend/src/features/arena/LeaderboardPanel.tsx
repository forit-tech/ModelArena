/**
 * Таблица результатов и вердикт о чемпионе.
 *
 * Экран отвечает не на вопрос «у кого число больше», а на вопрос «почему эта модель
 * выше той». Поэтому:
 *
 * - **«чемпион не выбран» — полноценный результат**, а не пустое место. Если превосходство
 *   лидера над точкой отсчёта не держится на фолдах, победитель не объявляется, и причина
 *   написана словами;
 * - **соседи по таблице, между которыми разница тонет в разбросе, помечены как
 *   неразличимые.** Порядок между ними определён объявленным заранее правилом, и правило
 *   названо. Без этой пометки первое место читается как превосходство, которого нет;
 * - **holdout показан отдельно** и подписан как не участвовавший в выборе.
 */
import { useState } from 'react'

import { getLeaderboard } from '../../api/endpoints'
import type { LeaderboardRow, PairedComparison } from '../../api/types'
import { FailureNotice } from '../../components/FailureNotice'
import { Notice } from '../../components/Notice'
import { useRequest } from '../../hooks/useRequest'

interface LeaderboardPanelProps {
  runId: string | null
  /** прогон ещё идёт — таблица неполна, и об этом надо предупредить */
  active: boolean
}

export function LeaderboardPanel({ runId, active }: LeaderboardPanelProps) {
  const [metric, setMetric] = useState<string>('')
  const board = useRequest(
    runId ? () => getLeaderboard(runId, metric || undefined) : null,
    [runId, metric, active],
  )

  if (!runId) {
    return null
  }

  const data = board.data?.leaderboard
  const ranked = data?.rows.filter((row) => row.rank !== null) ?? []
  const excluded = data?.rows.filter((row) => row.rank === null) ?? []

  return (
    <section className="panel">
      <h2>Таблица результатов</h2>

      {data ? (
        <p className="hint">
          Цель сравнения: <strong>{data.objective.description}</strong> Таблица пересчитывается
          из сохранённых предсказаний, поэтому смена цели не требует переобучения.
        </p>
      ) : null}

      <div className="row">
        <label>
          Ранжировать по{' '}
          <select
            aria-label="Метрика ранжирования"
            value={metric}
            onChange={(event) => setMetric(event.target.value)}
          >
            <option value="">цели по умолчанию</option>
            {(data?.rows[0]?.cross_validated ?? []).map((item) => (
              <option key={item.key} value={item.key}>
                {item.label}
              </option>
            ))}
          </select>
        </label>
        {board.loading ? <span className="muted">Считаем…</span> : null}
      </div>

      {board.failure ? <FailureNotice failure={board.failure} onRetry={board.reload} /> : null}

      {active ? (
        <Notice
          level="caution"
          title="Прогон ещё идёт"
          why="Часть участников не завершилась, поэтому таблица неполна и порядок может поменяться."
          action="Дождитесь окончания прогона, прежде чем делать выводы о победителе."
        />
      ) : null}

      {data ? <ChampionVerdict verdict={data.champion} /> : null}

      {ranked.length ? (
        <div className="scroll">
          <table className="leaderboard">
            <thead>
              <tr>
                <th>#</th>
                <th>Участник</th>
                <th>{metricLabel(data, metric)}</th>
                <th>Против следующего</th>
                <th>Против точки отсчёта</th>
              </tr>
            </thead>
            <tbody>
              {ranked.map((row) => (
                <tr key={row.contender_key} className={row.is_baseline ? 'baseline-row' : undefined}>
                  <td>{row.rank}</td>
                  <td>
                    {row.label}
                    {row.is_baseline ? (
                      <>
                        {' '}
                        <span className="tag">точка отсчёта</span>
                      </>
                    ) : null}
                  </td>
                  <td>{formatScore(row)}</td>
                  <td>
                    <Verdict comparison={row.versus_next} />
                  </td>
                  <td>
                    <Verdict comparison={row.versus_baseline} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}

      {excluded.length ? (
        <>
          <h3 className="subhead">Вне ранжирования</h3>
          <p className="hint">
            Эти участники не исчезли из отчёта: у каждого названа причина. Пустое место
            читалось бы как проигрыш.
          </p>
          <ul className="contenders">
            {excluded.map((row) => (
              <li key={row.contender_key} className="contender skipped">
                <div className="contender-head">
                  <strong>{row.label}</strong>
                  {row.score !== null ? (
                    <span className="muted">результат есть: {row.score.toFixed(4)}</span>
                  ) : null}
                </div>
                <p className="contender-reason">{row.reason}</p>
              </li>
            ))}
          </ul>
        </>
      ) : null}

      {data?.notes.map((note) => (
        <p key={note} className="hint">
          {note}
        </p>
      ))}
    </section>
  )
}

function ChampionVerdict({ verdict }: { verdict: LeaderboardRowVerdict }) {
  //отсутствие чемпиона — это вывод, а не пустота: он заслуживает такой же заметности
  const chosen = verdict.contender_key !== null

  return (
    <Notice
      level={chosen ? 'info' : 'caution'}
      title={chosen ? `Чемпион: ${verdict.label}` : 'Чемпион не выбран'}
      why={verdict.reason}
      action={
        chosen
          ? 'Прежде чем брать модель в работу, посмотрите на соседей по таблице: неразличимые с ней отмечены.'
          : 'Это результат прогона, а не ошибка. Смотрите на причину: возможно, признаки не дают сигнала или разница тонет в разбросе.'
      }
    />
  )
}

type LeaderboardRowVerdict = {
  contender_key: string | null
  label: string
  reason: string
}

function Verdict({ comparison }: { comparison: PairedComparison | null }) {
  if (!comparison) {
    return <span className="muted">—</span>
  }

  return (
    <span className={comparison.stable ? 'stable' : 'indistinguishable'} title={comparison.explanation}>
      {comparison.stable ? 'устойчиво выше' : 'неразличимы'}
      <span className="muted">
        {' '}
        {comparison.mean_difference >= 0 ? '+' : ''}
        {comparison.mean_difference.toFixed(4)} · {comparison.folds_won}/{comparison.folds_compared}
      </span>
    </span>
  )
}

function formatScore(row: LeaderboardRow): string {
  if (row.score === null) {
    return 'не посчитана'
  }

  return row.std === null ? row.score.toFixed(4) : `${row.score.toFixed(4)} ± ${row.std.toFixed(4)}`
}

function metricLabel(
  data: { objective: { metric: string }; rows: LeaderboardRow[] } | undefined,
  metric: string,
): string {
  const key = metric || data?.objective.metric
  const found = data?.rows.find((row) => row.cross_validated.length)?.cross_validated.find(
    (item) => item.key === key,
  )
  return found?.label ?? key ?? 'Метрика'
}
