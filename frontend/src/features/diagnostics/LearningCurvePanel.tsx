/**
 * Кривая обучения: хватает ли данных.
 *
 * Вывод берётся строго из наблюдаемой кривой. «Добавьте 5000 строк и получите +3%» —
 * обещание, которого измерение не даёт, и его здесь нет. Есть четыре честных состояния:
 * качество растёт, вышло на полку, разброс велик, данных мало для вывода.
 *
 * Вычисление переобучает модель на каждой точке, поэтому запускается по кнопке
 * и сообщает стоимость до начала.
 */
import { useState } from 'react'

import { ApiFailure } from '../../api/client'
import { getLearningCurve } from '../../api/diagnostics-endpoints'
import type { LearningCurve } from '../../api/diagnostics-types'
import { FailureNotice } from '../../components/FailureNotice'
import { Notice } from '../../components/Notice'

const VERDICT_TITLES: Record<string, string> = {
  growing: 'Качество продолжает расти',
  plateau: 'Кривая вышла на полку',
  noisy: 'Разброс перекрывает прирост',
  insufficient: 'Данных мало для вывода',
}

/* Подпись обязана соответствовать вердикту. «Разброс перекрывает прирост» и «качество
   растёт» — разные утверждения, и обещать отдачу от новых строк во втором случае нельзя. */
const VERDICT_ACTIONS: Record<string, string> = {
  growing:
    'Отдача от дополнительных строк ещё есть. Насколько именно — из кривой не следует.',
  plateau: 'Узкое место не в объёме данных: смотрите на признаки и на выбор модели.',
  noisy:
    'Вывода о пользе новых данных эта кривая не даёт. Сузить разброс помогут больше фолдов или больше данных.',
  insufficient: 'Точек мало: постройте кривую на прогоне с большим числом фолдов.',
}

const VERDICT_LEVEL: Record<string, 'info' | 'caution'> = {
  growing: 'info',
  plateau: 'info',
  noisy: 'caution',
  insufficient: 'caution',
}

interface LearningCurvePanelProps {
  runId: string
  contenderKey: string
}

export function LearningCurvePanel({ runId, contenderKey }: LearningCurvePanelProps) {
  const [data, setData] = useState<LearningCurve | null>(null)
  const [failure, setFailure] = useState<ApiFailure | null>(null)
  const [loading, setLoading] = useState(false)

  const build = async () => {
    setLoading(true)
    setFailure(null)

    try {
      const response = await getLearningCurve(runId, contenderKey)
      setData(response.learning_curve)
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

  const values = data?.points.map((point) => point.mean) ?? []
  const low = values.length ? Math.min(...values) : 0
  const high = values.length ? Math.max(...values) : 1
  const span = high - low || 1

  return (
    <section className="panel">
      <h2>Кривая обучения</h2>
      <p className="hint">
        Модель обучается на растущих долях той же обучающей части и проверяется на той же
        проверочной. Так вопрос «хватает ли данных» решается измерением, а не ощущением.
      </p>

      <div className="row">
        <button type="button" className="primary" disabled={loading} onClick={build}>
          {loading ? 'Строим…' : 'Построить кривую'}
        </button>
        {loading ? (
          <span className="muted">модель обучается заново на каждой точке — это долго</span>
        ) : null}
      </div>

      {failure ? <FailureNotice failure={failure} onRetry={build} /> : null}

      {data ? (
        <>
          <p className="hint">{data.cost.note}</p>

          {data.points.length ? (
            <div className="scroll">
              <table className="leaderboard">
                <thead>
                  <tr>
                    <th>Строк в обучении</th>
                    <th>{data.metric_label}</th>
                    <th>Разброс по фолдам</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {data.points.map((point) => (
                    <tr key={point.fraction}>
                      <td>{point.train_rows}</td>
                      <td>{point.mean.toFixed(4)}</td>
                      <td className="muted">
                        {point.spread === null ? '—' : `± ${point.spread.toFixed(4)}`}
                      </td>
                      <td className="bar-cell">
                        <span
                          className="bar"
                          style={{ width: `${((point.mean - low) / span) * 90 + 10}%` }}
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}

          <Notice
            level={VERDICT_LEVEL[data.interpretation.verdict] ?? 'caution'}
            title={VERDICT_TITLES[data.interpretation.verdict] ?? 'Вывод по кривой'}
            why={data.interpretation.text}
            action={VERDICT_ACTIONS[data.interpretation.verdict] ?? VERDICT_ACTIONS.insufficient}
          />

          {data.skipped.length ? (
            <p className="hint">
              {/* кривая с дырой не должна выглядеть сплошной */}
              Пропущено точек: {data.skipped.length}. {data.skipped.slice(0, 3).join('; ')}
            </p>
          ) : null}
        </>
      ) : null}
    </section>
  )
}
