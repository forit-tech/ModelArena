/**
 * Диагностика одной модели: что с ней происходит.
 *
 * Отвечает на другой вопрос, чем таблица результатов, и на экране они не смешиваются.
 * Таблица говорит, какую модель выбрать по цели; диагностика — что с выбранной не так.
 *
 * Ни одного вывода без чисел. Слово «переобучение» здесь появляется только рядом
 * с измеренным разрывом и порогом, при котором оно сказано; если замера нет, экран
 * пишет «не измерено» и не делает вид, что разрыва нет.
 */
import { getDiagnostics } from '../../api/diagnostics-endpoints'
import type { DiagnosticFinding, Diagnostics } from '../../api/diagnostics-types'
import { FailureNotice } from '../../components/FailureNotice'
import { Notice } from '../../components/Notice'
import { useRequest } from '../../hooks/useRequest'
import { ErrorRowsPanel } from './ErrorRowsPanel'
import { ImportancePanel } from './ImportancePanel'
import { LearningCurvePanel } from './LearningCurvePanel'

const SEVERITY_LEVEL: Record<string, 'info' | 'caution' | 'danger'> = {
  info: 'info',
  caution: 'caution',
  high: 'danger',
}

interface DiagnosticsPanelProps {
  runId: string
  contenderKey: string
  label: string
  onClose: () => void
}

export function DiagnosticsPanel({ runId, contenderKey, label, onClose }: DiagnosticsPanelProps) {
  const report = useRequest(() => getDiagnostics(runId, contenderKey), [runId, contenderKey])
  const data = report.data?.diagnostics

  return (
    <>
      <section className="panel">
        <div className="row spread">
          <h2>Диагностика: {label}</h2>
          <button type="button" onClick={onClose}>
            Закрыть
          </button>
        </div>

        {report.failure ? (
          <FailureNotice failure={report.failure} onRetry={report.reload} />
        ) : null}
        {report.loading && !data ? <p className="hint">Считаем…</p> : null}

        {data ? (
          <>
            {data.findings.map((finding) => (
              <FindingNotice key={finding.code} finding={finding} />
            ))}

            {data.notes.map((note) => (
              <p key={note} className="hint">
                {note}
              </p>
            ))}

            <h3 className="subhead">По фолдам, {data.metric_label}</h3>
            <p className="hint">
              Разрыв измерен: на обучающей части модель проверена тем же конвейером, которым
              считалась проверочная. Эти строки она видела при обучении, поэтому результат
              по ним завышен по построению — и именно поэтому годится как верхняя опора.
            </p>
            <div className="scroll">
              <table className="leaderboard">
                <thead>
                  <tr>
                    <th>Фолд</th>
                    <th>Проверочная</th>
                    <th>Обучающая</th>
                    <th>Разрыв</th>
                    <th>Строк</th>
                  </tr>
                </thead>
                <tbody>
                  {data.folds.map((fold) => (
                    <tr key={fold.fold}>
                      <td>{fold.fold}</td>
                      <td>{show(fold.validation_score)}</td>
                      <td>{show(fold.train_score)}</td>
                      <td className={fold.gap !== null && fold.gap > 0 ? 'indistinguishable' : undefined}>
                        {fold.gap === null ? 'не измерен' : fold.gap.toFixed(4)}
                      </td>
                      <td className="muted">
                        {fold.validation_rows} / {fold.probe_rows}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <dl className="facts">
              <div>
                <dt>Среднее по фолдам</dt>
                <dd>{show(data.mean_validation)}</dd>
              </div>
              <div>
                <dt>Разброс между фолдами</dt>
                <dd>{show(data.spread_validation)}</dd>
              </div>
              <div>
                <dt>Средний разрыв</dt>
                <dd>{data.mean_gap === null ? 'не измерен' : data.mean_gap.toFixed(4)}</dd>
              </div>
            </dl>

            <details>
              <summary className="hint">Пороги, при которых слой делает выводы</summary>
              <ul className="journal">
                {Object.entries(data.thresholds).map(([key, value]) => (
                  <li key={key}>
                    <strong>{key}</strong>{' '}
                    {value.value !== undefined
                      ? value.value
                      : `осторожно от ${value.caution}, высоко от ${value.high}`}{' '}
                    — <span className="muted">{value.meaning}</span>
                  </li>
                ))}
              </ul>
            </details>
          </>
        ) : null}
      </section>

      {data?.classification?.available ? <ClassificationBlock data={data} /> : null}
      {data?.regression?.available ? <RegressionBlock data={data} /> : null}

      {data ? (
        <>
          <ErrorRowsPanel runId={runId} contenderKey={contenderKey} taskType={data.task_type} />
          <ImportancePanel runId={runId} contenderKey={contenderKey} />
          <LearningCurvePanel runId={runId} contenderKey={contenderKey} />
        </>
      ) : null}
    </>
  )
}

function FindingNotice({ finding }: { finding: DiagnosticFinding }) {
  return (
    <Notice
      level={SEVERITY_LEVEL[finding.severity] ?? 'caution'}
      title={finding.title}
      why={
        <>
          {finding.explanation} <span className="muted">{finding.consequence}</span>
        </>
      }
      action={finding.suggestion}
    >
      <details>
        <summary className="hint">Измерения, на которых основан вывод</summary>
        <pre className="evidence">{JSON.stringify(finding.evidence, null, 2)}</pre>
      </details>
    </Notice>
  )
}

function ClassificationBlock({ data }: { data: Diagnostics }) {
  const classification = data.classification
  if (!classification?.confusion || !classification.per_class) {
    return null
  }

  const { labels, rows } = classification.confusion

  return (
    <section className="panel">
      <h2>Разбор классификации</h2>
      <p className="hint">
        {classification.is_binary
          ? 'Бинарная задача: есть положительный класс, ложные срабатывания и пропуски.'
          : 'Многоклассовая задача: показано, какой класс с каким путается.'}{' '}
        Посчитано по {classification.n_rows} out-of-fold предсказаниям.
      </p>

      <h3 className="subhead">Матрица ошибок</h3>
      <div className="scroll">
        <table className="leaderboard">
          <thead>
            <tr>
              <th>факт \ предсказано</th>
              {labels.map((label) => (
                <th key={label}>{label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => (
              <tr key={labels[index]}>
                <th scope="row">{labels[index]}</th>
                {row.map((value, column) => (
                  <td
                    key={`${labels[index]}-${labels[column]}`}
                    className={index === column ? 'stable' : undefined}
                  >
                    {value}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h3 className="subhead">По классам</h3>
      <div className="scroll">
        <table className="leaderboard">
          <thead>
            <tr>
              <th>Класс</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1</th>
              <th>Наблюдений</th>
            </tr>
          </thead>
          <tbody>
            {classification.per_class.map((row) => (
              <tr key={row.label}>
                <td>{row.label}</td>
                <td>{row.precision.toFixed(4)}</td>
                <td>{row.recall.toFixed(4)}</td>
                <td>{row.f1.toFixed(4)}</td>
                <td className="muted">{row.support}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h3 className="subhead">Уверенность</h3>
      {classification.confidence?.available ? (
        <>
          <p className="hint">{classification.confidence.note}</p>
          <div className="scroll">
            <table className="leaderboard">
              <thead>
                <tr>
                  <th>Уверенность</th>
                  <th>Обещано</th>
                  <th>Угадано</th>
                  <th>Строк</th>
                </tr>
              </thead>
              <tbody>
                {(classification.confidence.bins ?? []).map((bin) => (
                  <tr key={`${bin.from}-${bin.to}`}>
                    <td>
                      {bin.from.toFixed(2)}–{bin.to.toFixed(2)}
                    </td>
                    <td>{bin.mean_confidence.toFixed(4)}</td>
                    <td
                      className={
                        bin.mean_confidence - bin.accuracy > 0.1 ? 'indistinguishable' : 'stable'
                      }
                    >
                      {bin.accuracy.toFixed(4)}
                    </td>
                    <td className="muted">{bin.count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        <p className="hint">{classification.confidence?.reason ?? 'Неприменимо.'}</p>
      )}
    </section>
  )
}

function RegressionBlock({ data }: { data: Diagnostics }) {
  const regression = data.regression
  if (!regression?.residuals) {
    return null
  }

  const residuals = regression.residuals

  return (
    <section className="panel">
      <h2>Разбор регрессии</h2>
      <p className="hint">
        Матрица ошибок и полнота здесь не имеют смысла и не показываются. Вопросы у регрессии
        другие: смещена ли ошибка в одну сторону и растёт ли она с величиной цели.
      </p>

      <Notice
        level={residuals.systematic_bias ? 'caution' : 'info'}
        title={residuals.systematic_bias ? 'Систематическое смещение' : 'Смещения не видно'}
        why={residuals.bias_note}
        action={
          residuals.systematic_bias
            ? 'Смещение постоянно по знаку: его можно скорректировать, но сначала стоит понять причину.'
            : 'Специальных действий не требуется.'
        }
      />

      <h3 className="subhead">Ошибка по диапазонам цели</h3>
      <div className="scroll">
        <table className="leaderboard">
          <thead>
            <tr>
              <th>Диапазон</th>
              <th>MAE</th>
              <th>RMSE</th>
              <th>Средний остаток</th>
              <th>Строк</th>
            </tr>
          </thead>
          <tbody>
            {(regression.by_target_range ?? []).map((bucket) => (
              <tr key={`${bucket.from}-${bucket.to}`}>
                <td>
                  {bucket.from.toFixed(2)}–{bucket.to.toFixed(2)}
                </td>
                <td>{bucket.mae.toFixed(4)}</td>
                <td>{bucket.rmse.toFixed(4)}</td>
                <td>{bucket.mean_residual.toFixed(4)}</td>
                <td className="muted">{bucket.count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h3 className="subhead">Крупнейшие ошибки</h3>
      <div className="scroll">
        <table className="leaderboard">
          <thead>
            <tr>
              <th>Строка</th>
              <th>Факт</th>
              <th>Предсказано</th>
              <th>Ошибка</th>
              <th>Относительная</th>
            </tr>
          </thead>
          <tbody>
            {(regression.largest_errors ?? []).slice(0, 10).map((row) => (
              <tr key={row.row_id}>
                <td>{row.row_id}</td>
                <td>{row.y_true.toFixed(4)}</td>
                <td>{row.y_pred.toFixed(4)}</td>
                <td>{row.absolute_error.toFixed(4)}</td>
                <td className="muted">
                  {/* относительная ошибка определена не всегда: при цели около нуля
                      она превращается в тысячи процентов и означает лишь маленький знаменатель */}
                  {row.relative_error === null
                    ? 'не определена'
                    : `${(row.relative_error * 100).toFixed(1)}%`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

function show(value: number | null): string {
  return value === null ? 'не посчитана' : value.toFixed(4)
}
