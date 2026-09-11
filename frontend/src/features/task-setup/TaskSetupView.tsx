/**
 * Постановка задачи: цель, признаки, протокол оценки.
 *
 * Раздел ничего не обучает и не имеет права выглядеть так, будто обучает.
 * Каждое решение backend показывается вместе с объяснением, а каждое предупреждение —
 * вместе с тем, что с ним делать (D-19).
 */
import { useState } from 'react'

import { analyzeTask, getDataset } from '../../api/endpoints'
import { FailureNotice } from '../../components/FailureNotice'
import { Notice } from '../../components/Notice'
import { useRequest } from '../../hooks/useRequest'
import { ReadinessPanel } from './ReadinessPanel'

const TASK_TYPE_LABELS: Record<string, string> = {
  binary: 'бинарная классификация',
  multiclass: 'многоклассовая классификация',
  regression: 'регрессия',
}

const SPLITTER_LABELS: Record<string, string> = {
  stratified_k_fold: 'стратифицированные фолды',
  k_fold: 'обычные фолды',
  group_k_fold: 'фолды по сущностям',
  stratified_group_k_fold: 'стратифицированные фолды по сущностям',
  forward_chaining: 'скользящее окно по времени',
}

interface TaskSetupViewProps {
  datasetId: string | null
  /** Вызывается, когда постановка разобрана: прогон запускается только по ней. */
  onAnalyzed?: (targetColumn: string) => void
}

export function TaskSetupView({ datasetId, onAnalyzed }: TaskSetupViewProps) {
  const [target, setTarget] = useState<string>('')
  const [analyzed, setAnalyzed] = useState<string | null>(null)

  const card = useRequest(datasetId ? () => getDataset(datasetId) : null, [datasetId])
  const analysis = useRequest(
    datasetId && analyzed ? () => analyzeTask(datasetId, analyzed) : null,
    [datasetId, analyzed],
  )

  if (!datasetId) {
    return (
      <section className="panel">
        <h2>Постановка задачи</h2>
        <p className="hint">Сначала выберите датасет в разделе «Датасеты».</p>
      </section>
    )
  }

  return (
    <>
      <section className="panel">
        <h2>Целевая колонка</h2>
        <p className="hint">
          Что предсказываем. От этого выбора зависят тип задачи, набор метрик и стратегия проверки.
        </p>

        {card.failure ? <FailureNotice failure={card.failure} onRetry={card.reload} /> : null}

        <div className="row">
          <select
            aria-label="Целевая колонка"
            value={target}
            onChange={(event) => setTarget(event.target.value)}
          >
            <option value="">— выберите —</option>
            {card.data?.dataset.columns.map((column) => (
              <option key={column.name} value={column.name}>
                {column.name} ({column.logical_type})
              </option>
            ))}
          </select>

          <button
            type="button"
            className="primary"
            disabled={!target || analysis.loading}
            onClick={() => {
              setAnalyzed(target)
              onAnalyzed?.(target)
            }}
          >
            Разобрать постановку
          </button>

          {analysis.loading ? <span className="muted">Считаем…</span> : null}
        </div>

        {analysis.failure ? (
          <FailureNotice failure={analysis.failure} onRetry={analysis.reload} />
        ) : null}
      </section>

      {analysis.data ? (
        <>
          <ReadinessPanel readiness={analysis.data.readiness} />

          <section className="panel">
            <h2>Задача</h2>
            <p className="hint">Предложение системы. Решение остаётся за вами.</p>

            <div className="row">
              <span className="tag accent">{TASK_TYPE_LABELS[analysis.data.task.task_type]}</span>
              {analysis.data.task.positive_label ? (
                <span className="tag">положительный класс: {analysis.data.task.positive_label}</span>
              ) : null}
              <span className="tag">
                строк для обучения: {analysis.data.task.usable_row_count} из {analysis.data.task.row_count}
              </span>
            </div>

            <div className="explain">{analysis.data.task.explanation}</div>

            {analysis.data.task.warnings.map((warning) => (
              <Notice
                key={warning}
                title={warning}
                action="Учитывайте при чтении метрик. Изменить состав данных можно только в DataArena."
              />
            ))}

            {analysis.data.task.suggested_exclusions.length > 0 ? (
              <>
                <h3 style={{ fontSize: 14, marginTop: 16 }}>Предложено исключить из признаков</h3>
                <div className="scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>Колонка</th>
                        <th>Причина</th>
                        <th>Источник</th>
                      </tr>
                    </thead>
                    <tbody>
                      {analysis.data.task.suggested_exclusions.map((record) => (
                        <tr key={record.column}>
                          <td>{record.column}</td>
                          <td className="muted" style={{ whiteSpace: 'normal' }}>
                            {record.reason}
                          </td>
                          <td className="muted">{record.source}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            ) : null}

            <p className="muted" style={{ marginTop: 10 }}>
              Признаков: {analysis.data.task.feature_columns.length} —{' '}
              {analysis.data.task.feature_columns.join(', ')}
            </p>
          </section>

          <section className="panel">
            <h2>Протокол оценки</h2>
            <p className="hint">
              Как будет проверяться качество. Ранжирование пойдёт по кросс-валидации,
              holdout остаётся независимым подтверждением.
            </p>

            <div className="row">
              <span className="tag accent">
                {SPLITTER_LABELS[analysis.data.protocol.cv_splitter] ?? analysis.data.protocol.cv_splitter}
              </span>
              <span className="tag">фолдов: {analysis.data.protocol.n_splits}</span>
              <span className="tag">
                holdout: {Math.round(analysis.data.protocol.holdout_size * 100)}%,{' '}
                {analysis.data.protocol.holdout_strategy}
              </span>
              <span className="tag">seed: {analysis.data.protocol.seed}</span>
            </div>

            <div className="explain">{analysis.data.protocol.explanation}</div>

            {analysis.data.protocol.feasibility_notes.map((note) => (
              <Notice
                key={note}
                level="info"
                title={note}
                action="Ничего делать не нужно — система уже подстроила протокол под данные, но знать об этом стоит."
              />
            ))}

            {analysis.data.protocol.warnings.map((warning) => (
              <Notice
                key={warning}
                title={warning}
                action="Если структура в данных действительно есть — выберите соответствующую колонку в настройках протокола, иначе метрика будет оптимистичной."
              />
            ))}

            {analysis.data.protocol.warnings.length === 0 &&
            analysis.data.protocol.feasibility_notes.length === 0 ? (
              <p className="muted" style={{ marginTop: 10 }}>
                Ограничений и рисков протокола не обнаружено.
              </p>
            ) : null}
          </section>

          <section className="panel">
            <h2>Дальше</h2>
            <p className="hint">
              Обучение появится на следующих этапах. Сейчас ModelArena умеет принять датасет,
              поставить задачу и выбрать протокол — но моделей ещё не запускает.
            </p>
          </section>
        </>
      ) : null}
    </>
  )
}
