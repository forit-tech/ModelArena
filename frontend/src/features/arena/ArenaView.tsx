/**
 * Прогон: состояние, участники, прогресс, остановка.
 *
 * Раздел обязан показывать то, что произошло на самом деле. Три правила, нарушение
 * которых делает экран красивее и лживее:
 *
 * - **упавший участник не исчезает из списка.** Пустое место читается как «модель
 *   проиграла», хотя её либо не запускали, либо она отказала с конкретной причиной;
 * - **PARTIAL — не FAILED и не SUCCEEDED.** Прогон, где часть моделей не обучилась,
 *   не имеет права выглядеть удачным, но и не является провалом: посчитанное годится;
 * - **прогресса «37%» не бывает.** Показывается счёт посчитанных фолдов — единственная
 *   единица, факт завершения которой система действительно наблюдает.
 */
import { useCallback, useEffect, useRef, useState } from 'react'

import { ApiFailure } from '../../api/client'
import { cancelRun, getRun, getRunEvents, startRun } from '../../api/endpoints'
import type { ContenderRun, MetricSet, RunEvent, RunState, RunView } from '../../api/types'
import { FailureNotice } from '../../components/FailureNotice'
import { Notice } from '../../components/Notice'

const POLL_INTERVAL_MS = 1000

const RUN_STATE_LABELS: Record<RunState, string> = {
  PENDING: 'в очереди',
  RUNNING: 'идёт',
  PARTIAL: 'частично',
  SUCCEEDED: 'завершён',
  FAILED: 'неудача',
  CANCELLED: 'остановлен',
  INTERRUPTED: 'прерван перезапуском',
}

const RUN_STATE_EXPLANATIONS: Record<RunState, string> = {
  PENDING: 'Прогон создан, обучение вот-вот начнётся.',
  RUNNING: 'Обучение идёт. Уже посчитанные результаты доступны сразу.',
  PARTIAL: 'Часть участников не завершилась. Посчитанные результаты годны для сравнения между собой.',
  SUCCEEDED: 'Все запланированные участники завершились успешно.',
  FAILED: 'Пригодного результата нет.',
  CANCELLED: 'Остановлено вами. Всё, что успело досчитаться, сохранено.',
  INTERRUPTED: 'Backend был перезапущен во время прогона. Продолжить его нельзя.',
}

const CONTENDER_STATE_LABELS: Record<string, string> = {
  PENDING: 'ожидает',
  RUNNING: 'обучается',
  SUCCEEDED: 'готов',
  SKIPPED: 'не запускался',
  FAILED: 'отказ',
  CANCELLED: 'остановлен',
}

interface ArenaViewProps {
  datasetId: string | null
  targetColumn: string | null
}

export function ArenaView({ datasetId, targetColumn }: ArenaViewProps) {
  const [runId, setRunId] = useState<string | null>(null)
  const [view, setView] = useState<RunView | null>(null)
  const [events, setEvents] = useState<RunEvent[]>([])
  const [failure, setFailure] = useState<ApiFailure | null>(null)
  const [starting, setStarting] = useState(false)
  const [duplicate, setDuplicate] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  const [journalBroken, setJournalBroken] = useState(false)
  //опрос не должен трогать состояние после ухода с экрана или смены прогона
  const generation = useRef(0)

  const poll = useCallback(async (id: string, current: number) => {
    try {
      const next = await getRun(id)

      if (current === generation.current) {
        setView(next)
        setFailure(null)
      }
    } catch (error: unknown) {
      if (current === generation.current) {
        setFailure(
          error instanceof ApiFailure
            ? error
            : new ApiFailure('frontend_error', 'unexpected', String(error)),
        )
      }

      return
    }

    // журнал запрашивается отдельно: он вспомогательный, и его недоступность
    // не должна уносить с экрана состояние прогона и уже посчитанные результаты
    try {
      const journal = await getRunEvents(id)

      if (current === generation.current) {
        setEvents(Array.isArray(journal.events) ? journal.events : [])
        setJournalBroken(false)
      }
    } catch {
      if (current === generation.current) {
        setJournalBroken(true)
      }
    }
  }, [])

  useEffect(() => {
    if (!runId) {
      return
    }

    const current = ++generation.current
    void poll(runId, current)

    const timer = window.setInterval(() => {
      void poll(runId, current)
    }, POLL_INTERVAL_MS)

    return () => {
      generation.current += 1
      window.clearInterval(timer)
    }
  }, [runId, poll])

  useEffect(() => {
    //прогон закончился — опрашивать больше нечего
    if (view && !view.active) {
      generation.current += 1
    }
  }, [view])

  const onStart = async () => {
    if (!datasetId || !targetColumn) {
      return
    }

    setStarting(true)
    setFailure(null)

    try {
      const response = await startRun(datasetId, targetColumn, null)
      setDuplicate(response.duplicate_of_active_run)
      setEvents([])
      setView(null)
      setRunId(response.run.run_id)
    } catch (error: unknown) {
      setFailure(
        error instanceof ApiFailure
          ? error
          : new ApiFailure('frontend_error', 'unexpected', String(error)),
      )
    } finally {
      setStarting(false)
    }
  }

  const onCancel = async () => {
    if (!runId) {
      return
    }

    setCancelling(true)

    try {
      await cancelRun(runId)
      await poll(runId, generation.current)
    } catch (error: unknown) {
      setFailure(
        error instanceof ApiFailure
          ? error
          : new ApiFailure('frontend_error', 'unexpected', String(error)),
      )
    } finally {
      setCancelling(false)
    }
  }

  if (!datasetId || !targetColumn) {
    return (
      <section className="panel">
        <h2>Прогон</h2>
        <p className="hint">
          Сначала выберите датасет и разберите постановку задачи — прогон запускается по уже
          утверждённому протоколу.
        </p>
      </section>
    )
  }

  const run = view?.run
  const progress = view?.progress

  return (
    <>
      <section className="panel">
        <h2>Прогон</h2>
        <p className="hint">
          Все участники обучаются на одних и тех же фолдах, с одним seed и одним набором признаков.
          Способ подготовки признаков может различаться по семействам моделей — честность задаётся
          протоколом, а не одинаковым кодированием.
        </p>

        <div className="row">
          <button type="button" className="primary" disabled={starting || view?.active} onClick={onStart}>
            {starting ? 'Запускаем…' : 'Запустить прогон'}
          </button>

          {view?.active ? (
            <button type="button" disabled={cancelling} onClick={onCancel}>
              {cancelling ? 'Останавливаем…' : 'Остановить'}
            </button>
          ) : null}
        </div>

        {duplicate ? (
          <Notice
            level="info"
            title="Этот эксперимент уже идёт"
            why="Тот же датасет, та же задача, тот же протокол и те же участники — второй такой же прогон только занял бы ресурсы."
            action="Показан уже запущенный прогон. Дождитесь его окончания или остановите."
          />
        ) : null}

        {failure ? <FailureNotice failure={failure} onRetry={() => runId && poll(runId, generation.current)} /> : null}

        {run ? <RunHeader run={run} progress={progress} /> : null}
      </section>

      {run ? <ContenderList contenders={run.contenders} /> : null}
      {journalBroken ? (
        <Notice
          level="caution"
          title="Журнал событий сейчас недоступен"
          why="Состояние прогона и результаты участников выше от этого не зависят — они читаются отдельным запросом."
          action="Обновится при следующем опросе. Если не вернётся, посмотрите лог backend."
        />
      ) : null}
      {events.length ? <Journal events={events} /> : null}
    </>
  )
}

function RunHeader({ run, progress }: { run: RunView['run']; progress?: RunView['progress'] }) {
  const level = run.state === 'FAILED' ? 'danger' : run.state === 'SUCCEEDED' ? 'info' : 'caution'

  return (
    <div className="run-header">
      <Notice
        level={level}
        title={`Состояние прогона: ${RUN_STATE_LABELS[run.state]}`}
        why={run.error_message || RUN_STATE_EXPLANATIONS[run.state]}
        action={
          run.state === 'INTERRUPTED'
            ? 'Запустите прогон заново — сохранённые результаты остаются доступными.'
            : 'Результаты завершённых участников доступны независимо от состояния прогона.'
        }
      />

      <dl className="facts">
        <div>
          <dt>Фолдов посчитано</dt>
          <dd>
            {progress ? `${progress.folds_completed} из ${progress.folds_planned}` : '—'}
          </dd>
        </div>
        <div>
          <dt>Участников завершено</dt>
          <dd>
            {progress ? `${progress.contenders_finished} из ${progress.contenders_planned}` : '—'}
          </dd>
        </div>
        <div>
          <dt>Отпечаток разбиения</dt>
          <dd>
            <code>{run.spec.fold_assignment_hash.slice(0, 12)}</code>
          </dd>
        </div>
        <div>
          <dt>Seed</dt>
          <dd>{run.spec.seed}</dd>
        </div>
      </dl>
      <p className="hint">
        Отпечаток разбиения одинаков у всех участников — это проверяемая, а не обещанная гарантия
        того, что их сравнивали на одних и тех же фолдах.
      </p>
    </div>
  )
}

function ContenderList({ contenders }: { contenders: ContenderRun[] }) {
  return (
    <section className="panel">
      <h2>Участники</h2>
      <p className="hint">
        Список полный. Недоступная библиотека, неуместная на этих данных модель и отказ обучения
        показываются с причиной и не исчезают: пустое место читалось бы как проигрыш.
      </p>

      <ul className="contenders">
        {contenders.map((item) => (
          <ContenderCard key={item.contender_key} contender={item} />
        ))}
      </ul>
    </section>
  )
}

function ContenderCard({ contender }: { contender: ContenderRun }) {
  const cross = contender.metrics.find((part) => part.split === 'cv')

  return (
    <li className={`contender ${contender.state.toLowerCase()}`}>
      <div className="contender-head">
        <strong>{contender.label}</strong>
        {contender.is_baseline ? <span className="tag">точка отсчёта</span> : null}
        <span className={`state ${contender.state.toLowerCase()}`}>
          {CONTENDER_STATE_LABELS[contender.state] ?? contender.state}
        </span>
        {contender.state === 'RUNNING' || contender.state === 'SUCCEEDED' ? (
          <span className="muted">
            фолд {contender.folds_completed} из {contender.n_folds}
          </span>
        ) : null}
      </div>

      {contender.error_message ? (
        <p className="contender-reason">
          <span className="muted">{contender.error_code}:</span> {contender.error_message}
        </p>
      ) : null}

      {contender.state === 'SKIPPED' && !contender.error_message ? (
        <p className="contender-reason">{contender.selection_reason}</p>
      ) : null}

      {contender.warnings.map((warning) => (
        <p key={warning} className="contender-reason">
          {warning}
        </p>
      ))}

      {cross ? <MetricRow metrics={cross} /> : null}

      {contender.coverage && !contender.coverage.complete ? (
        <p className="hint">{contender.coverage.note}</p>
      ) : null}
    </li>
  )
}

function MetricRow({ metrics }: { metrics: MetricSet }) {
  return (
    <table className="metrics">
      <tbody>
        {metrics.metrics.map((metric) => (
          <tr key={metric.key}>
            <th scope="row">{metric.label}</th>
            <td>
              {metric.value === null ? (
                //не «0», а отдельное состояние: неопределённую метрику нельзя показывать числом
                <span className="muted" title={metric.note}>
                  не посчитана
                </span>
              ) : (
                <>
                  {metric.value.toFixed(4)}
                  {metric.std !== null ? <span className="muted"> ± {metric.std.toFixed(4)}</span> : null}
                </>
              )}
            </td>
            <td className="muted">{metric.note}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function Journal({ events }: { events: RunEvent[] }) {
  return (
    <section className="panel">
      <h2>Журнал</h2>
      <p className="hint">
        Каждая запись — факт, который система наблюдала: фолд посчитан, участник завершён.
        Процентов, выведенных из времени, здесь нет.
      </p>

      <ol className="journal">
        {events.map((event, index) => (
          <li key={`${event.at}-${index}`}>
            <span className="muted">{event.at.slice(11, 19)}</span> {event.message}
          </li>
        ))}
      </ol>
    </section>
  )
}
