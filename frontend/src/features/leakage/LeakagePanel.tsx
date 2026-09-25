import type {
  LeakageEvidenceLevel,
  LeakageReport,
  LeakageSignal,
} from '../../api/types'
import { Notice } from '../../components/Notice'

const RISK_LABEL: Record<LeakageReport['risk'], string> = {
  none: 'явных механизмов не найдено',
  suspected: 'есть подозрения',
  high: 'высокий риск',
  confirmed: 'механизм подтверждён',
}

const EVIDENCE_LABEL: Record<LeakageEvidenceLevel, string> = {
  structural: 'Структурные доказательства',
  evidence: 'Измеренные свидетельства',
  heuristic: 'Эвристические подозрения',
}

const EVIDENCE_HINT: Record<LeakageEvidenceLevel, string> = {
  structural:
    'Механизм следует из структуры данных или фактического пересечения границ протокола.',
  evidence:
    'Механизм поддержан измерением на данных, но требует интерпретации в контексте процесса.',
  heuristic:
    'Повод проверить колонку вручную. Одно имя признака не является доказательством утечки.',
}

interface LeakagePanelProps {
  report: LeakageReport
  context?: 'task' | 'run'
}

export function LeakagePanel({ report, context = 'task' }: LeakagePanelProps) {
  const grouped = (['structural', 'evidence', 'heuristic'] as LeakageEvidenceLevel[])
    .map((level) => ({
      level,
      signals: report.signals.filter((signal) => signal.evidence_level === level),
    }))
    .filter((group) => group.signals.length > 0)

  return (
    <section className="panel">
      <h2>Проверка утечек</h2>
      <p className="hint">
        {context === 'run'
          ? 'Это снимок Leakage Guard на момент запуска именно этого прогона. Он относится к тем же признакам и тем же границам фолдов, по которым построен leaderboard.'
          : 'Leakage Guard проверяет признаки вместе с выбранной задачей и протоколом. Уровни доказательности разделены намеренно: подозрение по имени не равно доказанному механизму.'}
      </p>

      <div className="row">
        <span className={report.risk === 'none' ? 'tag' : 'tag accent'}>
          {RISK_LABEL[report.risk]}
        </span>
        <span className="muted">
          сигналов: {report.signals.length} · не проверено: {report.not_evaluated.length}
        </span>
      </div>

      <div className="explain">{report.summary}</div>

      {report.not_evaluated.length > 0 ? (
        <>
          <Notice
            level="caution"
            title={`Проверено не всё: ${report.not_evaluated.length}`}
            why="Невыполненная проверка не означает, что соответствующего механизма утечки нет."
            action="Посмотрите причины ниже и не трактуйте молчание Guard по этим пунктам как чистый результат."
          />
          <h3 className="subhead">Что не удалось проверить</h3>
          <div className="grid">
            {report.not_evaluated.map((item) => (
              <article className="card" key={`${item.check}:${item.scope}`}>
                <div className="row">
                  <strong>{item.check}</strong>
                  <span className="tag">область: {item.scope}</span>
                </div>
                <p>{item.reason}</p>
                <p className="muted">{item.consequence}</p>
              </article>
            ))}
          </div>
        </>
      ) : (
        <Notice
          level="info"
          title="Все проверки Guard выполнились"
          why="В отчёте нет технически пропущенных механизмов."
          action="Это всё равно не доказывает отсутствие утечки: бизнес-момент появления признака знает только человек."
        />
      )}

      {grouped.map(({ level, signals }) => (
        <div key={level}>
          <h3 className="subhead">{EVIDENCE_LABEL[level]}</h3>
          <p className="hint">{EVIDENCE_HINT[level]}</p>
          <div className="grid">
            {signals.map((signal) => (
              <LeakageSignalCard key={signal.code} signal={signal} />
            ))}
          </div>
        </div>
      ))}

      {report.signals.length === 0 ? (
        <p className="muted" style={{ marginTop: 12 }}>
          Найденных сигналов нет. Читайте это вместе с блоком «что не удалось проверить» и
          ограничением Guard ниже.
        </p>
      ) : null}

      <div className="explain" style={{ marginTop: 14 }}>
        <strong>Ограничение Guard.</strong> {report.disclaimer}
      </div>
    </section>
  )
}

function LeakageSignalCard({ signal }: { signal: LeakageSignal }) {
  return (
    <article className="card">
      <div className="row">
        <strong>{signal.code}</strong>
        <span className="tag">{signal.scope}</span>
        {signal.blocking ? <span className="tag accent">блокирует</span> : null}
      </div>

      {signal.columns.length > 0 ? (
        <p className="muted">Колонки: {signal.columns.join(', ')}</p>
      ) : null}

      <p>{signal.explanation}</p>
      <p>
        <strong>Механизм:</strong> {signal.mechanism}
      </p>
      <p>
        <strong>Что делать:</strong> {signal.suggested_action}
      </p>
      <p className="muted">
        Наблюдение: {formatObserved(signal.observed_value)}
        {signal.threshold ? ` · порог: ${signal.threshold}` : ''}
      </p>
    </article>
  )
}

function formatObserved(value: unknown): string {
  if (value === null || value === undefined) {
    return '—'
  }

  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return String(value)
  }

  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}
