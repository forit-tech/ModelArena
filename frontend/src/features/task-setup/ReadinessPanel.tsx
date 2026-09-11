/**
 * Готовность к честному сравнению.
 *
 * Показывается статус и список причин, а не балл: одно число прячет, какой именно сигнал
 * его испортил, и провоцирует улучшать метрику вместо данных (D-22).
 */
import { Notice } from '../../components/Notice'
import type { ReadinessFinding, ReadinessReport, ReadinessStatus } from '../../api/types'

const STATUS_LABEL: Record<ReadinessStatus, string> = {
  READY: 'Готово к сравнению',
  CAUTION: 'Сравнение возможно с оговорками',
  HIGH_RISK: 'Высокий риск ложных выводов',
  BLOCKED: 'Честное сравнение невозможно',
}

const STATUS_HINT: Record<ReadinessStatus, string> = {
  READY: 'Препятствий не обнаружено.',
  CAUTION: 'Результат читается с поправками, названными ниже.',
  HIGH_RISK: 'Leaderboard может ранжировать артефакты данных, а не качество моделей.',
  BLOCKED: 'Пока причина не устранена, любые метрики будут вводить в заблуждение.',
}

function levelOf(finding: ReadinessFinding): 'info' | 'caution' | 'danger' {
  if (finding.blocking || finding.severity === 'high_risk') {
    return 'danger'
  }

  return finding.severity === 'info' ? 'info' : 'caution'
}

export function ReadinessPanel({ readiness }: { readiness: ReadinessReport }) {
  return (
    <section className="panel">
      <h2>Готовность</h2>
      <p className="hint">
        Отвечает не на вопрос «хороший ли датасет», а на вопрос «можно ли честно сравнивать
        модели на этих данных при выбранной задаче и протоколе».
      </p>

      <div className="row">
        <span className={`tag ${readiness.status === 'READY' ? '' : 'accent'}`}>
          {STATUS_LABEL[readiness.status]}
        </span>
        <span className="muted">{STATUS_HINT[readiness.status]}</span>
      </div>

      <div className="explain">{readiness.summary}</div>

      {readiness.findings.map((finding) => (
        <Notice
          key={`${finding.code}:${finding.scope}`}
          level={levelOf(finding)}
          title={
            <>
              {finding.explanation}{' '}
              <span className="tag">{finding.scope}</span>
              {finding.blocking ? <span className="tag">блокирует</span> : null}
            </>
          }
          why={
            <>
              {finding.consequence} <span className="muted">Порог: {finding.threshold}.</span>
            </>
          }
          action={finding.suggested_action}
        />
      ))}
    </section>
  )
}
