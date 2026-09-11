/**
 * Показ неудачи запроса. Повтор — явное действие пользователя (D-18): молчаливый
 * ретрай скрывает причину и удлиняет ожидание.
 */
import { ApiFailure } from '../api/client'
import { BACKEND_PORT } from '../config'
import { Notice } from './Notice'

interface FailureNoticeProps {
  failure: ApiFailure
  onRetry?: () => void
}

export function FailureNotice({ failure, onRetry }: FailureNoticeProps) {
  const view = failure.view(BACKEND_PORT)

  return (
    <Notice level="danger" title={view.title} why={view.explanation} action={view.action}>
      {onRetry ? (
        <div className="row" style={{ marginTop: 8 }}>
          <button type="button" onClick={onRetry}>
            Повторить запрос
          </button>
        </div>
      ) : null}
    </Notice>
  )
}
