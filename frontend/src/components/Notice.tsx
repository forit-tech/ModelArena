/**
 * Предупреждение, которое обязано объяснять (D-19).
 *
 * Три поля не по вкусу, а по правилу: плашка без ответа «что делать» обучает
 * пользователя игнорировать все плашки подряд, включая те, которые важны.
 * Поэтому `action` — обязательное свойство, и «ничего, просто учитывайте»
 * нужно писать словами, а не оставлять пустоту.
 */
import type { ReactNode } from 'react'

export type NoticeLevel = 'info' | 'caution' | 'danger'

interface NoticeProps {
  level?: NoticeLevel
  /** Что обнаружено. */
  title: ReactNode
  /** Почему это важно. */
  why?: ReactNode
  /** Что можно сделать. Обязательно. */
  action: ReactNode
  children?: ReactNode
}

export function Notice({ level = 'caution', title, why, action, children }: NoticeProps) {
  return (
    <div className={`notice ${level}`} role={level === 'danger' ? 'alert' : 'status'}>
      <div className="notice-title">{title}</div>
      {why ? <div className="notice-why">{why}</div> : null}
      <div className="notice-action">{action}</div>
      {children}
    </div>
  )
}
