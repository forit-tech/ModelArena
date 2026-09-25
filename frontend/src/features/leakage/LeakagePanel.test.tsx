import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import type { LeakageReport } from '../../api/types'
import { LeakagePanel } from './LeakagePanel'

function report(overrides: Partial<LeakageReport> = {}): LeakageReport {
  return {
    risk: 'high',
    summary: 'Найдены сигналы, требующие проверки.',
    signals: [
      {
        code: 'target_copy_among_features',
        evidence_level: 'structural',
        scope: 'features',
        columns: ['outcome_after'],
        observed_value: 1,
        threshold: 'exact match',
        explanation: 'Признак совпадает с целью.',
        mechanism: 'Модель получает ответ до оценки и leaderboard становится оптимистичным.',
        suggested_action: 'Исключить колонку и повторить прогон.',
        blocking: true,
      },
      {
        code: 'future_name',
        evidence_level: 'heuristic',
        scope: 'features',
        columns: ['status_after'],
        observed_value: 'status_after',
        threshold: '',
        explanation: 'Название похоже на поле, известное после события.',
        mechanism: 'Если поле появляется после целевого события, оно раскрывает будущее.',
        suggested_action: 'Проверить момент появления поля.',
        blocking: false,
      },
    ],
    not_evaluated: [
      {
        check: 'single_feature_predictive',
        scope: 'features',
        reason: 'Проверка не выполнилась.',
        consequence: 'По этому механизму вывода нет.',
      },
    ],
    disclaimer: 'Guard не может доказать ни наличие, ни отсутствие утечки.',
    ...overrides,
  }
}

describe('LeakagePanel', () => {
  it('разделяет уровни доказательности и показывает механизм, а не только тревогу', () => {
    render(<LeakagePanel report={report()} />)

    expect(screen.getByText('Структурные доказательства')).toBeInTheDocument()
    expect(screen.getByText('Эвристические подозрения')).toBeInTheDocument()
    expect(screen.getByText(/Модель получает ответ до оценки/)).toBeInTheDocument()
    expect(screen.getByText(/Проверить момент появления поля/)).toBeInTheDocument()
  })

  it('обязательно показывает невыполненные проверки и ограничение Guard', () => {
    render(<LeakagePanel report={report()} context="run" />)

    expect(screen.getByText('Проверено не всё: 1')).toBeInTheDocument()
    expect(screen.getByText('Проверка не выполнилась.')).toBeInTheDocument()
    expect(screen.getByText('По этому механизму вывода нет.')).toBeInTheDocument()
    expect(screen.getByText(/не может доказать ни наличие, ни отсутствие/)).toBeInTheDocument()
  })
})
