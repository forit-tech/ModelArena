/**
 * Честность таблицы результатов на экране.
 *
 * Проверяются утверждения, которые экран делает о превосходстве: что «чемпион не выбран»
 * показан как вывод, а не как пустота; что неразличимые соседи названы неразличимыми;
 * что участник, снятый ограничением, остаётся виден с причиной.
 */
import { render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { LeaderboardPanel } from './LeaderboardPanel'
import type { LeaderboardRow, LeakageReport, PairedComparison } from '../../api/types'

function comparison(overrides: Partial<PairedComparison> = {}): PairedComparison {
  return {
    leader: 'a',
    trailing: 'b',
    metric: 'ROC-AUC',
    mean_difference: 0.02,
    std_difference: 0.001,
    folds_compared: 5,
    folds_won: 5,
    stable: true,
    explanation: 'Впереди на всех 5 фолдах.',
    ...overrides,
  }
}

function row(overrides: Partial<LeaderboardRow> = {}): LeaderboardRow {
  return {
    contender_key: 'model',
    label: 'Модель',
    family: 'linear',
    is_baseline: false,
    rank: 1,
    score: 0.9,
    std: 0.01,
    per_fold: [0.9, 0.9],
    estimated_cost: 1,
    elapsed_seconds: 8.4,
    model_bytes: 44_040_192,
    cross_validated: [
      {
        key: 'roc_auc',
        label: 'ROC-AUC',
        direction: 'higher',
        value: 0.9,
        std: 0.01,
        per_fold: [0.9, 0.9],
        note: '',
      },
    ],
    holdout: [],
    constraints: [],
    eligible: true,
    reason: '',
    versus_next: null,
    versus_baseline: null,
    ...overrides,
  }
}

const cleanLeakage: LeakageReport = {
  risk: 'none',
  summary: 'Явных механизмов утечки не обнаружено.',
  signals: [],
  not_evaluated: [],
  disclaimer:
    'Guard не может доказать ни наличие, ни отсутствие утечки без знания реального процесса.',
}

function mockBoard(
  rows: LeaderboardRow[],
  champion: Record<string, unknown>,
  leakage: LeakageReport = cleanLeakage,
) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)

      if (url.endsWith('/analysis')) {
        return new Response(
          JSON.stringify({
            run_id: 'run_0123456789abcdef0123',
            analysis: {
              readiness: { status: 'READY', summary: '', context: {}, findings: [] },
              leakage,
            },
            note: '',
          }),
          { status: 200, headers: { 'content-type': 'application/json' } },
        )
      }

      return new Response(
        JSON.stringify({
          run_id: 'run_0123456789abcdef0123',
          run_state: 'SUCCEEDED',
          leaderboard: {
            objective: {
              metric: 'roc_auc',
              task_type: 'binary',
              higher_is_better: true,
              constraints: [],
              tie_breakers: [],
              min_gain_over_baseline: 0,
              description: 'Максимизируем roc_auc.',
            },
            rows,
            champion: {
              contender_key: null,
              label: '',
              reason: '',
              over_baseline: null,
              over_runner_up: null,
              decided_by_tie_breaker: '',
              ...champion,
            },
            baseline_key: 'baseline_majority',
            notes: ['Ранжирование сделано по кросс-валидации.'],
          },
        }),
        { status: 200, headers: { 'content-type': 'application/json' } },
      )
    }),
  )
}

describe('LeaderboardPanel', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('отсутствие чемпиона показывает как вывод, а не как пустоту', async () => {
    mockBoard([row()], {
      contender_key: null,
      reason: 'Лидер таблицы не превзошёл точку отсчёта устойчиво.',
    })
    render(<LeaderboardPanel runId="run_0123456789abcdef0123" active={false} />)

    await waitFor(() => expect(screen.getByText('Чемпион не выбран')).toBeInTheDocument())
    expect(screen.getByText(/не превзошёл точку отсчёта/)).toBeInTheDocument()
    //это результат прогона, а не сбой — так и должно быть написано
    expect(screen.getByText(/результат прогона, а не ошибка/)).toBeInTheDocument()
  })

  it('неразличимых соседей называет неразличимыми, а не просто упорядочивает', async () => {
    mockBoard(
      [
        row({ contender_key: 'first', label: 'Первая', versus_next: comparison({ stable: false, mean_difference: 0.0001 }) }),
        row({ contender_key: 'second', label: 'Вторая', rank: 2, score: 0.8999 }),
      ],
      { contender_key: 'first', label: 'Первая', reason: 'Чемпион.' },
    )
    render(<LeaderboardPanel runId="run_0123456789abcdef0123" active={false} />)

    await waitFor(() => expect(screen.getByText('неразличимы')).toBeInTheDocument())
    //устойчивое превосходство обозначается иначе, чем неразличимость
    expect(screen.queryByText('устойчиво выше')).toBeNull()
  })

  it('снятого ограничением участника оставляет в отчёте с причиной', async () => {
    mockBoard(
      [
        row(),
        row({
          contender_key: 'greedy',
          label: 'Жадная',
          rank: null,
          score: 0.95,
          eligible: false,
          reason: 'Не выполнены условия: precision не ниже 0.99',
        }),
      ],
      { contender_key: 'model', label: 'Модель', reason: 'Чемпион.' },
    )
    render(<LeaderboardPanel runId="run_0123456789abcdef0123" active={false} />)

    await waitFor(() => expect(screen.getByText('Вне ранжирования')).toBeInTheDocument())
    expect(screen.getByText('Жадная')).toBeInTheDocument()
    expect(screen.getByText(/precision не ниже 0.99/)).toBeInTheDocument()
    //результат у неё есть, просто к первому месту она не допущена
    expect(screen.getByText(/результат есть: 0.9500/)).toBeInTheDocument()
  })

  it('показывает измеренное время и размер отдельно от оценки до запуска', async () => {
    mockBoard(
      [row({ estimated_cost: 99, elapsed_seconds: 8.4, model_bytes: 44_040_192 })],
      { contender_key: 'model', label: 'Модель', reason: 'Чемпион.' },
    )
    render(<LeaderboardPanel runId="run_0123456789abcdef0123" active={false} />)

    await waitFor(() => expect(screen.getByText('8.40 с')).toBeInTheDocument())
    expect(screen.getByText('42.0 МБ')).toBeInTheDocument()
    expect(screen.getByText('99.00')).toBeInTheDocument()
  })

  it('не скрывает проверки утечки, которые выполнить не удалось', async () => {
    mockBoard(
      [row()],
      { contender_key: 'model', label: 'Модель', reason: 'Чемпион.' },
      {
        ...cleanLeakage,
        not_evaluated: [
          {
            check: 'single_feature_predictive',
            scope: 'features',
            reason: 'Проверка не выполнилась из-за несовместимого типа.',
            consequence: 'По этому механизму вывода нет.',
          },
        ],
      },
    )
    render(<LeaderboardPanel runId="run_0123456789abcdef0123" active={false} />)

    await waitFor(() => expect(screen.getByText('Проверено не всё: 1')).toBeInTheDocument())
    expect(screen.getByText(/несовместимого типа/)).toBeInTheDocument()
    expect(screen.getByText(/не может доказать/)).toBeInTheDocument()
  })

  it('предупреждает, что на идущем прогоне таблица неполна', async () => {
    mockBoard([row()], { contender_key: 'model', label: 'Модель', reason: 'Чемпион.' })
    render(<LeaderboardPanel runId="run_0123456789abcdef0123" active />)

    await waitFor(() => expect(screen.getByText('Прогон ещё идёт')).toBeInTheDocument())
    expect(screen.getByText(/порядок может поменяться/)).toBeInTheDocument()
  })
})
