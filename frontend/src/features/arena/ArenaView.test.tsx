/**
 * Честность экрана прогона.
 *
 * Проверяется не вёрстка, а утверждения, которые экран делает о результате: что упавший
 * участник виден, что «частично» отличается от «неудачи», что неопределённая метрика
 * не показывается числом и что остановка доступна ровно пока прогон идёт.
 */
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ArenaView } from './ArenaView'
import type { ContenderRun, RunState, RunView } from '../../api/types'

function contender(overrides: Partial<ContenderRun>): ContenderRun {
  return {
    contender_key: 'model',
    label: 'Модель',
    family: 'linear',
    preprocessing_profile: 'scaled_onehot',
    is_baseline: false,
    state: 'SUCCEEDED',
    selection_reason: '',
    error_code: null,
    error_message: '',
    folds_completed: 3,
    n_folds: 3,
    started_at: null,
    finished_at: null,
    elapsed_seconds: 1,
    coverage: null,
    warnings: [],
    metrics: [],
    ...overrides,
  }
}

function runView(state: RunState, contenders: ContenderRun[], active = false): RunView {
  return {
    run: {
      run_id: 'run_0123456789abcdef0123',
      created_at: '2026-09-08T10:00:00Z',
      started_at: '2026-09-08T10:00:01Z',
      finished_at: null,
      state,
      cancel_requested: false,
      error_code: null,
      error_message: '',
      notes: [],
      spec: {
        dataset_id: 'ds_0123456789abcdef',
        fold_assignment_hash: 'abcdef0123456789',
        n_splits: 3,
        holdout_rows: 40,
        train_pool_rows: 160,
        seed: 42,
        experiment_fingerprint: 'fp',
        budget: { max_parallel_fits: 2, threads_per_fit: 3, memory_budget_mb: 4096 },
        protocol: {
          cv_splitter: 'stratified_k_fold',
          n_splits: 3,
          holdout_strategy: 'stratified',
          holdout_size: 0.2,
          shuffle: true,
          seed: 42,
          explanation: '',
          warnings: [],
          feasibility_notes: [],
        },
        task: { task_type: 'binary', target_column: 'churned', class_labels: ['no', 'yes'] },
      },
      contenders,
    },
    progress: {
      folds_completed: 6,
      folds_planned: 9,
      contenders_finished: 2,
      contenders_planned: 3,
      completed_fraction: 0.6667,
    },
    active,
  }
}

function mockApi(view: RunView) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      const body = url.includes('/events')
        ? { since: 0, events: [{ kind: 'fold_completed', at: '2026-09-08T10:00:05.000Z', message: 'фолд 2 из 3 посчитан', contender_key: 'model', fold: 2, n_folds: 3 }] }
        : url.endsWith('/runs')
          ? { run: view.run, duplicate_of_active_run: false }
          : view

      return new Response(JSON.stringify(body), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      })
    }),
  )
}

describe('ArenaView', () => {
  beforeEach(() => vi.useRealTimers())
  afterEach(() => vi.unstubAllGlobals())

  it('без разобранной постановки не предлагает запуск', () => {
    render(<ArenaView datasetId="ds_1" targetColumn={null} />)

    expect(screen.queryByRole('button', { name: /Запустить прогон/ })).toBeNull()
    expect(screen.getByText(/Сначала выберите датасет/)).toBeInTheDocument()
  })

  it('показывает упавшего участника с причиной, а не прячет его', async () => {
    mockApi(
      runView('PARTIAL', [
        contender({ contender_key: 'baseline', label: 'Baseline', is_baseline: true }),
        contender({
          contender_key: 'broken',
          label: 'Сломанная модель',
          state: 'FAILED',
          error_code: 'training_failed',
          error_message: 'Обучение на фолде 0 прервалось: InvalidParameterError.',
        }),
      ]),
    )
    render(<ArenaView datasetId="ds_1" targetColumn="churned" />)
    await userEvent.click(screen.getByRole('button', { name: /Запустить прогон/ }))

    await waitFor(() => expect(screen.getByText('Сломанная модель')).toBeInTheDocument())
    expect(screen.getByText(/InvalidParameterError/)).toBeInTheDocument()
    expect(screen.getByText('отказ')).toBeInTheDocument()
  })

  it('различает «частично» и «неудачу»', async () => {
    mockApi(runView('PARTIAL', [contender({})]))
    render(<ArenaView datasetId="ds_1" targetColumn="churned" />)
    await userEvent.click(screen.getByRole('button', { name: /Запустить прогон/ }))

    await waitFor(() =>
      expect(screen.getByText(/Состояние прогона: частично/)).toBeInTheDocument(),
    )
    //посчитанное годится для сравнения — это должно быть сказано словами
    expect(screen.getByText(/Часть участников не завершилась/)).toBeInTheDocument()
  })

  it('неопределённую метрику показывает как «не посчитана», а не нулём', async () => {
    mockApi(
      runView('SUCCEEDED', [
        contender({
          metrics: [
            {
              split: 'cv',
              n_rows: 160,
              metrics: [
                {
                  key: 'roc_auc',
                  label: 'ROC-AUC',
                  direction: 'higher',
                  value: null,
                  std: null,
                  per_fold: [null, null],
                  note: 'На этой части присутствует один класс из 2.',
                },
              ],
            },
          ],
        }),
      ]),
    )
    render(<ArenaView datasetId="ds_1" targetColumn="churned" />)
    await userEvent.click(screen.getByRole('button', { name: /Запустить прогон/ }))

    await waitFor(() => expect(screen.getByText('не посчитана')).toBeInTheDocument())
    expect(screen.queryByText('0.0000')).toBeNull()
    expect(screen.getByText(/один класс из 2/)).toBeInTheDocument()
  })

  it('показывает счёт посчитанных фолдов, а не выдуманный процент', async () => {
    mockApi(runView('RUNNING', [contender({ state: 'RUNNING', folds_completed: 2 })], true))
    render(<ArenaView datasetId="ds_1" targetColumn="churned" />)
    await userEvent.click(screen.getByRole('button', { name: /Запустить прогон/ }))

    await waitFor(() => expect(screen.getByText('6 из 9')).toBeInTheDocument())
    //счёт есть и в карточке участника, и в журнале: обе записи о фактах, а не о процентах
    expect(screen.getAllByText(/фолд 2 из 3/)).toHaveLength(2)
    expect(screen.queryByText(/%/)).toBeNull()
  })

  it('предлагает остановку только пока прогон идёт', async () => {
    mockApi(runView('RUNNING', [contender({ state: 'RUNNING' })], true))
    render(<ArenaView datasetId="ds_1" targetColumn="churned" />)
    await userEvent.click(screen.getByRole('button', { name: /Запустить прогон/ }))

    await waitFor(() => expect(screen.getByRole('button', { name: 'Остановить' })).toBeEnabled())
  })

  it('пропущенного участника показывает с причиной, а не убирает из списка', async () => {
    mockApi(
      runView('SUCCEEDED', [
        contender({
          contender_key: 'catboost',
          label: 'CatBoost',
          state: 'SKIPPED',
          error_code: 'contender_unavailable',
          error_message: 'Пакет «catboost» не установлен.',
        }),
      ]),
    )
    render(<ArenaView datasetId="ds_1" targetColumn="churned" />)
    await userEvent.click(screen.getByRole('button', { name: /Запустить прогон/ }))

    await waitFor(() => expect(screen.getByText('CatBoost')).toBeInTheDocument())
    expect(screen.getByText('не запускался')).toBeInTheDocument()
    expect(screen.getByText(/не установлен/)).toBeInTheDocument()
  })
})
