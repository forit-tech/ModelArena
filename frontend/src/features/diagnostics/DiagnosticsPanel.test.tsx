/**
 * Честность экранов диагностики и применения модели.
 *
 * Проверяются утверждения, а не вёрстка: «разрыв не измерен» не должен выглядеть как
 * «разрыва нет», отказ по схеме должен доходить до пользователя с названием колонки,
 * а несопоставимые эксперименты — быть помечены до того, как рядом окажутся их числа.
 */
import { render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { DiagnosticsPanel } from './DiagnosticsPanel'
import { ExperimentsView } from '../experiments/ExperimentsView'

function respond(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  })
}

function diagnosticsBody(overrides: Record<string, unknown> = {}) {
  return {
    diagnostics: {
      contender_key: 'model',
      task_type: 'binary',
      metric: 'roc_auc',
      metric_label: 'ROC-AUC',
      higher_is_better: true,
      folds: [
        {
          fold: 0,
          validation_rows: 40,
          validation_score: 0.8,
          probe_rows: 0,
          train_score: null,
          gap: null,
        },
      ],
      mean_validation: 0.8,
      spread_validation: null,
      mean_gap: null,
      findings: [],
      thresholds: {
        relative_gap: { caution: 0.1, high: 0.25, meaning: 'разрыв в долях метрики' },
      },
      classification: null,
      regression: null,
      notes: ['Замер на обучающей части отсутствует: разрыв измерить нечем.'],
      ...overrides,
    },
  }
}

describe('DiagnosticsPanel', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('без замера на обучающей части пишет «не измерен», а не ноль', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => respond(diagnosticsBody())))
    render(
      <DiagnosticsPanel runId="run_0123456789abcdef0123" contenderKey="model" label="Модель" onClose={() => {}} />,
    )

    await waitFor(() => expect(screen.getAllByText('не измерен').length).toBeGreaterThan(0))
    //ноль здесь был бы утверждением, которого измерение не давало
    expect(screen.queryByText('0.0000')).toBeNull()
    expect(screen.getByText(/измерить нечем/)).toBeInTheDocument()
  })

  it('предупреждение о переобучении показывает числа и порог', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        respond(
          diagnosticsBody({
            mean_gap: 0.18,
            findings: [
              {
                code: 'train_validation_gap',
                severity: 'high',
                title: 'Разрыв между обучающей и проверочной частью по ROC-AUC',
                evidence: { gap: 0.18, relative_threshold: 0.25 },
                explanation: 'На обучающей части ROC-AUC = 0.9800, на проверочной 0.8000.',
                consequence: 'На новых данных результат будет ближе к проверочному числу.',
                suggestion: 'Упростить модель или добавить данных.',
              },
            ],
          }),
        ),
      ),
    )
    render(
      <DiagnosticsPanel runId="run_0123456789abcdef0123" contenderKey="model" label="Модель" onClose={() => {}} />,
    )

    await waitFor(() =>
      expect(screen.getByText(/На обучающей части ROC-AUC = 0.9800/)).toBeInTheDocument(),
    )
    //измерения доступны рядом с выводом, а не спрятаны
    expect(screen.getByText(/Измерения, на которых основан вывод/)).toBeInTheDocument()
    expect(screen.getByText(/Пороги, при которых слой делает выводы/)).toBeInTheDocument()
  })
})

describe('ExperimentsView', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('несопоставимые прогоны помечает до показа их чисел', async () => {
    const experiment = (runId: string, seed: number) => ({
      run_id: runId,
      created_at: '2026-09-14T10:00:00+00:00',
      finished_at: null,
      state: 'SUCCEEDED',
      error_code: null,
      runtime_seconds: 5,
      dataset: { dataset_id: 'ds_1', fingerprint: 'fp' },
      task: {
        task_type: 'binary',
        target_column: 'churned',
        n_features: 3,
        feature_columns: ['a', 'b', 'c'],
        positive_label: 'True',
        class_labels: ['False', 'True'],
      },
      protocol: {
        cv_splitter: 'stratified_k_fold',
        n_splits: 3,
        holdout_rows: 40,
        train_pool_rows: 160,
        seed,
        fold_assignment_hash: `hash${seed}`,
      },
      experiment_fingerprint: `fp${seed}`,
      contenders: { planned: 2, succeeded: 2, failed: 0, skipped: 0, cancelled: 0, keys: [] },
      champion: {
        objective: 'roc_auc',
        objective_description: 'Максимизируем roc_auc.',
        contender_key: 'model',
        label: 'Модель',
        reason: 'Чемпион.',
        score: 0.9,
      },
      artifacts: [],
    })

    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        if (String(input).includes('compare')) {
          return respond({
            runs: [experiment('run_a', 1), experiment('run_b', 2)],
            differences: [
              {
                key: 'fold_assignment_hash',
                label: 'фактическое разбиение',
                values: [
                  { run_id: 'run_a', value: 'hash1' },
                  { run_id: 'run_b', value: 'hash2' },
                ],
                consequence: 'Фолды разные. Разница частично объясняется разбиением.',
              },
            ],
            comparable: false,
            verdict: 'Условия отличаются по пунктам: фактическое разбиение.',
          })
        }

        return respond({ experiments: [experiment('run_a', 1), experiment('run_b', 2)], total: 2 })
      }),
    )

    const { default: userEvent } = await import('@testing-library/user-event')
    render(<ExperimentsView onOpenRun={() => {}} />)

    await waitFor(() => expect(screen.getAllByRole('checkbox')).toHaveLength(2))
    for (const box of screen.getAllByRole('checkbox')) {
      await userEvent.click(box)
    }
    await userEvent.click(screen.getByRole('button', { name: /Сравнить выбранные/ }))

    await waitFor(() => expect(screen.getByText('Условия отличаются')).toBeInTheDocument())
    //текст встречается и в вердикте, и в списке различий — оба места нужны
    expect(screen.getAllByText(/фактическое разбиение/).length).toBeGreaterThan(1)
    expect(screen.getByText(/Фолды разные/)).toBeInTheDocument()
  })
})
