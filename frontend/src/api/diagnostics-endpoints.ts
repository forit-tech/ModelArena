/**
 * Запросы диагностики, артефактов и экспериментов.
 *
 * Таймауты здесь больше обычных и это не перестраховка: кривая обучения переобучает
 * модель на каждой точке, а перестановочная важность повторяет предсказание по числу
 * повторов. Ждать их дольше — честнее, чем отменять на середине и показывать пустоту.
 */
import { request } from './client'
import type {
  Diagnostics,
  ErrorSelection,
  ExperimentCard,
  ExperimentComparison,
  Importance,
  LearningCurve,
  ModelCard,
  PredictionResult,
} from './diagnostics-types'

export const getDiagnostics = (runId: string, contenderKey: string, metric?: string) =>
  request<{ diagnostics: Diagnostics }>(
    `/api/arena/runs/${runId}/contenders/${contenderKey}/diagnostics${
      metric ? `?metric=${encodeURIComponent(metric)}` : ''
    }`,
  )

export const getErrorRows = (runId: string, contenderKey: string, kind: string, limit = 25) =>
  request<{ errors: ErrorSelection }>(
    `/api/arena/runs/${runId}/contenders/${contenderKey}/errors?kind=${kind}&limit=${limit}`,
  )

export const getLearningCurve = (runId: string, contenderKey: string) =>
  request<{ learning_curve: LearningCurve }>(
    `/api/arena/runs/${runId}/contenders/${contenderKey}/learning-curve`,
    { timeoutMs: 300_000 },
  )

export const getImportance = (runId: string, contenderKey: string, repeats = 5) =>
  request<{ importance: Importance }>(
    `/api/models/${runId}/${contenderKey}/importance?repeats=${repeats}`,
    { timeoutMs: 180_000 },
  )

export const getModelCard = (runId: string, contenderKey: string) =>
  request<ModelCard>(`/api/models/${runId}/${contenderKey}`)

export const predictOne = (
  runId: string,
  contenderKey: string,
  values: Record<string, unknown>,
) =>
  request<{ result: PredictionResult }>(`/api/models/${runId}/${contenderKey}/predict`, {
    method: 'POST',
    body: JSON.stringify({ values }),
  })

export const predictBatch = (runId: string, contenderKey: string, file: File) => {
  const form = new FormData()
  form.append('file', file)
  return request<{ result: PredictionResult; csv: string }>(
    `/api/models/${runId}/${contenderKey}/predict-batch`,
    { method: 'POST', body: form, timeoutMs: 180_000 },
  )
}

export const listExperiments = () =>
  request<{ experiments: ExperimentCard[]; total: number }>('/api/experiments')

export const compareExperiments = (runIds: string[]) =>
  request<ExperimentComparison>('/api/experiments/compare', {
    method: 'POST',
    body: JSON.stringify({ run_ids: runIds }),
  })
