import { request } from './client'
import type {
  AnalyzeResponse,
  DatasetCard,
  DatasetSnapshot,
  PreviewPage,
  RunCard,
  RunEvent,
  RunSummary,
  RunView,
  StartRunResponse,
} from './types'

export const listDatasets = () =>
  request<{ datasets: DatasetSnapshot[]; total: number }>('/api/datasets')

export const getDataset = (datasetId: string) =>
  request<DatasetCard>(`/api/datasets/${datasetId}`)

export const previewDataset = (datasetId: string, offset = 0, limit = 25) =>
  request<PreviewPage>(`/api/datasets/${datasetId}/preview?offset=${offset}&limit=${limit}`)

export const importDataset = (file: File) => {
  const form = new FormData()
  form.append('file', file)
  // импорт пакета включает пересчёт отпечатка по всем строкам, поэтому таймаут больше
  return request<{ dataset: DatasetSnapshot }>('/api/datasets', {
    method: 'POST',
    body: form,
    timeoutMs: 120_000,
  })
}

export const analyzeTask = (datasetId: string, targetColumn: string) =>
  request<AnalyzeResponse>(`/api/datasets/${datasetId}/task/analyze`, {
    method: 'POST',
    body: JSON.stringify({ target_column: targetColumn }),
  })

export const startRun = (datasetId: string, targetColumn: string, selected: string[] | null) =>
  request<StartRunResponse>('/api/arena/runs', {
    method: 'POST',
    body: JSON.stringify({
      dataset_id: datasetId,
      target_column: targetColumn,
      selected_contenders: selected,
    }),
  })

export const getRun = (runId: string) => request<RunView>(`/api/arena/runs/${runId}`)

export const listRuns = () =>
  request<{ runs: RunSummary[] }>('/api/arena/runs')

export const getRunEvents = (runId: string, since = 0) =>
  request<{ since: number; events: RunEvent[] }>(`/api/arena/runs/${runId}/events?since=${since}`)

export const cancelRun = (runId: string) =>
  request<{ run: RunCard }>(`/api/arena/runs/${runId}/cancel`, { method: 'POST' })
