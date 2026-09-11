/**
 * Честность интерфейса: успех не показывается раньше ответа сервера, повторный клик
 * не отправляет второй импорт, устаревший ответ не затирает свежий, структурированная
 * ошибка доходит до экрана вместе с тем, что делать.
 */
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ApiFailure } from '../../api/client'
import * as endpoints from '../../api/endpoints'
import type { DatasetSnapshot } from '../../api/types'
import { DatasetsView } from './DatasetsView'

function snapshot(overrides: Partial<DatasetSnapshot> = {}): DatasetSnapshot {
  return {
    dataset_id: 'ds_0000000000000000000001',
    name: 'sales',
    source: 'file',
    fingerprint: 'abcdef0123456789abcdef',
    fingerprint_algorithm: 'dataarena-logical-sha256-v1',
    row_count: 60,
    column_count: 3,
    columns: [{ name: 'spend', dtype: 'Float64', logical_type: 'float', nullable: false }],
    created_at: '2026-09-07T09:00:00Z',
    row_key: [],
    package_ref: null,
    warnings: [],
    ...overrides,
  }
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('импорт', () => {
  it('не показывает успех, пока сервер не ответил', async () => {
    let release: (value: { dataset: DatasetSnapshot }) => void = () => {}
    vi.spyOn(endpoints, 'listDatasets').mockResolvedValue({ datasets: [], total: 0 })
    vi.spyOn(endpoints, 'importDataset').mockReturnValue(
      new Promise((resolve) => {
        release = resolve
      }),
    )

    render(<DatasetsView selectedId={null} onSelect={() => {}} />)
    await userEvent.upload(
      screen.getByLabelText('Файл датасета'),
      new File(['x'], 'sales.parquet'),
    )

    // пока запрос в полёте, никакого «импортировано» на экране быть не должно
    expect(screen.queryByText(/Импортирован/)).not.toBeInTheDocument()
    expect(screen.getByText(/Импорт идёт/)).toBeInTheDocument()

    release({ dataset: snapshot() })
    await waitFor(() => expect(screen.getByText(/Импортирован/)).toBeInTheDocument())
  })

  it('блокирует поле на время запроса, чтобы двойной клик не отправил два импорта', async () => {
    vi.spyOn(endpoints, 'listDatasets').mockResolvedValue({ datasets: [], total: 0 })
    const importSpy = vi
      .spyOn(endpoints, 'importDataset')
      .mockReturnValue(new Promise(() => {}))

    render(<DatasetsView selectedId={null} onSelect={() => {}} />)
    const input = screen.getByLabelText('Файл датасета')
    await userEvent.upload(input, new File(['x'], 'sales.parquet'))

    await waitFor(() => expect(input).toBeDisabled())
    expect(importSpy).toHaveBeenCalledTimes(1)
  })

  it('структурированная ошибка доходит до экрана вместе с действием', async () => {
    vi.spyOn(endpoints, 'listDatasets').mockResolvedValue({ datasets: [], total: 0 })
    vi.spyOn(endpoints, 'importDataset').mockRejectedValue(
      new ApiFailure('api_error', 'package_unsafe_path', 'Путь выходит за пределы пакета'),
    )

    render(<DatasetsView selectedId={null} onSelect={() => {}} />)
    await userEvent.upload(screen.getByLabelText('Файл датасета'), new File(['x'], 'evil.dapkg.zip'))

    await waitFor(() =>
      expect(screen.getByText('Путь выходит за пределы пакета')).toBeInTheDocument(),
    )
    //код ошибки не должен потеряться по дороге через транспорт
    expect(screen.getByText(/package_unsafe_path/)).toBeInTheDocument()
  })

  it('недоступный backend не выглядит как ошибка датасета', async () => {
    vi.spyOn(endpoints, 'listDatasets').mockRejectedValue(
      new ApiFailure('backend_unavailable', 'backend_unavailable', 'Пустой ответ'),
    )

    render(<DatasetsView selectedId={null} onSelect={() => {}} />)

    await waitFor(() => expect(screen.getByText(/backend недоступен/)).toBeInTheDocument())
    expect(screen.getByText(/Запустите backend на порту 8520/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Повторить запрос' })).toBeInTheDocument()
  })
})

describe('устаревший ответ', () => {
  it('медленный ответ по прежнему датасету не затирает свежий', async () => {
    vi.spyOn(endpoints, 'listDatasets').mockResolvedValue({ datasets: [], total: 0 })
    vi.spyOn(endpoints, 'previewDataset').mockResolvedValue({
      dataset_id: 'x',
      columns: [],
      rows: [],
      offset: 0,
      limit: 25,
      total_rows: 0,
    })

    const slow = new Promise((resolve) =>
      setTimeout(() => resolve({ dataset: snapshot({ name: 'старый' }), profile: profileOf('старый') }), 30),
    )
    const fast = Promise.resolve({ dataset: snapshot({ name: 'новый' }), profile: profileOf('новый') })
    const getSpy = vi.spyOn(endpoints, 'getDataset')
    getSpy.mockReturnValueOnce(slow as never).mockReturnValueOnce(fast as never)

    const view = render(<DatasetsView selectedId="ds_0000000000000000000001" onSelect={() => {}} />)
    view.rerender(<DatasetsView selectedId="ds_0000000000000000000002" onSelect={() => {}} />)

    await waitFor(() => expect(screen.getByText('новый')).toBeInTheDocument())
    //ждём, пока медленный ответ гарантированно придёт, и проверяем, что он проигнорирован
    await new Promise((resolve) => setTimeout(resolve, 60))
    expect(screen.queryByText('старый')).not.toBeInTheDocument()
  })
})

function profileOf(columnName: string) {
  return {
    row_count: 60,
    column_count: 1,
    duplicate_row_count: 0,
    missing_cell_count: 0,
    missing_cell_ratio: 0,
    columns: [
      {
        name: columnName,
        dtype: 'Float64',
        logical_type: 'float',
        missing_count: 0,
        missing_ratio: 0,
        unique_count: 60,
        unique_ratio: 1,
        is_constant: false,
        is_near_constant: false,
        is_probable_id: false,
        id_reason: null,
        numeric_stats: null,
        top_values: null,
        examples: [],
      },
    ],
  }
}
