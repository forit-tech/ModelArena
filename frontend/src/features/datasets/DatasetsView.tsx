/**
 * Датасеты: импорт, список, карточка с профилем и страница строк.
 *
 * Правила честности, которые здесь соблюдаются буквально:
 *
 * * успех показывается только после ответа сервера — «загружено» до записи снимка
 *   было бы враньём;
 * * повторный клик по загрузке заблокирован, пока запрос в полёте;
 * * профиль и предупреждения снимка показываются с указанием источника.
 */
import { useRef, useState } from 'react'

import { ApiFailure } from '../../api/client'
import { getDataset, importDataset, listDatasets, previewDataset } from '../../api/endpoints'
import type { DatasetSnapshot } from '../../api/types'
import { FailureNotice } from '../../components/FailureNotice'
import { Notice } from '../../components/Notice'
import { useRequest } from '../../hooks/useRequest'

interface DatasetsViewProps {
  selectedId: string | null
  onSelect: (datasetId: string | null) => void
}

export function DatasetsView({ selectedId, onSelect }: DatasetsViewProps) {
  const [importing, setImporting] = useState(false)
  const [importFailure, setImportFailure] = useState<ApiFailure | null>(null)
  const [imported, setImported] = useState<DatasetSnapshot | null>(null)
  const fileInput = useRef<HTMLInputElement>(null)

  const datasets = useRequest(listDatasets, [imported])
  const card = useRequest(selectedId ? () => getDataset(selectedId) : null, [selectedId])
  const preview = useRequest(selectedId ? () => previewDataset(selectedId, 0, 25) : null, [selectedId])

  async function handleImport(file: File) {
    // двойной клик не должен отправить два импорта: кнопка выключена всё время полёта
    setImporting(true)
    setImportFailure(null)

    try {
      const response = await importDataset(file)
      setImported(response.dataset)
      onSelect(response.dataset.dataset_id)
    } catch (error) {
      setImportFailure(
        error instanceof ApiFailure
          ? error
          : new ApiFailure('frontend_error', 'unexpected', String(error)),
      )
    } finally {
      setImporting(false)

      if (fileInput.current) {
        // сброс позволяет выбрать тот же файл повторно после исправления
        fileInput.current.value = ''
      }
    }
  }

  return (
    <>
      <section className="panel">
        <h2>Импорт датасета</h2>
        <p className="hint">
          Dataset Package из DataArena (<code>.dapkg.zip</code>) или таблица напрямую: parquet, csv,
          json, jsonl, xlsx. ModelArena читает данные, но не редактирует их — подготовка в DataArena.
        </p>

        <div className="row">
          <input
            ref={fileInput}
            type="file"
            aria-label="Файл датасета"
            disabled={importing}
            onChange={(event) => {
              const file = event.target.files?.[0]

              if (file) {
                void handleImport(file)
              }
            }}
          />
          {importing ? <span className="muted">Импорт идёт, страница ждёт ответа сервера…</span> : null}
        </div>

        {importFailure ? <FailureNotice failure={importFailure} /> : null}

        {imported ? (
          <Notice
            level="info"
            title={`Импортирован «${imported.name}»: ${imported.row_count} строк, ${imported.column_count} колонок`}
            why={`Отпечаток ${imported.fingerprint.slice(0, 16)}… — идентичность данных, а не имени файла. Повторный импорт тех же данных вернёт этот же снимок.`}
            action="Перейдите к постановке задачи или выберите другой датасет."
          />
        ) : null}

        {imported?.warnings.map((warning) => (
          <Notice
            key={warning}
            title={warning}
            action="Учитывайте это при интерпретации результатов; изменить данные можно только в DataArena."
          />
        ))}
      </section>

      <section className="panel">
        <h2>Снимки</h2>
        <p className="hint">Снимок неизменяем: на него ссылаются эксперименты.</p>

        {datasets.failure ? <FailureNotice failure={datasets.failure} onRetry={datasets.reload} /> : null}
        {datasets.loading ? <p className="muted">Загрузка списка…</p> : null}

        {datasets.data && datasets.data.total === 0 ? (
          <p className="muted">Пока ничего не импортировано.</p>
        ) : null}

        <div className="grid">
          {datasets.data?.datasets.map((snapshot) => (
            <button
              key={snapshot.dataset_id}
              type="button"
              className="card"
              aria-pressed={snapshot.dataset_id === selectedId}
              onClick={() => onSelect(snapshot.dataset_id)}
            >
              <div className="name">{snapshot.name}</div>
              <div className="meta">
                {snapshot.row_count} строк · {snapshot.column_count} колонок
              </div>
              <div className="meta">
                <span className={`tag ${snapshot.source === 'package' ? 'accent' : ''}`}>
                  {snapshot.source === 'package' ? 'Dataset Package' : 'файл напрямую'}
                </span>
              </div>
            </button>
          ))}
        </div>
      </section>

      {selectedId ? (
        <section className="panel">
          <h2>Профиль</h2>
          <p className="hint">
            Посчитан ModelArena по самим данным. Статистика из пакета, если она есть, показывается
            как подсказка и на решения не влияет.
          </p>

          {card.failure ? <FailureNotice failure={card.failure} onRetry={card.reload} /> : null}
          {card.loading ? <p className="muted">Считаем профиль…</p> : null}

          {card.data ? (
            <>
              <div className="row muted" style={{ marginBottom: 10 }}>
                <span>Строк: {card.data.profile.row_count}</span>
                <span>Колонок: {card.data.profile.column_count}</span>
                <span>Полных дублей строк: {card.data.profile.duplicate_row_count}</span>
                <span>Пропущенных ячеек: {card.data.profile.missing_cell_count}</span>
              </div>

              <div className="scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Колонка</th>
                      <th>Тип</th>
                      <th>Пропуски</th>
                      <th>Различных</th>
                      <th>Замечания</th>
                    </tr>
                  </thead>
                  <tbody>
                    {card.data.profile.columns.map((column) => (
                      <tr key={column.name}>
                        <td>{column.name}</td>
                        <td className="muted">{column.logical_type}</td>
                        <td>{(column.missing_ratio * 100).toFixed(1)}%</td>
                        <td>{column.unique_count}</td>
                        <td className="muted">
                          {column.is_probable_id ? <span className="tag">идентификатор</span> : null}
                          {column.is_constant ? <span className="tag">константа</span> : null}
                          {column.is_near_constant && !column.is_constant ? (
                            <span className="tag">почти константа</span>
                          ) : null}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {card.data.profile.duplicate_row_count > 0 ? (
                <Notice
                  title={`Полных дублей строк: ${card.data.profile.duplicate_row_count}`}
                  why="Одинаковые строки, попав одновременно в обучение и в проверку, превращают измерение обобщения в измерение памяти — метрика окажется завышенной."
                  action="Удалите дубли в DataArena либо выберите разбиение по сущности, если повторы осмысленны."
                />
              ) : null}
            </>
          ) : null}
        </section>
      ) : null}

      {selectedId ? (
        <section className="panel">
          <h2>Строки</h2>
          <p className="hint">По сети уходит только видимая страница, а не весь датасет.</p>

          {preview.failure ? <FailureNotice failure={preview.failure} onRetry={preview.reload} /> : null}

          {preview.data ? (
            <div className="scroll">
              <table>
                <thead>
                  <tr>
                    {preview.data.columns.map((column) => (
                      <th key={column}>{column}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {preview.data.rows.map((row, index) => (
                    <tr key={index}>
                      {preview.data?.columns.map((column) => (
                        <td key={column}>{formatCell(row[column])}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
        </section>
      ) : null}
    </>
  )
}

function formatCell(value: unknown): string {
  if (value === null || value === undefined) {
    // пропуск и пустая строка — разные вещи, и выглядеть они обязаны по-разному
    return '—'
  }

  const text = String(value)
  return text.length > 120 ? `${text.slice(0, 120)}…` : text
}
