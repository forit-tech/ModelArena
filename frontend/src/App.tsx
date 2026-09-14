/**
 * Оболочка приложения: разделы и то, что между ними передаётся.
 *
 * Порядок разделов повторяет реальный путь работы: данные → постановка → прогон →
 * применение, плюс история экспериментов сбоку. Выбранные датасет, цель, прогон и участник
 * живут здесь, а не внутри разделов: иначе разделы разошлись бы в том, о чём именно речь,
 * и «применить модель» относилось бы не к тому прогону, который открыт рядом.
 */
import { useState } from 'react'

import { ArenaView } from './features/arena/ArenaView'
import { DatasetsView } from './features/datasets/DatasetsView'
import { ExperimentsView } from './features/experiments/ExperimentsView'
import { TaskSetupView } from './features/task-setup/TaskSetupView'
import { TestDriveView } from './features/testdrive/TestDriveView'

type Section = 'datasets' | 'task' | 'arena' | 'experiments' | 'testdrive'

const SECTIONS: { key: Section; label: string }[] = [
  { key: 'datasets', label: 'Датасеты' },
  { key: 'task', label: 'Постановка задачи' },
  { key: 'arena', label: 'Прогон' },
  { key: 'experiments', label: 'Эксперименты' },
  { key: 'testdrive', label: 'Применение' },
]

export function App() {
  const [section, setSection] = useState<Section>('datasets')
  const [datasetId, setDatasetId] = useState<string | null>(null)
  //прогон запускается по разобранной постановке, поэтому цель живёт здесь
  const [targetColumn, setTargetColumn] = useState<string | null>(null)
  const [runId, setRunId] = useState<string | null>(null)
  const [contenderKey, setContenderKey] = useState<string | null>(null)

  return (
    <div className="app">
      <header className="app-header">
        <h1 className="app-title">
          Model<span>Arena</span>
        </h1>
        <span className="muted">честное сравнение моделей на подготовленном датасете</span>

        <nav className="nav">
          {SECTIONS.map((item) => (
            <button
              key={item.key}
              type="button"
              aria-current={section === item.key ? 'page' : undefined}
              onClick={() => setSection(item.key)}
            >
              {item.label}
            </button>
          ))}
        </nav>
      </header>

      <main>
        {section === 'datasets' ? (
          <DatasetsView
            selectedId={datasetId}
            onSelect={(id) => {
              setDatasetId(id)
              setTargetColumn(null)
              setRunId(null)
              setContenderKey(null)
              setSection('task')
            }}
          />
        ) : null}

        {section === 'task' ? (
          <TaskSetupView datasetId={datasetId} onAnalyzed={setTargetColumn} />
        ) : null}

        {section === 'arena' ? (
          <ArenaView
            datasetId={datasetId}
            targetColumn={targetColumn}
            externalRunId={runId}
            onRunChanged={setRunId}
            onUseModel={(key) => {
              setContenderKey(key)
              setSection('testdrive')
            }}
          />
        ) : null}

        {section === 'experiments' ? (
          <ExperimentsView
            onOpenRun={(id) => {
              setRunId(id)
              setSection('arena')
            }}
          />
        ) : null}

        {section === 'testdrive' ? (
          <TestDriveView runId={runId} contenderKey={contenderKey} />
        ) : null}
      </main>
    </div>
  )
}
