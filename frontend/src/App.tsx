import { useState } from 'react'

import { ArenaView } from './features/arena/ArenaView'
import { DatasetsView } from './features/datasets/DatasetsView'
import { TaskSetupView } from './features/task-setup/TaskSetupView'

type Section = 'datasets' | 'task' | 'arena'

export function App() {
  const [section, setSection] = useState<Section>('datasets')
  const [datasetId, setDatasetId] = useState<string | null>(null)
  //прогон запускается по разобранной постановке, поэтому цель живёт здесь,
  //а не внутри раздела: иначе разделы разошлись бы в том, что именно обучается
  const [targetColumn, setTargetColumn] = useState<string | null>(null)

  return (
    <div className="app">
      <header className="app-header">
        <h1 className="app-title">
          Model<span>Arena</span>
        </h1>
        <span className="muted">честное сравнение моделей на подготовленном датасете</span>

        <nav className="nav">
          <button
            type="button"
            aria-current={section === 'datasets' ? 'page' : undefined}
            onClick={() => setSection('datasets')}
          >
            Датасеты
          </button>
          <button
            type="button"
            aria-current={section === 'task' ? 'page' : undefined}
            onClick={() => setSection('task')}
          >
            Постановка задачи
          </button>
          <button
            type="button"
            aria-current={section === 'arena' ? 'page' : undefined}
            onClick={() => setSection('arena')}
          >
            Прогон
          </button>
        </nav>
      </header>

      <main>
        {section === 'datasets' ? (
          <DatasetsView
            selectedId={datasetId}
            onSelect={(id) => {
              setDatasetId(id)
              setTargetColumn(null)
              setSection('task')
            }}
          />
        ) : section === 'task' ? (
          <TaskSetupView datasetId={datasetId} onAnalyzed={setTargetColumn} />
        ) : (
          <ArenaView datasetId={datasetId} targetColumn={targetColumn} />
        )}
      </main>
    </div>
  )
}
