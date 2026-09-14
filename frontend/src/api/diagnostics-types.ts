/**
 * Типы диагностики, артефактов и экспериментов.
 *
 * Вынесены отдельным файлом, потому что это отдельная вертикаль продукта: «что с моделью
 * происходит» и «что с ней делать дальше», а не «какую модель выбрать».
 */
import type { RunState } from './types'

// ---------------------------------------------------------------- диагностика

export interface DiagnosticFinding {
  code: string
  severity: 'high' | 'caution' | 'info'
  title: string
  /** измерения, на которых основан вывод: без них это мнение, а не находка */
  evidence: Record<string, unknown>
  explanation: string
  consequence: string
  suggestion: string
}

export interface DiagnosticFold {
  fold: number
  validation_rows: number
  validation_score: number | null
  probe_rows: number
  train_score: number | null
  /** null означает «разрыв не измерен», а не «разрыва нет» */
  gap: number | null
}

export interface PerClassRow {
  label: string
  precision: number
  recall: number
  f1: number
  support: number
}

export interface ClassificationDiagnostics {
  available: boolean
  reason?: string
  n_rows?: number
  labels?: string[]
  is_binary?: boolean
  positive_label?: string | null
  confusion?: {
    labels: string[]
    rows: number[][]
    top_confusions: { actual: string; predicted: string; count: number }[]
  }
  per_class?: PerClassRow[]
  confidence?: {
    available: boolean
    reason?: string
    bins?: {
      from: number
      to: number
      count: number
      mean_confidence: number
      accuracy: number
    }[]
    note?: string
    overconfident_bins?: number
  }
}

export interface RegressionDiagnostics {
  available: boolean
  reason?: string
  n_rows?: number
  residuals?: {
    mean: number
    std: number
    median: number
    min: number
    max: number
    systematic_bias: boolean
    bias_note: string
    target_range: { min: number; max: number }
  }
  by_target_range?: {
    from: number
    to: number
    count: number
    mae: number
    rmse: number
    mean_residual: number
  }[]
  largest_errors?: {
    row_id: number
    fold: number
    y_true: number
    y_pred: number
    residual: number
    absolute_error: number
    relative_error: number | null
  }[]
}

export interface Diagnostics {
  contender_key: string
  task_type: string
  metric: string
  metric_label: string
  higher_is_better: boolean
  folds: DiagnosticFold[]
  mean_validation: number | null
  spread_validation: number | null
  mean_gap: number | null
  findings: DiagnosticFinding[]
  thresholds: Record<string, { caution?: number; high?: number; value?: number; meaning: string }>
  classification: ClassificationDiagnostics | null
  regression: RegressionDiagnostics | null
  notes: string[]
}

export interface ErrorRow {
  row_id: number
  fold: number
  y_true: string | number
  y_pred: string | number
  /** только колонки, которые модель действительно видела */
  features: Record<string, unknown>
  /** остальное из снимка, включая целевую колонку */
  context: Record<string, unknown>
  probabilities?: Record<string, number>
  absolute_error?: number
  residual?: number
}

export interface ErrorSelection {
  kind: string
  total_matching: number
  shown: number
  limit: number
  note: string
  rows: ErrorRow[]
}

export interface ImportanceRow {
  feature: string
  mean_drop: number
  std_drop: number
  per_repeat: number[]
}

export interface Importance {
  available: boolean
  reason?: string
  metric_label?: string
  measured_on?: string
  n_rows?: number
  repeats?: number
  rows?: ImportanceRow[]
  correlated_groups?: { features: string[]; correlation: number }[]
  interpretation?: string
  caveats?: string[]
}

export interface LearningCurve {
  available: boolean
  metric_label: string
  higher_is_better: boolean
  cost: {
    fits: number
    folds: number
    points: number
    within_limit: boolean
    limit: number
    note: string
  }
  points: {
    fraction: number
    train_rows: number
    folds_measured: number
    mean: number
    spread: number | null
    per_fold: number[]
  }[]
  skipped: string[]
  interpretation: { verdict: string; text: string }
}

// ---------------------------------------------------------------- артефакты

export interface ModelManifest {
  format: string
  format_version: string
  created_at: string
  run_id: string
  contender_key: string
  label: string
  preprocessing_profile: string
  task: {
    task_type: string
    target_column: string
    class_labels: string[]
    positive_label: string | null
    feature_columns: string[]
  }
  dataset: { dataset_id: string; fingerprint: string }
  protocol: Record<string, unknown>
  feature_schema: {
    columns: string[]
    dtypes: Record<string, string>
    numeric: string[]
    categorical: string[]
    datetime: string[]
  }
  training_rows: number
  environment: Record<string, string>
  model_sha256: string
  model_bytes: number
}

export interface ModelCard {
  run_id: string
  contender_key: string
  manifest: ModelManifest
  environment_warnings: string[]
  trust_boundary: string
}

export interface PredictionResult {
  task_type: string
  n_rows: number
  schema: { ok: boolean; missing: string[]; unexpected: string[]; notes: string[] }
  class_labels: string[]
  has_probabilities: boolean
  predictions: {
    index: number
    prediction: string | number
    probabilities?: Record<string, number>
  }[]
}

// ---------------------------------------------------------------- эксперименты

export interface ExperimentCard {
  run_id: string
  created_at: string
  finished_at: string | null
  state: RunState
  error_code: string | null
  runtime_seconds: number | null
  dataset: { dataset_id: string; fingerprint: string }
  task: {
    task_type: string
    target_column: string
    n_features: number
    feature_columns: string[]
    positive_label: string | null
    class_labels: string[]
  }
  protocol: {
    cv_splitter: string
    n_splits: number
    holdout_rows: number
    train_pool_rows: number
    seed: number
    fold_assignment_hash: string
  }
  experiment_fingerprint: string
  contenders: {
    planned: number
    succeeded: number
    failed: number
    skipped: number
    cancelled: number
    keys: string[]
  }
  champion: {
    objective: string
    objective_description: string
    contender_key: string | null
    label: string
    reason: string
    score: number | null
  } | null
  artifacts: { contender_key: string; model_sha256: string; model_bytes: number }[]
}

export interface ExperimentComparison {
  runs: ExperimentCard[]
  differences: {
    key: string
    label: string
    values: { run_id: string; value: unknown }[]
    consequence: string
  }[]
  comparable: boolean
  verdict: string
}
