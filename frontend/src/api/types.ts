/**
 * Типы ответов backend.
 *
 * Пока написаны руками; по DESIGN §11 (риск T-10) их следует генерировать из OpenAPI,
 * чтобы они не разошлись со схемами. Генерация — отдельная задача этапа, здесь важно,
 * что типы соответствуют фактическим ответам, проверенным integration-тестами backend.
 */

export interface ColumnSpec {
  name: string
  dtype: string
  logical_type: string
  nullable: boolean
}

export interface DatasetSnapshot {
  dataset_id: string
  name: string
  source: 'package' | 'file'
  fingerprint: string
  fingerprint_algorithm: string
  row_count: number
  column_count: number
  columns: ColumnSpec[]
  created_at: string
  row_key: string[]
  package_ref: Record<string, unknown> | null
  warnings: string[]
}

export interface ColumnProfile {
  name: string
  dtype: string
  logical_type: string
  missing_count: number
  missing_ratio: number
  unique_count: number
  unique_ratio: number
  is_constant: boolean
  is_near_constant: boolean
  is_probable_id: boolean
  id_reason: string | null
  numeric_stats: Record<string, number> | null
  top_values: { value: unknown; count: number; ratio: number }[] | null
  examples: unknown[]
}

export interface DatasetProfile {
  row_count: number
  column_count: number
  duplicate_row_count: number
  missing_cell_count: number
  missing_cell_ratio: number
  columns: ColumnProfile[]
}

export interface ExclusionRecord {
  column: string
  reason: string
  source: string
}

export interface TaskProposal {
  target_column: string
  task_type: 'binary' | 'multiclass' | 'regression'
  class_labels: string[]
  positive_label: string | null
  feature_columns: string[]
  suggested_exclusions: ExclusionRecord[]
  time_column_candidates: string[]
  group_column_candidates: string[]
  row_count: number
  usable_row_count: number
  explanation: string
  warnings: string[]
}

export interface ProtocolProposal {
  cv_splitter: string
  n_splits: number
  holdout_strategy: string
  holdout_size: number
  shuffle: boolean
  seed: number
  explanation: string
  warnings: string[]
  feasibility_notes: string[]
}

export type ReadinessStatus = 'READY' | 'CAUTION' | 'HIGH_RISK' | 'BLOCKED'

export interface ReadinessFinding {
  code: string
  severity: 'info' | 'caution' | 'high_risk'
  scope: string
  observed_value: unknown
  threshold: string
  explanation: string
  consequence: string
  suggested_action: string
  blocking: boolean
}

export interface ReadinessReport {
  status: ReadinessStatus
  summary: string
  context: Record<string, number>
  findings: ReadinessFinding[]
}

export type LeakageEvidenceLevel = 'structural' | 'evidence' | 'heuristic'
export type LeakageRisk = 'none' | 'suspected' | 'high' | 'confirmed'

export interface LeakageSignal {
  code: string
  evidence_level: LeakageEvidenceLevel
  scope: string
  columns: string[]
  observed_value: unknown
  threshold: string
  explanation: string
  mechanism: string
  suggested_action: string
  blocking: boolean
}

export interface LeakageNotEvaluated {
  check: string
  scope: string
  reason: string
  consequence: string
}

export interface LeakageReport {
  risk: LeakageRisk
  summary: string
  signals: LeakageSignal[]
  /** Невыполненная проверка не равна чистому результату и обязана быть видимой в UI. */
  not_evaluated: LeakageNotEvaluated[]
  disclaimer: string
}

export interface AnalyzeResponse {
  task: TaskProposal
  protocol: ProtocolProposal
  readiness: ReadinessReport
  leakage: LeakageReport
}

export interface RunAnalysisResponse {
  run_id: string
  analysis: { readiness: ReadinessReport; leakage: LeakageReport } | null
  note: string
}

export interface DatasetCard {
  dataset: DatasetSnapshot
  profile: DatasetProfile
}

export interface PreviewPage {
  dataset_id: string
  columns: string[]
  rows: Record<string, unknown>[]
  offset: number
  limit: number
  total_rows: number
}

export type RunState =
  | 'PENDING'
  | 'RUNNING'
  | 'PARTIAL'
  | 'SUCCEEDED'
  | 'FAILED'
  | 'CANCELLED'
  | 'INTERRUPTED'

export type ContenderState =
  | 'PENDING'
  | 'RUNNING'
  | 'SUCCEEDED'
  | 'SKIPPED'
  | 'FAILED'
  | 'CANCELLED'

export interface MetricValue {
  key: string
  label: string
  direction: string
  /** null означает «посчитать не удалось» и сопровождается note; это не ноль */
  value: number | null
  std: number | null
  per_fold: (number | null)[]
  note: string
}

export interface MetricSet {
  split: string
  n_rows: number
  metrics: MetricValue[]
}

export interface ContenderRun {
  contender_key: string
  label: string
  family: string
  preprocessing_profile: string
  is_baseline: boolean
  state: ContenderState
  selection_reason: string
  error_code: string | null
  error_message: string
  folds_completed: number
  n_folds: number
  started_at: string | null
  finished_at: string | null
  elapsed_seconds: number | null
  artifact: {
    model_sha256: string
    model_bytes: number
    format_version: string
    training_rows: number
  } | null
  coverage: { covered_rows: number; expected_rows: number; complete: boolean; note: string } | null
  warnings: string[]
  metrics: MetricSet[]
}

export interface RunCard {
  run_id: string
  created_at: string
  started_at: string | null
  finished_at: string | null
  state: RunState
  cancel_requested: boolean
  error_code: string | null
  error_message: string
  notes: string[]
  spec: {
    dataset_id: string
    fold_assignment_hash: string
    n_splits: number
    holdout_rows: number
    train_pool_rows: number
    seed: number
    experiment_fingerprint: string
    budget: { max_parallel_fits: number; threads_per_fit: number; memory_budget_mb: number }
    protocol: ProtocolProposal
    task: { task_type: string; target_column: string; class_labels: string[] }
  }
  contenders: ContenderRun[]
}

export interface RunProgress {
  folds_completed: number
  folds_planned: number
  contenders_finished: number
  contenders_planned: number
  /** null, когда планировать нечего: ноль в знаменателе — не «0%» */
  completed_fraction: number | null
}

export interface RunEvent {
  kind: string
  at: string
  message: string
  contender_key: string | null
  fold: number | null
  n_folds: number | null
}

export interface RunView {
  run: RunCard
  progress: RunProgress
  active: boolean
}

export interface StartRunResponse {
  run: RunCard
  duplicate_of_active_run: boolean
}

export interface RunSummary {
  run_id: string
  created_at: string
  finished_at: string | null
  state: RunState
  error_code: string | null
  dataset_id: string
  task_type: string
  target_column: string
  experiment_fingerprint: string
  contenders: { planned: number; succeeded: number; failed: number; skipped: number }
}

export interface PairedComparison {
  leader: string
  trailing: string
  metric: string
  mean_difference: number
  std_difference: number
  folds_compared: number
  folds_won: number
  /** false означает «разница не держится на разбиении», а не «проиграл» */
  stable: boolean
  explanation: string
}

export interface ConstraintStatus {
  description: string
  metric: string
  observed: number | null
  /** null — проверить не удалось; это не то же самое, что «не прошёл» */
  satisfied: boolean | null
}

export interface LeaderboardRow {
  contender_key: string
  label: string
  family: string
  is_baseline: boolean
  /** null — участник в ранжировании не участвует, причина в reason */
  rank: number | null
  score: number | null
  std: number | null
  per_fold: (number | null)[]
  /** Априорная оценка до запуска, не измеренное время. */
  estimated_cost: number
  /** Фактическое wall-clock время завершённого участника. */
  elapsed_seconds: number | null
  /** Размер сохранённого модельного артефакта; null, если артефакт не создан. */
  model_bytes: number | null
  cross_validated: MetricValue[]
  holdout: MetricValue[]
  constraints: ConstraintStatus[]
  eligible: boolean
  reason: string
  versus_next: PairedComparison | null
  versus_baseline: PairedComparison | null
}

export interface ChampionVerdict {
  /** null означает «чемпион не выбран», и reason объясняет почему */
  contender_key: string | null
  label: string
  reason: string
  over_baseline: PairedComparison | null
  over_runner_up: PairedComparison | null
  decided_by_tie_breaker: string
}

export interface Leaderboard {
  objective: {
    metric: string
    task_type: string
    higher_is_better: boolean
    constraints: { description: string; metric: string }[]
    tie_breakers: { key: string; rule: string }[]
    min_gain_over_baseline: number
    description: string
  }
  rows: LeaderboardRow[]
  champion: ChampionVerdict
  baseline_key: string | null
  notes: string[]
}

export interface LeaderboardResponse {
  run_id: string
  run_state: RunState
  leaderboard: Leaderboard
}
