/**
 * Транспортный слой.
 *
 * Единственная его задача сверх передачи данных — честно назвать причину неудачи (D-18).
 * Формально «backend остановлен» и «backend вернул ошибку» приходят в браузер одинаково:
 * ответом с кодом 500. Различить их можно только тем, что наш backend **всегда** отдаёт
 * непустое тело в формате контракта ошибок. Пустое тело с кодом 500 — это прокси Vite,
 * который не смог достучаться до сервиса.
 *
 * Показывать пользователю «Backend вернул пустой ответ (500)» в этой ситуации — значит
 * отправить его чинить данные вместо того, чтобы запустить сервис.
 */

export type FailureKind =
  | 'backend_unavailable'
  | 'timeout'
  | 'api_error'
  | 'frontend_error'

export interface FailureView {
  /** Что обнаружено. */
  title: string
  /** Почему это важно и что за этим стоит. */
  explanation: string
  /** Что можно сделать. Пустая строка запрещена: см. D-19. */
  action: string
}

export class ApiFailure extends Error {
  readonly kind: FailureKind
  readonly code: string
  readonly status: number | null
  readonly details: Record<string, unknown>

  constructor(
    kind: FailureKind,
    code: string,
    message: string,
    options: { status?: number | null; details?: Record<string, unknown> } = {},
  ) {
    super(message)
    this.name = 'ApiFailure'
    this.kind = kind
    this.code = code
    this.status = options.status ?? null
    this.details = options.details ?? {}
  }

  /** Представление для интерфейса: что, почему и что делать (D-19). */
  view(backendPort: number): FailureView {
    switch (this.kind) {
      case 'backend_unavailable':
        return {
          title: 'ModelArena backend недоступен',
          explanation:
            'Интерфейс не смог соединиться с сервисом. С данными и с датасетом это не связано.',
          action: `Запустите backend на порту ${backendPort} и повторите запрос.`,
        }
      case 'timeout':
        return {
          title: 'Backend не ответил вовремя',
          explanation:
            'Запрос отменён по таймауту. Операция могла продолжить выполняться на сервере.',
          action: 'Повторите запрос. Если повторяется — проверьте нагрузку и логи backend.',
        }
      case 'frontend_error':
        return {
          title: 'Сбой на стороне интерфейса',
          explanation: this.message,
          action: 'Перезагрузите страницу. Если повторяется — это дефект интерфейса, а не данных.',
        }
      case 'api_error':
      default:
        return {
          title: this.message,
          explanation: `Код ошибки: ${this.code}.`,
          action: 'Исправьте входные данные или настройки запроса и повторите.',
        }
    }
  }
}

interface ErrorBody {
  error?: { code?: string; message?: string; details?: Record<string, unknown> }
  detail?: string
}

const DEFAULT_TIMEOUT_MS = 30_000

async function readBody(response: Response): Promise<{ text: string; parsed: unknown }> {
  const text = await response.text()

  if (text.trim() === '') {
    return { text, parsed: null }
  }

  try {
    return { text, parsed: JSON.parse(text) as unknown }
  } catch {
    return { text, parsed: null }
  }
}

function failureFromResponse(response: Response, parsed: unknown, text: string): ApiFailure {
  // наш backend всегда отвечает объектом контракта ошибок; всё остальное на этом пути
  // пришло не от него
  if (parsed && typeof parsed === 'object' && 'error' in parsed) {
    const body = parsed as ErrorBody
    return new ApiFailure(
      'api_error',
      body.error?.code ?? 'unknown_error',
      body.error?.message ?? body.detail ?? 'Запрос отклонён backend.',
      { status: response.status, details: body.error?.details },
    )
  }

  if (text.trim() === '') {
    // ровно тот случай: прокси не достучался до сервиса и отдал пустой ответ
    return new ApiFailure(
      'backend_unavailable',
      'backend_unavailable',
      'Пустой ответ без тела ошибки — сервис не отвечает.',
      { status: response.status },
    )
  }

  return new ApiFailure(
    'frontend_error',
    'unexpected_response',
    'Backend вернул ответ, не соответствующий контракту API.',
    { status: response.status },
  )
}

function buildHeaders(body: BodyInit | null | undefined): HeadersInit {
  // FormData сама проставляет multipart-границу: свой Content-Type здесь ломает разбор.
  // Для строкового тела Content-Type обязателен — без него FastAPI не считает его JSON
  // и отвечает ошибкой валидации, хотя тело корректное.
  if (body instanceof FormData) {
    return { Accept: 'application/json' }
  }

  if (typeof body === 'string') {
    return { Accept: 'application/json', 'Content-Type': 'application/json' }
  }

  return { Accept: 'application/json' }
}

export interface RequestOptions {
  method?: string
  body?: BodyInit | null
  timeoutMs?: number
  signal?: AbortSignal
}

export async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), options.timeoutMs ?? DEFAULT_TIMEOUT_MS)

  if (options.signal) {
    // сигнал, отменённый ДО вызова, не породит события abort: без явной проверки
    // запрос ушёл бы на сервер, а его результат перезаписал бы более свежее состояние
    if (options.signal.aborted) {
      clearTimeout(timeout)
      throw new ApiFailure('frontend_error', 'cancelled', 'Запрос отменён до отправки.')
    }

    options.signal.addEventListener('abort', () => controller.abort(), { once: true })
  }

  let response: Response

  try {
    response = await fetch(path, {
      method: options.method ?? 'GET',
      body: options.body ?? null,
      headers: buildHeaders(options.body),
      signal: controller.signal,
    })
  } catch (error) {
    // fetch отклоняет промис двумя способами: отмена по таймауту и сетевой сбой.
    // Второй означает, что до сервиса не дошли вообще, а не что он что-то ответил
    if (controller.signal.aborted) {
      throw new ApiFailure('timeout', 'timeout', 'Backend не ответил за отведённое время.')
    }

    throw new ApiFailure(
      'backend_unavailable',
      'backend_unavailable',
      error instanceof Error ? error.message : 'Соединение не установлено.',
    )
  } finally {
    clearTimeout(timeout)
  }

  const { text, parsed } = await readBody(response)

  if (!response.ok) {
    throw failureFromResponse(response, parsed, text)
  }

  if (parsed === null) {
    throw new ApiFailure(
      'frontend_error',
      'unexpected_response',
      'Успешный ответ не содержит корректного JSON.',
      { status: response.status },
    )
  }

  return parsed as T
}
