/**
 * Транспортный слой обязан назвать причину неудачи, а не показать «500».
 *
 * Главный случай (D-18): остановленный backend приходит через прокси Vite пустым ответом
 * с кодом 500 — формально это валидный ответ, и от ошибки API его отличает только то,
 * что наш backend всегда отдаёт непустое тело контракта ошибок.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ApiFailure, request } from './client'

const PORT = 8520

function respond(status: number, body: string, ok = false) {
  return vi.fn().mockResolvedValue({
    ok,
    status,
    text: () => Promise.resolve(body),
  } as unknown as Response)
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('различение причин неудачи', () => {
  it('пустой 500 читается как недоступность backend, а не как ошибка данных', async () => {
    vi.stubGlobal('fetch', respond(500, ''))

    const failure = await request('/api/datasets').catch((error: unknown) => error)

    expect(failure).toBeInstanceOf(ApiFailure)
    const view = (failure as ApiFailure).view(PORT)
    expect((failure as ApiFailure).kind).toBe('backend_unavailable')
    expect(view.title).toContain('недоступен')
    // пользователя нельзя отправлять чинить датасет, когда не запущен сервис
    expect(view.action).toContain(String(PORT))
    expect(view.title).not.toContain('500')
  })

  it('сетевой сбой читается как недоступность backend', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))

    const failure = (await request('/api/datasets').catch((error: unknown) => error)) as ApiFailure

    expect(failure.kind).toBe('backend_unavailable')
  })

  it('структурированная ошибка API сохраняет код и сообщение', async () => {
    vi.stubGlobal(
      'fetch',
      respond(
        400,
        JSON.stringify({
          error: { code: 'package_unsafe_path', message: 'Путь выходит за пределы пакета', details: {} },
          detail: 'Путь выходит за пределы пакета',
        }),
      ),
    )

    const failure = (await request('/api/datasets').catch((error: unknown) => error)) as ApiFailure

    expect(failure.kind).toBe('api_error')
    expect(failure.code).toBe('package_unsafe_path')
    expect(failure.view(PORT).title).toBe('Путь выходит за пределы пакета')
  })

  it('ответ не по контракту — это сбой интерфейса, а не ошибка данных', async () => {
    vi.stubGlobal('fetch', respond(502, '<html>Bad Gateway</html>'))

    const failure = (await request('/api/datasets').catch((error: unknown) => error)) as ApiFailure

    expect(failure.kind).toBe('frontend_error')
  })

  it('таймаут отличается от недоступности', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation(
        (_url: string, init: RequestInit) =>
          new Promise((_resolve, reject) => {
            init.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')))
          }),
      ),
    )

    const failure = (await request('/api/datasets', { timeoutMs: 5 }).catch(
      (error: unknown) => error,
    )) as ApiFailure

    expect(failure.kind).toBe('timeout')
    expect(failure.view(PORT).action).toContain('Повторите')
  })

  it('уже отменённый сигнал не даёт уйти запросу', async () => {
    // иначе результат отменённого запроса вернулся бы и перезаписал более свежее состояние
    const fetchSpy = respond(200, '{}', true)
    vi.stubGlobal('fetch', fetchSpy)
    const controller = new AbortController()
    controller.abort()

    const failure = (await request('/api/datasets', { signal: controller.signal }).catch(
      (error: unknown) => error,
    )) as ApiFailure

    expect(failure.code).toBe('cancelled')
    expect(fetchSpy).not.toHaveBeenCalled()
  })
})

describe('каждая причина объясняет, что делать', () => {
  it.each(['backend_unavailable', 'timeout', 'api_error', 'frontend_error'] as const)(
    '%s несёт непустое действие',
    (kind) => {
      const view = new ApiFailure(kind, 'code', 'сообщение').view(PORT)

      // плашка без ответа «что делать» обучает игнорировать все плашки подряд (D-19)
      expect(view.title.length).toBeGreaterThan(0)
      expect(view.explanation.length).toBeGreaterThan(0)
      expect(view.action.length).toBeGreaterThan(0)
    },
  )
})

describe('заголовки запроса', () => {
  it('строковое тело уходит с Content-Type: application/json', async () => {
    // без этого заголовка FastAPI не считает тело за JSON и отвечает ошибкой валидации,
    // хотя тело корректное. Юнит-тесты с моком fetch этого не ловили — нашла ручная проверка
    const fetchSpy = respond(200, '{}', true)
    vi.stubGlobal('fetch', fetchSpy)

    await request('/api/x', { method: 'POST', body: JSON.stringify({ a: 1 }) })

    const init = fetchSpy.mock.calls[0]?.[1] as RequestInit
    expect((init.headers as Record<string, string>)['Content-Type']).toBe('application/json')
  })

  it('у FormData свой Content-Type не ставится: он ломает multipart-границу', async () => {
    const fetchSpy = respond(200, '{}', true)
    vi.stubGlobal('fetch', fetchSpy)

    await request('/api/x', { method: 'POST', body: new FormData() })

    const init = fetchSpy.mock.calls[0]?.[1] as RequestInit
    expect((init.headers as Record<string, string>)['Content-Type']).toBeUndefined()
  })
})
