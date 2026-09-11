/**
 * Загрузка данных с защитой от устаревшего ответа.
 *
 * Без счётчика поколений медленный ответ по первому датасету приходит после быстрого
 * по второму и затирает его: пользователь смотрит на карточку одного датасета,
 * а видит профиль другого. Это не гипотетика, а обычное поведение сети.
 */
import { useCallback, useEffect, useRef, useState } from 'react'

import { ApiFailure } from '../api/client'

interface RequestState<T> {
  data: T | null
  failure: ApiFailure | null
  loading: boolean
}

export function useRequest<T>(
  load: (() => Promise<T>) | null,
  deps: unknown[],
): RequestState<T> & { reload: () => void } {
  const [state, setState] = useState<RequestState<T>>({ data: null, failure: null, loading: false })
  const [attempt, setAttempt] = useState(0)
  const generation = useRef(0)

  const reload = useCallback(() => setAttempt((value) => value + 1), [])

  useEffect(() => {
    if (!load) {
      setState({ data: null, failure: null, loading: false })
      return
    }

    const current = ++generation.current
    setState((previous) => ({ ...previous, loading: true, failure: null }))

    load()
      .then((data) => {
        // ответ из устаревшего поколения не имеет права трогать состояние
        if (current === generation.current) {
          setState({ data, failure: null, loading: false })
        }
      })
      .catch((error: unknown) => {
        if (current !== generation.current) {
          return
        }

        setState({
          data: null,
          failure:
            error instanceof ApiFailure
              ? error
              : new ApiFailure('frontend_error', 'unexpected', String(error)),
          loading: false,
        })
      })

    return () => {
      // размонтирование тоже переводит поколение: результат уже никому не нужен
      generation.current += 1
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, attempt])

  return { ...state, reload }
}
