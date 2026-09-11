import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// backend слушает 8520; порт вынесен в одно место, чтобы сообщение о недоступности
// и настройка прокси не разъезжались
export const BACKEND_PORT = 8520

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: `http://127.0.0.1:${BACKEND_PORT}`,
        changeOrigin: true,
      },
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    css: false,
  },
})
