/**
 * Конфигурация ESLint.
 *
 * Скрипт `npm run lint` был объявлен в package.json, но конфигурации рядом не лежало,
 * то есть проверка не выполнялась ни разу и молча «проходила». Объявленная, но
 * не работающая проверка хуже отсутствующей: на неё ссылаются как на пройденный этап.
 */
import js from '@eslint/js'
import reactHooks from 'eslint-plugin-react-hooks'
import globals from 'globals'
import typescript from 'typescript-eslint'

export default typescript.config(
  { ignores: ['dist', 'node_modules', 'coverage'] },
  js.configs.recommended,
  ...typescript.configs.recommended,
  {
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      ecmaVersion: 2022,
      globals: { ...globals.browser, ...globals.es2021 },
    },
    plugins: { 'react-hooks': reactHooks },
    rules: {
      ...reactHooks.configs.recommended.rules,
      //неиспользуемое имя, начинающееся с подчёркивания, — намеренно проигнорированный аргумент
      '@typescript-eslint/no-unused-vars': [
        'error',
        { argsIgnorePattern: '^_', varsIgnorePattern: '^_' },
      ],
    },
  },
  {
    files: ['**/*.test.{ts,tsx}', 'src/test/**'],
    languageOptions: { globals: { ...globals.browser, ...globals.node } },
  },
)
