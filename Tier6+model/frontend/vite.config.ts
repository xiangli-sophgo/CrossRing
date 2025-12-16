import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'fs'
import path from 'path'

// 读取VERSION文件
const versionFile = path.resolve(__dirname, '../VERSION')
const version = fs.existsSync(versionFile)
  ? fs.readFileSync(versionFile, 'utf-8').trim()
  : '0.0.0'

export default defineConfig({
  plugins: [react()],
  define: {
    __APP_VERSION__: JSON.stringify(version),
  },
  server: {
    port: 3100,
    proxy: {
      '/api': {
        target: 'http://localhost:8003',
        changeOrigin: true,
      },
    },
  },
})
