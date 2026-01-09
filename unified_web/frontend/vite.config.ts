import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import fs from 'fs'

// 读取VERSION文件
const versionFile = path.resolve(__dirname, '../VERSION')
const version = fs.existsSync(versionFile)
  ? fs.readFileSync(versionFile, 'utf-8').trim()
  : '0.0.0'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  // 从父目录 (unified_web) 加载 .env 文件
  const env = loadEnv(mode, path.resolve(__dirname, '..'), '')
  const apiPort = env.VITE_API_PORT || '8002'
  const apiTarget = `http://localhost:${apiPort}`

  return {
    plugins: [react()],
    define: {
      __APP_VERSION__: JSON.stringify(version),
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    server: {
      port: 3002,
      proxy: {
        '/api/simulation/ws': {
          target: apiTarget,
          ws: true,
          changeOrigin: true,
        },
        '/api': {
          target: apiTarget,
          changeOrigin: true,
        },
        '/ws': {
          target: apiTarget,
          ws: true,
          changeOrigin: true,
        },
      },
    },
    build: {
      outDir: 'dist',
      sourcemap: true,
      rollupOptions: {
        output: {
          manualChunks: {
            'vendor-react': ['react', 'react-dom', 'react-router-dom'],
            'vendor-ui': ['antd', '@ant-design/icons'],
            'vendor-chart': ['echarts', 'echarts-for-react'],
            'vendor-graph': ['cytoscape', 'react-cytoscapejs'],
            'vendor-table': ['handsontable', '@handsontable/react'],
          },
        },
      },
    },
  }
})
