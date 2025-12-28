import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import fs from 'fs'

// 读取VERSION文件
const versionFile = path.resolve(__dirname, '../VERSION')
const version = fs.existsSync(versionFile)
  ? fs.readFileSync(versionFile, 'utf-8').trim()
  : '0.0.0'

// https://vitejs.dev/config/
export default defineConfig({
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
    port: 3002,  // 使用新端口，避免与现有项目冲突
    proxy: {
      // WebSocket 代理必须放在 /api 之前，否则会被 /api 匹配
      '/api/simulation/ws': {
        target: 'http://localhost:8000',
        ws: true,
        changeOrigin: true,
      },
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: 'http://localhost:8000',
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
})
