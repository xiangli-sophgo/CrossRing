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
    host: '127.0.0.1', // 避免 Mac 上 localhost IPv6 解析延迟
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8004', // 使用 127.0.0.1 替代 localhost
        changeOrigin: true,
      },
    },
    watch: {
      // Mac 上使用 polling 模式可能更稳定，但通常不需要
      // usePolling: true,
      // 忽略不需要监听的目录
      ignored: ['**/node_modules/**', '**/.git/**'],
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor-react': ['react', 'react-dom'],
          'vendor-three': ['three', '@react-three/fiber', '@react-three/drei', '@react-spring/three'],
          'vendor-ui': ['antd', '@ant-design/icons'],
          'vendor-chart': ['echarts', 'echarts-for-react'],
        },
      },
    },
  },
  // 优化依赖预构建
  optimizeDeps: {
    include: [
      'three',
      '@react-three/fiber',
      '@react-three/drei',
      '@react-spring/three',
      'react',
      'react-dom',
      'antd',
    ],
  },
})
