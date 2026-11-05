import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The API server runs on :8000; the dev server forwards /api to it.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { '/api': 'http://localhost:8000' },
  },
})
