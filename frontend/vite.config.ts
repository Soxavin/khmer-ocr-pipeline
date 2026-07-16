import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Served in production at /app from the pipeline process (webapp/api.py mounts
// frontend/dist). In dev, /api is proxied to the running pipeline on :8600.
export default defineConfig({
  base: '/app/',
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': 'http://localhost:8600',
    },
  },
})
