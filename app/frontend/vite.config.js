import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// `npm run dev` proxies /api to the Flask server (python app/server.py on port 5000).
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { '/api': 'http://127.0.0.1:5000' } },
  test: { environment: 'jsdom', globals: true, setupFiles: './src/setupTests.js', css: false },
});
