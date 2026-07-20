import { defineConfig } from "vite";

// Dev server proxies /api to the FastAPI backend (uvicorn api:app --port 8000,
// see README.md) so the browser talks to a single origin and CORS never comes
// up in local dev.
export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
