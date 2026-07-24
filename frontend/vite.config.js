import { defineConfig } from "vite";

// Dev-server proxy so the frontend can call same-origin /api and /health
// paths while the real FastAPI backend runs separately on :8000 — no CORS
// setup needed on the backend.
export default defineConfig({
  server: {
    proxy: {
      "/api": "http://localhost:8000",
      "/health": "http://localhost:8000",
    },
  },
});
