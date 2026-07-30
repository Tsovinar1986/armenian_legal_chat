import { defineConfig } from "vite";

// Dev-server proxy so the frontend can call same-origin /api and /health
// paths while the real FastAPI backend runs separately on :8010 — no CORS
// setup needed on the backend. Not :8000: on this machine that port is
// already held by an unrelated project's dev server (a different repo's
// `uvicorn backend.main:app`), which coincidentally also answers /health,
// so pointing here at :8000 would silently proxy to the wrong app instead
// of failing loudly.
export default defineConfig({
  server: {
    port: 5171,
    // Without this, an already-occupied 5171 makes Vite silently fall back
    // to the next free port (5172, 5173, ...) instead of erroring -- easy
    // to miss and then wonder why localhost:5171 isn't loading anything.
    strictPort: true,
    proxy: {
      "/api": "http://localhost:8010",
      "/health": "http://localhost:8010",
    },
  },
});
