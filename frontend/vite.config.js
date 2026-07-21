import { defineConfig } from "vite";

// Dev server proxies /api to the FastAPI backend (uvicorn api:app --port 8000,
// see README.md) so the browser talks to a single origin and CORS never comes
// up in local dev.
export default defineConfig({
  server: {
    port: 5173,
    // 127.0.0.1, not "localhost" -- on some systems (commonly Windows) Node
    // resolves "localhost" to the IPv6 loopback (::1) first, but
    // `uvicorn --host 0.0.0.0` only binds IPv4, so the proxy's connection
    // gets refused (ECONNREFUSED ::1:8000) even though the backend is
    // actually running and reachable on IPv4 localhost. Targeting the
    // literal IPv4 address sidesteps the resolution ambiguity entirely.
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/health": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
