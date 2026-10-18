import { loadEnv } from "vite";
import { defineConfig } from "vitest/config";

export default defineConfig(({ mode }) => {
  // loadEnv reads .env, .env.local and .env.<mode>[.local] from the project
  // root. An empty prefix keeps the non-VITE_ variables visible too.
  const env = loadEnv(mode, process.cwd(), "");
  const proxyTarget = env.VITE_API_PROXY_TARGET || "http://127.0.0.1:8000";

  return {
    server: {
      host: "0.0.0.0",
      port: 5173,
      strictPort: true,
      proxy: {
        "/api": {
          target: proxyTarget,
          changeOrigin: true,
        },
      },
    },
    preview: {
      host: "0.0.0.0",
      port: 5173,
      strictPort: true,
    },
    test: {
      environment: "node",
      include: ["tests/**/*.test.ts"],
      restoreMocks: true,
    },
  };
});
