import { defineConfig } from "vite";

export default defineConfig({
  root: ".",
  publicDir: "public",
  // GitHub Pages serves project sites from a /<repo-name>/ subpath; local
  // dev and any future root-domain host should stay at "/". Set via the
  // VITE_BASE_PATH env var in CI rather than hardcoding the repo name here.
  base: process.env.VITE_BASE_PATH || "/",
  build: {
    outDir: "dist",
  },
});
