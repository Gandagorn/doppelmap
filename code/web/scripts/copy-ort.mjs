/** Put the onnxruntime WebAssembly runtime where the browser can fetch it.
 *
 *  The runtime is 13MB and belongs to the deployed site, not to the
 *  repository, so it is copied out of node_modules at build time into
 *  public/ort/ (which .gitignore already covers) rather than committed.
 *  Self-hosted rather than pulled from a CDN: the site asks a third party
 *  for nothing, and the runtime cannot change underneath a pinned model.
 *
 *  Only the plain wasm build is copied. The jsep/webgpu variants are
 *  another 40MB and would buy nothing here, and real threading needs
 *  cross-origin isolation headers that GitHub Pages does not send.
 */
import { copyFileSync, mkdirSync, statSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const from = dirname(require.resolve("onnxruntime-web"));
const to = join(dirname(fileURLToPath(import.meta.url)), "..", "public", "ort");

mkdirSync(to, { recursive: true });
let total = 0;
// ort.wasm.min.mjs is the loader; the other two are the runtime it fetches.
// The ".bundle." variants are deliberately avoided: they reference the wasm
// through import.meta.url, so a bundler emits a second 14MB copy of it.
for (const name of ["ort.wasm.min.mjs",
                    "ort-wasm-simd-threaded.wasm",
                    "ort-wasm-simd-threaded.mjs"]) {
  copyFileSync(join(from, name), join(to, name));
  total += statSync(join(to, name)).size;
}
console.log(`copy-ort: ${(total / 1048576).toFixed(1)} MB -> public/ort/`);
