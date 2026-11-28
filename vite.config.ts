import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import fs from 'fs';
import path from 'path';
import {defineConfig, type Plugin} from 'vite';

/** BLAKE3-pinned WASM toolchain blobs copied from node_modules/prism-code (lazy-loaded in Studio only). */
const PRISM_PINNED_BLOBS = [
  '4f/302c18e02a46b0d015c57bd389a0754b301ec876aa7a637f786c4562274ecb', // lexlean.wasm.gz
  'ec/13d669b62c9a0223a937e04c3df18235fa67878ca53ba926f75904fe9c4610', // lean-kernel.mjs
  '2f/254140590074724ca1a8033080e0ae0834a018e06481ad7ccf7de9b80f175a', // lean-kernel.wasm.gz
  'f5/502e3d8e4fd563f8bd352b427883149abd5dca0e6c47002fe686a5ddeae995', // base.ndjson.gz
] as const;

function buildTemplateFoundationJson(prismCodeRoot: string): string {
  const canonicalDir = path.join(prismCodeRoot, 'templates/canonical');
  const index = JSON.parse(fs.readFileSync(path.join(canonicalDir, 'index.json'), 'utf8')) as {
    schema: string;
    label: string;
    files: Record<string, string>;
  };
  const files: Record<string, string> = {};
  for (const [relPath, label] of Object.entries(index.files)) {
    if (!relPath.startsWith('app/') || relPath === 'app/src/App.lex.tex') continue;
    const objPath = path.join(canonicalDir, 'objects', label.replace(/^blake3:/, ''));
    files[relPath.slice(4)] = fs.readFileSync(objPath, 'utf8');
  }
  return JSON.stringify({schema: 'verdant-ui/prism-foundation/1', templateLabel: index.label, files});
}

function prismToolchainPlugin(rootDir: string): Plugin {
  const prismCodeRoot = path.resolve(rootDir, 'node_modules/prism-code');
  const evaluatorSrc = path.join(prismCodeRoot, 'src/journey/guest/evaluator.wasm');
  const pinnedRoot = path.join(prismCodeRoot, 'wasm/pinned');
  const verdantHoloSrc = path.resolve(rootDir, 'app/.prism/build/verdant-ui.holo');
  let cachedFoundation: string | null = null;

  return {
    name: 'verdant-prism-toolchain',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const cleanUrl = (req.url ?? '').split('?')[0];
        if (cleanUrl.endsWith('/holos/verdant-ui.holo') && fs.existsSync(verdantHoloSrc)) {
          res.setHeader('Content-Type', 'application/octet-stream');
          res.end(fs.readFileSync(verdantHoloSrc));
          return;
        }
        const marker = '/prism-toolchain/';
        const idx = cleanUrl.indexOf(marker);
        if (idx === -1) return next();
        const rel = cleanUrl.slice(idx + marker.length);
        if (rel === 'template-foundation.json') {
          cachedFoundation ??= buildTemplateFoundationJson(prismCodeRoot);
          res.setHeader('Content-Type', 'application/json; charset=utf-8');
          res.end(cachedFoundation);
          return;
        }
        if (rel === 'evaluator.wasm') {
          res.setHeader('Content-Type', 'application/wasm');
          res.end(fs.readFileSync(evaluatorSrc));
          return;
        }
        if ((PRISM_PINNED_BLOBS as readonly string[]).includes(rel)) {
          res.setHeader('Content-Type', 'application/octet-stream');
          res.end(fs.readFileSync(path.join(pinnedRoot, rel)));
          return;
        }
        return next();
      });
    },
    closeBundle(error?: Error): void {
      if (error) return;
      const distDir = path.resolve(rootDir, 'dist');
      if (!fs.existsSync(distDir) || !fs.existsSync(prismCodeRoot)) return;
      const outDir = path.join(distDir, 'prism-toolchain');
      fs.mkdirSync(outDir, {recursive: true});
      fs.copyFileSync(evaluatorSrc, path.join(outDir, 'evaluator.wasm'));
      for (const rel of PRISM_PINNED_BLOBS) {
        const dest = path.join(outDir, rel);
        fs.mkdirSync(path.dirname(dest), {recursive: true});
        fs.copyFileSync(path.join(pinnedRoot, rel), dest);
      }
      fs.writeFileSync(
        path.join(outDir, 'template-foundation.json'),
        buildTemplateFoundationJson(prismCodeRoot),
      );
      if (fs.existsSync(verdantHoloSrc)) {
        const holosDir = path.join(distDir, 'holos');
        fs.mkdirSync(holosDir, {recursive: true});
        fs.copyFileSync(verdantHoloSrc, path.join(holosDir, 'verdant-ui.holo'));
      }
    },
  };
}

export default defineConfig(() => {
  const prismSrc = path.resolve(import.meta.dirname, 'node_modules/prism-code/src');
  return {
    base: process.env.VITE_BASE_PATH || './',
    worker: {
      format: 'es' as const,
    },
    plugins: [
      react(),
      tailwindcss(),
      prismToolchainPlugin(import.meta.dirname),
      // Copies src/assets into dist/ so root-relative URLs like
      // "/src/assets/images/..." still resolve in production builds.
      {
        name: 'copy-unbundled-src-assets',
        apply: 'build',
        closeBundle(error?: Error): void {
          if (error) {
            return;
          }
          const distDir = path.resolve(import.meta.dirname, 'dist');
          const srcAssets = path.resolve(import.meta.dirname, 'src/assets');
          const distAssets = path.resolve(distDir, 'src/assets');
          if (fs.existsSync(distDir) && fs.existsSync(srcAssets)) {
            fs.cpSync(srcAssets, distAssets, {recursive: true, force: true});
          }
        },
      },
    ],
    build: {
      rolldownOptions: {
        output: {
          codeSplitting: {
            groups: [
              { name: 'vendor-react', test: /node_modules[\\/](react|react-dom|scheduler)[\\/]/ },
              { name: 'vendor-assistant-ui', test: /node_modules[\\/]@assistant-ui[\\/]/ },
              { name: 'vendor-markdown', test: /node_modules[\\/](remark-|rehype-|unified|mdast|micromark|hast|unist|vfile)/ },
              { name: 'vendor-prism', test: /node_modules[\\/]prism-code[\\/]/ },
              { name: 'verdant-studio-organs', test: /src[\\/]engine[\\/](prismMediaOrgans|prismMediaRuntimes|prismVisualRuntimes|prismAudioRuntimes|prismEditorialRuntimes|prismGameRuntimes|proceduralPictureRenderer|prismCompiler|verdantStudioApi)\.ts$/ },
            ],
          },
        },
      },
    },
    resolve: {
      alias: {
        '@': path.resolve(import.meta.dirname, '.'),
        '@prism-code-internal': prismSrc,
      },
    },
    server: {
      hmr: process.env.DISABLE_HMR !== 'true',
      watch: process.env.DISABLE_HMR === 'true' ? null : {},
      headers: {
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'credentialless',
      },
    },
    preview: {
      headers: {
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'credentialless',
      },
    },
  };
});
