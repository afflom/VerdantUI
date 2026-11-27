# VerdantUI Conformance & Standards Register

> Generated mechanically by `npm run studio:publish` from `model/` (Rule R1). Do not edit by hand.

- **Service Charter:** Free Public Benefit AI Service (Apache-2.0, 100% client-side, zero servers, zero tracking, zero paywalls)
- **Compute Substrate:** `afflom/verdant-engine@v2.22.4` (`9d22f5ac364d`)
- **Lifecycle Authority:** `afflom/verdant-studio@2df2c6b035af`
- **Performance Budgets:** Turn OPFS stall `<= 0 ms`, Frame yield KPI `<= 16666 us`, JS bundle `<= 3600 KiB`

## Locked Architectural & Performance Invariants

| ID | Owner | Evidence | Statement |
| --- | --- | --- | --- |
| `VUI-CLIENT-01` | `scripts/ratchet-gate.ts` | `model/verdant-ratchet.toml` | 100% client-side runtime with zero external server or backend API dependencies |
| `VUI-ENGINE-01` | `src/engine/vtensorProvider.ts` | `model/engine-pin.json` | Exclusive afflom/verdant-engine vTensor WASM SIMD substrate (zero WebGPU/foreign processor APIs) |
| `VUI-MT-01` | `src/engine/vtensorThreadPool.ts` | `public/coi-serviceworker.js` | Multi-threaded SharedArrayBuffer WASM SIMD pool with COOP/COEP isolation, watchdog failover, and saturation-safe worker sizing |
| `VUI-OPFS-01` | `src/engine/vramBackingStore.ts` | `src/engine/vramBackingStore.test.ts` | Lock-free read-only OPFS model reads, UKV3 integrity-checked non-blocking KV persistence, pagehide teardown, and BroadcastChannel lease handover |
| `VUI-MOE-01` | `src/engine/llamaRuntime.ts` | `src/engine/scalingBenchmarks.test.ts` | Single-barrier Expert-Parallel Sparse MoE (OP_MOE_EXPERTS & hybrid OP_MOE_EXPERTS_BATCH4) with Theorem T3-MoE wave sizing |
| `VUI-SIMD-01` | `src/engine/verdantWasmSimd.ts` | `src/engine/verdantWasmSimd.test.ts` | Zero-spill XMM-domain scale shuffle and narrow-lane integer contractions across Q4_K, Q6_K, Q5_K, Q3_K, Q2_K, and Q4_0 |
| `VUI-PREFETCH-01` | `src/engine/llamaRuntime.ts` | `src/engine/vramBackingStore.ts` | Double-buffered async OPFS layer and LM-head stripe prefetch pipelining with in-flight slab deduplication |
| `VUI-PAGES-01` | `scripts/assemble-pages.ts` | `.github/workflows/pages.yml` | Subpath-safe GitHub Pages static bundle with commit-pinned fast CD workflow and enforced performance budgets |
| `VUI-DEDUP-01` | `src/engine/vramBootCache.ts` | `src/engine/vramBootCache.test.ts` | OPFS-to-runtime deduplication: cross-ID/ETag model reuse, .part slab salvage, content-fingerprinted KV/embedding/header/RoPE boot caches, and conflict-free MoE waves |
| `VUI-OBSERVER-01` | `src/engine/cognitiveObserver.ts` | `src/engine/cognitiveObserver.test.ts` | Unified VerdantUI Cognitive Observer: 8D Q10 harmonic belief state, Predictive Coding surprise, Active Inference Expected Free Energy + Prism CostSurface routing, FK-07 capability lattice, and lossless episodic continuity |
| `VUI-STUDIO-01` | `src/engine/prismStudio.ts` | `src/engine/prismStudio.test.ts` | Verdant Studio & Prism (.holo) Runtime: canonical SHA-256 packaging, JSON Schema 2020-12 Prism tool contracts, IndexedDB store, and FK-07 sandboxed Prism app execution |
| `VUI-STUDIO-02` | `src/engine/verdantStudioApi.ts` | `src/engine/verdantStudioApi.test.ts` | Declarative Verdant Studio API (afflom/verdant-studio parity): closed API operations with mandatory Req/Std/Der citations, monotone register admission, ISO 42010 traceable views, derived agenda, and self-healing repair loop |
| `VUI-STUDIO-03` | `src/engine/prismCompiler.ts` | `src/engine/prismCompiler.test.ts` | Mechanical Prism Compiler & Browser-Native prism/holo/6 WASM Verification: deterministic LexLean 1.2 + prism.design/1 compilation, OPFS BLAKE3 toolchain caching, Lean 4.32.2 WASM kernel proofs, and portable Core-Wasm guest execution |
| `VUI-LIFECYCLE-01` | `scripts/studio-lifecycle.ts` | `src/engine/studioLifecycle.test.ts` | Verdant Studio Product & Feature Lifecycle Governance: authoritative model/ registers, closed studio:* CLI, automated upstream engine/studio tracking, and deterministic semantic releases |
| `VUI-PUBLIC-BENEFIT-01` | `scripts/ratchet-gate.ts` | `model/performance-budgets.json` | Free Public Benefit Service Charter: Apache-2.0 open service with zero tracking/analytics SDKs, zero paywalls, zero server telemetry, and enforced ISO-25010 performance budgets |
| `VUI-UX-JARGON-01` | `scripts/ratchet-gate.ts` | `model/verdant-ratchet.toml` | Plain-Language User Surface Invariant: zero exposure of internal Verdant constructs (Observer, OpenTelemetry, vTensor, LexLean, Holon, FK-07, Q10, REQ-D*) in user-facing UI surfaces |

## Pinned Authorities

| Authority | Edition | Scope |
| --- | --- | --- |
| `VERDANT-ENGINE` | v2.22.4 | Exclusive browser WASM SIMD compute substrate, multi-core SharedArrayBuffer pool, and bounded memory governors |
| `VERDANT-STUDIO` | main@da5083b5ee22 | Closed-API product/feature lifecycle, well-founded Req/Std/Der admission, monotone locks, and automated upstream tracking |
| `PRISM-CODE` | 0.11.5 | Browser-native LexLean 1.2 compilation, Lean 4.32.2 kernel proofs, prism.design/1 compilation, and prism/holo/6 archives |
| `AFFLOM-ME` | main | Owner-centric possession hierarchy, kill switch, regulated detector hashonly/deny boundary, and seen-bytes outbound approval |
| `APACHE-2.0-PUBLIC-BENEFIT` | 2.0 | Open, free-to-use public benefit AI service with zero paywalls, zero ad/tracking SDKs, and 100% browser-local execution |
| `ISO-25010` | 2023 | Frame budget KPI (<= 16,666 us), zero turn-completion storage stalls, bounded memory, and multi-core throughput scaling |
| `ISO-42010` | 2022 | Stakeholder-traceable viewpoints and deterministic separation of model authority from rendered view |
| `OWASP-ASVS-5.0.0` | 5.0.0 | Strict CSP sandboxing, zero raw cross-context leakage, FK-07 denial-wins grants, and integrity-verified local persistence |
| `WCAG-2.2-AA` | 2.2 Level AA | Accessible public-benefit interface with plain-language copy and zero internal engineering jargon |
| `W3C-PROV-O` | REC-20130430 | Stakeholder producer/consumer provenance and content-addressed artifact lineage |
| `JSON-SCHEMA-2020-12` | 2020-12 | Typed declarative weight schemas and Prism tool stream contracts |
| `DORA-METRICS` | 2024 | Fast automated CI/CD (~60s GitHub Pages publish), automated upstream upgrade PRs, and deterministic semantic releases |
