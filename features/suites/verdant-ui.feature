Feature: verdant-ui

  VerdantUI conformance scenarios mapping every registered ID in model/ids.toml
  to its honesty level (`build` proved by Lean's kernel, or `open` verified by host gates).

  @VUI-HOLO-01 @build
  Scenario: VerdantUI compiles as a verified zero-capability Core-Wasm holo archive
    Given the LexLean model in app/src/App.lex.tex
    When prism-code check has Lean's kernel prove App.answers_with_the_request
    Then prism-code build emits a verified prism/holo/6 archive and replays its acceptance vectors

  @VUI-CLIENT-01 @open
  Scenario: 100% client-side runtime with zero external server or backend dependencies
    Given the VerdantUI source tree and package manifest
    When npm run ratchet executes all planted-defect checks
    Then zero server dependencies and zero server files exist

  @VUI-ENGINE-01 @open
  Scenario: Exclusive afflom/verdant-engine vTensor WASM SIMD substrate
    Given src/engine/vtensorProvider.ts and model/engine-pin.json
    When npm run ratchet and npm test execute
    Then zero WebGPU or foreign processor APIs exist and default GGML backends are refused

  @VUI-MT-01 @open
  Scenario: Multi-threaded SharedArrayBuffer WASM SIMD pool with COOP/COEP isolation
    Given public/coi-serviceworker.js and src/engine/vtensorThreadPool.ts
    When npm run verify:pages and npm run test:bench execute
    Then COOP/COEP headers and multi-core worker dispatch are verified

  @VUI-OPFS-01 @open
  Scenario: Lock-free OPFS model reads and integrity-checked non-blocking KV persistence
    Given src/engine/vramBackingStore.ts
    When npm test runs src/engine/vramBackingStore.test.ts
    Then OPFS read-only access, UKV3 integrity, and pagehide teardown pass

  @VUI-MOE-01 @open
  Scenario: Single-barrier Expert-Parallel Sparse MoE wave sizing
    Given src/engine/llamaRuntime.ts
    When npm run test:bench executes
    Then Theorem T3-MoE wave sizing and expert dispatch invariants pass

  @VUI-SIMD-01 @open
  Scenario: Zero-spill XMM-domain scale shuffle and narrow-lane integer contractions
    Given src/engine/verdantWasmSimd.ts
    When npm test runs src/engine/verdantWasmSimd.test.ts
    Then quantized dot-product contractions match reference outputs

  @VUI-PREFETCH-01 @open
  Scenario: Double-buffered async OPFS layer and LM-head stripe prefetch pipelining
    Given src/engine/llamaRuntime.ts and src/engine/vramBackingStore.ts
    When npm test runs engine runtime tests
    Then slab prefetch deduplication prevents duplicate OPFS fetches

  @VUI-PAGES-01 @open
  Scenario: Subpath-safe GitHub Pages static bundle within performance budgets
    Given scripts/assemble-pages.ts and scripts/verify-pages.ts
    When npm run build:pages executes
    Then all asset references are subpath-safe and JS chunks stay within budget

  @VUI-DEDUP-01 @open
  Scenario: OPFS-to-runtime deduplication and content-fingerprinted boot caches
    Given src/engine/vramBootCache.ts
    When npm test runs src/engine/vramBootCache.test.ts
    Then model reuse, .part salvage, and boot cache fingerprinting pass

  @VUI-OBSERVER-01 @open
  Scenario: Unified VerdantUI Cognitive Observer belief state and FK-07 capability lattice
    Given src/engine/cognitiveObserver.ts
    When npm test runs src/engine/cognitiveObserver.test.ts
    Then harmonic belief updates, surprise routing, and FK-07 denials pass

  @VUI-STUDIO-01 @open
  Scenario: Verdant Studio & Prism runtime canonical packaging and sandboxed execution
    Given src/engine/prismStudio.ts
    When npm test runs src/engine/prismStudio.test.ts
    Then canonical SHA-256 packaging and FK-07 sandboxed dispatch pass

  @VUI-STUDIO-02 @open
  Scenario: Declarative Verdant Studio API with closed operations and well-founded citations
    Given src/engine/verdantStudioApi.ts
    When npm test runs src/engine/verdantStudioApi.test.ts
    Then Req/Std/Der admission, ISO 42010 views, and self-healing repair pass

  @VUI-STUDIO-03 @open
  Scenario: Mechanical Prism Compiler and browser-native prism/holo/6 WASM verification
    Given src/engine/prismCompiler.ts
    When npm test runs src/engine/prismCompiler.test.ts
    Then deterministic LexLean compilation and Core-Wasm execution pass

  @VUI-LIFECYCLE-01 @open
  Scenario: Verdant Studio product and feature lifecycle governance
    Given scripts/studio-lifecycle.ts and model/ registers
    When npm run studio:check executes
    Then monotone locks, well-founded citations, and generated docs verify cleanly

  @VUI-PUBLIC-BENEFIT-01 @open
  Scenario: Free Public Benefit Service Charter with zero tracking or paywalls
    Given index.html, package.json, and model/performance-budgets.json
    When npm run ratchet executes
    Then zero tracking SDKs, zero paywall dependencies, and zero budget regressions exist

  @VUI-UX-JARGON-01 @open
  Scenario: Plain-language user surface with zero internal engineering jargon
    Given user-facing TSX components under src/
    When npm run ratchet executes
    Then zero internal Verdant terms leak into user-facing copy
