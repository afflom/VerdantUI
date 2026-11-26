# VerdantUI Owner Decisions

> Generated mechanically by `npm run studio:publish` from `model/owner-decisions.json`. Do not edit by hand.

## DEC-001 (2026-11-25)

- **Requirements:** `REQ-CLIENT-ONLY`
- **Decision:** "We will only ever support the client-side implementation of VerdantUI. There will never be a server outside of the VerdantUI client runtime."

## DEC-002 (2026-11-25)

- **Requirements:** `REQ-VERDANT-ENGINE-ONLY`, `REQ-NO-WEBGPU`
- **Decision:** "Verdant studio will never make use of WebGPU or any processor API outside of verdant-engine."

## DEC-003 (2026-11-25)

- **Requirements:** `REQ-PAGES-CD`, `REQ-FAST-PIPELINE`
- **Decision:** "We need each push to main to publish to github pages. We need to make sure that the repo stays optimized, so that the development pipeline is not slowed down by validation overhead."

## DEC-004 (2026-11-25)

- **Requirements:** `REQ-STUDIO-RATCHET`
- **Decision:** "We'll work towards the strict model of Verdant Studio using a version of the Verdant ratchet."

## DEC-005 (2026-11-25)

- **Requirements:** `REQ-MULTITHREAD-PERF`
- **Decision:** "We are dedicated to the highest possible performance; lowest latency, highest throughput implementation. The multi-threaded implementation is required. We are committed to robust and scalable solutions."

## DEC-006 (2026-11-25)

- **Requirements:** `REQ-STUDIO-HARNESS`
- **Decision:** "Verdant Studio will also be an agent harness in VerdantUI, so that Verdant Apps can be created, published, and managed in VerdantUI. We'll work on the Verdant Studio harness after we refine the VerdantUI implementation."

## DEC-007 (2026-11-26)

- **Requirements:** `REQ-FREE-PUBLIC-BENEFIT`
- **Decision:** "VerdantUI must be a best-in-class AI service that will provide a Free Public Benefit Service. The implementation of Verdant UI must be maintained to the highest standards including performance standards."

## DEC-008 (2026-11-26)

- **Requirements:** `REQ-STUDIO-LIFECYCLE`, `REQ-AUTO-RELEASE`
- **Decision:** "We must implement VerdantUI using Verdant Studio, so that we can maintain a healthy product lifecycle including the lifecycle of VerdantUI features, so that we can automate releases and upgrades and other aspects of software publishing."

## DEC-009 (2026-11-26)

- **Requirements:** `REQ-PROCEDURAL-PRISM-WEIGHTS`, `REQ-SPARSE-MEDIA-ACTIVATION`
- **Decision:** "The entire Verdant Prism must be procedurally synthesized from the Verdant Studio, so that VerdantUI is only used to route weights onto the Verdant Prism."

## DEC-010 (2026-11-26)

- **Requirements:** `REQ-POSSESSION-MODEL`, `REQ-NON-LEAKING-BOUNDARY`
- **Decision:** "We need to establish the user model within the VerdantUI unified model, so that the User can possess its parts of VerdantUI... The Observer also possesses the User in the same way... VerdantUI will route to external sources and must not leak any weights between contexts when it is sending information."

## DEC-011 (2026-11-26)

- **Requirements:** `REQ-ZERO-USER-JARGON`
- **Decision:** "We do not expose verdant constructs such as the Observer or other technical jargon to the user of VerdantUI."

## DEC-012 (2026-11-26)

- **Requirements:** `REQ-ZERO-USER-TUNABLES`, `REQ-COHERENT-MULTIMODAL-SENSES`
- **Decision:** "Take all jargon and bloat out of VerdantUI. Do not introduce a bunch of tunables to the user. Users are not going to be uploading their own weights or models. Coherent multi-modal models enabled by verdant-engine improve the performance of VerdantUI."

## DEC-013 (2026-11-26)

- **Requirements:** `REQ-AUTO-MEDIATYPE-INFERENCE`, `REQ-PRISM-BACKED-TOOL-CALLS`
- **Decision:** "Instead of exposing more clutter in the UI/UX such as the mediatype selection buttons, the model will select these for the user based on the user's natural language input. When a model attempts a tool-call, the tool called will be a prism. If no prism exists to meet the model's requirement, the model can create a prism, so that it can use its Prism to perform the task of its tool-call."
