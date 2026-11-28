Feature: verdant-ui-prism

  Lean-kernel-proved state machines, grants, compartments, parts, observer purity,
  GUI parity, WCAG 2.2 accessibility, and cell tuning compiled into verdant-ui.holo.

  @VUI-CONSENT-01 @build
  Scenario: Consent transitions unlock cleanly when model is loaded and gate pending visits
    Given VerdantUI.Consent in app/src/VerdantUI/Consent.lex.tex
    When prism-code check proves paused_to_accepted_never_locks and pending_requires_consent
    Then accepted or paused consent with a loaded model always yields ready/unlocked

  @VUI-CONSENT-REFUSE-01 @build @rejects
  Scenario: Unknown streaming consent state is refused by name
    Given VerdantUI.Consent
    When a request names an unknown consent state
    Then the holo refuses with "refused: unknown consent state"

  @VUI-GRANT-01 @build
  Scenario: Stateful organs admit storage grants and FK-07 denial always dominates
    Given VerdantUI.Grants in app/src/VerdantUI/Grants.lex.tex
    When prism-code check proves stateful_organ_admits_storage and denial_dominates_grant
    Then stateful organs receive storage even offline unless explicitly denied

  @VUI-GRANT-REFUSE-01 @build @rejects
  Scenario: Explicit FK-07 denial refuses capability access by name
    Given VerdantUI.Grants
    When a capability is explicitly denied
    Then the holo refuses with "refused: denied by FK-07"

  @VUI-GRANT-OFFLINE-01 @build @rejects
  Scenario: Offline mode explicitly refuses network capability requests
    Given VerdantUI.Grants
    When offline mode is enabled and an organ requests network access
    Then the holo refuses with "refused: offline mode denies network"

  @VUI-COMPARTMENT-01 @build
  Scenario: Personal Prisms stay visible in personal space while private work Prisms stay hidden
    Given VerdantUI.Compartment in app/src/VerdantUI/Compartment.lex.tex
    When prism-code check proves default_personal_visible and private_work_hidden_in_personal
    Then personal creations appear in personal space and private work creations stay hidden

  @VUI-COMPARTMENT-REFUSE-01 @build @rejects
  Scenario: Unknown compartment names are refused by name
    Given VerdantUI.Compartment
    When a compartment visibility check names an unknown compartment
    Then the holo refuses with "refused: unknown compartment"

  @VUI-PARTS-01 @build
  Scenario: Every registered VerdantUI MediaType organ and Space answers status dispatch
    Given VerdantUI.Parts in app/src/VerdantUI/Parts.lex.tex
    When prism-code check proves all_registered_parts_answer_status
    Then status dispatch succeeds for every registered part

  @VUI-PARTS-REFUSE-01 @build @rejects
  Scenario: Unknown part identifiers are refused by name
    Given VerdantUI.Parts
    When a part dispatch names an unregistered part
    Then the holo refuses with "refused: unknown part"

  @VUI-PARTS-VERB-REFUSE-01 @build @rejects
  Scenario: Unknown part verbs are refused by name
    Given VerdantUI.Parts
    When a part dispatch uses an unrecognized verb
    Then the holo refuses with "refused: unknown part verb"

  @VUI-OBSERVER-PURE-01 @build
  Scenario: Observing never changes what a participant experiences
    Given VerdantUI.Observer in app/src/VerdantUI/Observer.lex.tex
    When prism-code check proves observing_never_changes_what_a_participant_experiences
    Then observed state equals participant state by reflexivity

  @VUI-GUI-PARITY-01 @build
  Scenario: An agent gets the exact view tree a human participant gets
    Given VerdantUI.Gui in app/src/VerdantUI/Gui.lex.tex
    When prism-code check proves an_agent_gets_the_tree_a_person_gets
    Then agent and person view trees are identical

  @VUI-A11Y-WCAG-01 @build
  Scenario: Every VerdantUI surface view passes the WCAG 2.2 AA profile
    Given VerdantUI.Accessibility in app/src/VerdantUI/Accessibility.lex.tex
    When prism-code check proves every_part_view_passes_wcag_2_2_audit
    Then all 6 primary surface views satisfy WCAG 2.2 AA audit criteria

  @VUI-CELLS-01 @build
  Scenario: Natural-language tune routes valid Prism cell targets
    Given VerdantUI.Cells in app/src/VerdantUI/Cells.lex.tex
    When prism-code check proves tune_routes_valid_cell
    Then non-empty cell targets route cleanly

  @VUI-CELLS-REFUSE-01 @build @rejects
  Scenario: Empty cell target is refused by name
    Given VerdantUI.Cells
    When a tune command supplies an empty cell target
    Then the holo refuses with "refused: empty cell target"

  @VUI-TELEMETRY-01 @build
  Scenario: Valid W3C traceparent and monotonic OTEL span timing are accepted
    Given VerdantUI.Telemetry in app/src/VerdantUI/Telemetry.lex.tex
    When prism-code check proves valid_w3c_traceparent_accepted and monotonic_span_timing_accepted
    Then well-formed traceparent headers and monotonic span timings succeed

  @VUI-TELEMETRY-VER-REFUSE-01 @build @rejects
  Scenario: W3C traceparent version ff is refused by name
    Given VerdantUI.Telemetry
    When a telemetry command supplies a version ff traceparent
    Then the holo refuses with "refused: traceparent version ff invalid"

  @VUI-TELEMETRY-ZERO-REFUSE-01 @build @rejects
  Scenario: All-zero W3C trace-id or parent-id is refused by name
    Given VerdantUI.Telemetry
    When a telemetry command supplies an all-zero trace-id or parent-id
    Then the holo refuses with "refused: traceparent zero trace or parent id"

  @VUI-TELEMETRY-TIMING-REFUSE-01 @build @rejects
  Scenario: Span ending before it starts is refused by name
    Given VerdantUI.Telemetry
    When a telemetry timing command supplies endMs less than startMs
    Then the holo refuses with "refused: span end before start"
