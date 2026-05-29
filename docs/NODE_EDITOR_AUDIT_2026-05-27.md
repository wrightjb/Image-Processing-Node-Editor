# Node Editor Remaining Cleanup Plan (2026-05-27)

This document tracks only the remaining node-editor cleanup work after the
editor/node hook, node-owned toolbar, history dispatch, keyboard shortcut, import
undo, and duplicate-destination import-link fixes landed.

## Scope

- `node_editor/node_editor.py`
- `node_editor/history.py`
- `node_editor/graph_runtime.py`
- `node_editor/runtime_controller.py`
- tests under `tests/unit/` and `tests/integration/`

---

## Remaining cleanup plan

### 1) Replace simplified test link tags with realistic port tags

- Location: `tests/unit/test_node_editor_import_export.py`, plus any other tests
  that use abbreviated link tags such as `1:test_node:out` / `2:test_node:in`.
- Current issue:
  - Some tests use simplified two-part-ish link suffixes that do not match the
    real node port shape used by the editor (`node_id:node_tag:type:InputXX` /
    `OutputXX`).
  - `_mdl_add_link()` currently has a compatibility relaxation so these test
    tags can still be added when port parsing fails.
- Why this is next:
  - The relaxation is intentionally narrow, but it is still hacky because
    production model validation is being influenced by test fixtures.
  - Cleaning up test data lets `_mdl_add_link()` return to strict port parsing
    and keeps import/link behavior easier to reason about.
- Suggested implementation:
  1. Update import/export tests to use realistic full port tags everywhere.
  2. Remove the fallback branch in `_mdl_add_link()` that accepts links when
     `_cntrl_parse_port_tag()` returns `None`.
  3. Add/adjust a focused test asserting malformed imported link tags are
     rejected cleanly and do not mutate `_node_link_list`, `_link_registry`, or
     `_link_by_dest_port`.

### 2) Refactor hovered-port discovery to use registered ports

- Location: `node_editor/node_editor.py`
  (`_cntrl_get_hovered_output_port_tag`, `_cntrl_get_hovered_input_port_tag`).
- Current issue:
  - Hover lookup synthesizes possible tags by scanning all nodes, fixed port
    indices, and hardcoded data type names.
  - This is brittle, imposes hidden limits, and duplicates information already
    represented by node/port registries.
- Suggested implementation:
  1. Register actual input/output ports as nodes create them, including
     direction, type, index, node ref, and DPG tag.
  2. Replace synthetic tag probing with iteration over registered compatible
     ports or a direct lookup path if DearPyGui exposes enough context.
  3. Keep fallback behavior only where needed for imported/test graphs.
  4. Add tests around link insertion, insert-node-between-links, and occupied
     input replacement.

### 3) Continue import/link invariant hardening

- Location: `node_editor/node_editor.py` (`_cntrl_import_setting_dict_body`,
  `_cntrl_add_or_replace_link_by_tags`, `_mdl_add_link`).
- Current issue:
  - Duplicate-destination imports now use last-link-wins behavior, but malformed
    imported files still need clear documented behavior.
- Suggested implementation:
  1. Define canonical import invariants:
     - one incoming link per input/destination port,
     - `_node_link_list`, `_link_registry`, and `_link_by_dest_port` stay in
       sync,
     - rejected malformed links do not create partial model/view state.
  2. Add tests for malformed source/destination tags, unknown imported node IDs,
     duplicate links, and duplicate destinations.
  3. Ensure import undo/redo preserves those invariants.

### 4) Tighten graph/history identity boundaries if new edge cases appear

- Location: `node_editor/node_editor.py`, `node_editor/history.py`.
- Current issue:
  - `_history_node_id_remap` is now scoped better, but node recreation during
    undo/redo/import remains a subtle area.
- Suggested implementation:
  1. Add tests when new edge cases are found around undo/redo after import,
     delete, redo-branch replacement, or node ID collision.
  2. Consider whether node identity/remap helpers should move into a smaller
     history/identity helper if the editor class grows more history-specific
     state.

### 5) Runtime reliability guardrails

- Location: `node_editor/runtime_controller.py`.
- Current issue:
  - The async worker catches exceptions, prints traceback, and continues
    immediately.
- Suggested implementation:
  1. Add minimal backoff/rate-limiting for repeated runtime exceptions.
  2. Consider a development fail-fast option for easier debugging.
  3. Keep async DearPyGui race guidance in `docs/async-dpg-race-guide.md` in sync
     with any behavior changes.

### 6) Optional structural cleanup of `DpgNodeEditor`

- Location: `node_editor/node_editor.py`.
- Current issue:
  - The class still owns model state, DearPyGui view mutation, controller
    callbacks, history policy, and import/export orchestration.
- Suggested implementation:
  - Do not split files just for MVC purity.
  - Extract only when there is an obvious seam, such as a dedicated history
    service, import service, or port registry helper with strong test coverage.
