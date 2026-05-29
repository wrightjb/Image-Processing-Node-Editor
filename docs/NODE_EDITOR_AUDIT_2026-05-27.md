# Node Editor Remaining Cleanup Plan (2026-05-27)

This document tracks only the remaining node-editor cleanup work after the
editor/node hook, node-owned toolbar, history dispatch, keyboard shortcut, import
undo, and duplicate-destination import-link fixes landed.

For the larger typed-port model direction, see
`docs/TYPED_PORT_MODEL_REFACTOR_PLAN.md`.

## Scope

- `node_editor/node_editor.py`
- `node_editor/history.py`
- `node_editor/graph_runtime.py`
- `node_editor/runtime_controller.py`
- tests under `tests/unit/` and `tests/integration/`

---

## Remaining cleanup plan

### 1) Replace simplified test link tags with realistic port tags ✅ Done 2026-05-29

- Location: `tests/unit/test_node_editor_import_export.py`, plus any other tests
  that used abbreviated link tags such as `1:test_node:out` / `2:test_node:in`.
- Completed:
  1. Import/export tests now use realistic full port tags everywhere.
  2. `_mdl_add_link()` rejects malformed port tags strictly instead of accepting
     abbreviated compatibility fixtures.
  3. Malformed imported link tags are covered by a focused no-partial-link-state
     regression test.

### 2) Refactor hovered-port discovery to use registered ports

- Status: Fold into `docs/TYPED_PORT_MODEL_REFACTOR_PLAN.md` after
  `DpgNodeBase` can register typed ports and after
  `DeclarativeImageProcessNodeBase` is migrated.
- Location: `node_editor/node_editor.py`
  (`_cntrl_get_hovered_output_port_tag`, `_cntrl_get_hovered_input_port_tag`).
- Current issue:
  - Hover lookup synthesizes possible tags by scanning all nodes, fixed port
    indices, and hardcoded data type names.
  - This is brittle, imposes hidden limits, and duplicates information that
    should come from creation-time node/port registration.
- Sequencing note:
  - Do not do this as a standalone pre-refactor cleanup while `_port_registry`
    is still populated lazily by parsing tag strings. Use it as the first
    editor-side consumer of typed port registration instead.
- Suggested implementation:
  1. Register actual input/output ports as nodes create them, including
     direction, type, index, node ref, and DPG tag.
  2. Replace synthetic tag probing with iteration over registered compatible
     ports or a direct lookup path if DearPyGui exposes enough context.
  3. Keep fallback behavior only where needed for imported/test graphs.
  4. Keep characterization tests around add-node-from-hovered-output,
     add-node-to-hovered-input, insert-node-between-links, and occupied input
     replacement.

### 3) Continue import/link invariant hardening ✅ Done 2026-05-29

- Location: `node_editor/node_editor.py` (`_cntrl_import_setting_dict_body`,
  `_cntrl_add_or_replace_link_by_tags`, `_mdl_add_link`).
- Scope clarification:
  - This item is not general malformed-JSON/schema validation. It is limited to
    preserving link model/view invariants for imported link records that reach
    link handling.
- Completed:
  1. Link model validation now requires parseable `Output -> Input` endpoints
     with matching data types before mutating link state.
  2. Link view creation failures roll back model state so `_node_link_list`,
     `_link_registry`, `_link_by_dest_port`, and `_link_view_id_map` do not
     diverge.
  3. Duplicate-destination replacement is atomic: a failed replacement does not
     remove an existing accepted link, and import undo/redo payloads preserve the
     final accepted graph.
  4. Tests cover malformed tags, semantically invalid links, parseable but
     missing DPG ports, duplicate-destination replacement failure, and valid
     last-link-wins behavior.

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
