# Node Editor Audit (2026-05-27)

This document summarizes a targeted audit of `node_editor/` with emphasis on undo/redo integration, node-specific logic leakage, and structural maintainability.

## Scope

- `node_editor/node_editor.py`
- `node_editor/history.py`
- `node_editor/graph_runtime.py`
- `node_editor/runtime_controller.py`

---

## Corrections from follow-up review

The initial draft contained two incorrect findings that were based on a misread and are now removed:

- `ReplaceLinkCommand.redo()` is **not** called twice in current `_cntrl_redo`.
- `_cntrl_close_insert_link_popup_on_escape` does **not** have a duplicate `def` line in current `node_editor.py`.

---

## High-priority issues

### 1) Strong node-specific behavior in controller (`Cache`, `ResultImage`, `ResultImageLarge`)

- Location: `node_editor/node_editor.py` (`_cntrl_apply_parameter_side_effects`, `_cntrl_toggle_result_node`)
- Detail:
  - Controller checks hard-coded port names (`Cache`, `ResultImage`, `ResultImageLarge`).
  - It directly calls private hooks like `_on_cache_toggle` on node instances.
  - Result-node wiring and placement is handled in controller with special-case logic.
- Risk:
  - violates encapsulation and extensibility,
  - new node types require controller edits,
  - increases coupling between generic editor and specific node semantics.
- Suggested fix:
  - define a common, editor-facing contract in a base used by all nodes (candidate: `node/node_abc.py`, since not all nodes use the declarative base),
  - move editor-triggered side-effect behavior behind that contract.

### 2) Undo/redo command dispatch uses long `isinstance` chains

- Location: `node_editor/node_editor.py` (`_cntrl_undo`, `_cntrl_redo`)
- Detail: command objects already define `undo()`/`redo()`, but controller still manually dispatches each known command type.
- Risk:
  - unnecessary duplication,
  - adding new command classes requires touching dispatch code,
  - easier to introduce regressions during edits.
- Suggested fix: call `cmd.undo(self)` / `cmd.redo(self)` directly and use duck-typing/Protocol-style expectations rather than manual type branching.

---

## Medium-priority issues

### 3) Hovered port discovery is brute-force and type-hardcoded

- Location: `node_editor/node_editor.py` (`_cntrl_get_hovered_output_port_tag`, `_cntrl_get_hovered_input_port_tag`)
- Detail: loops over all nodes, fixed index range `0..99`, and hardcoded data type names including both `TimeMs` and `TimeMS`.
- Risk:
  - brittle and expensive on large graphs,
  - hidden constraints (max 100 ports) and naming inconsistency,
  - future port types require controller edits.
- Clarification on fix direction:
  - a registry can reduce nested work by indexing actual created ports + metadata, instead of synthesizing possible tags,
  - if DPG context/sender offers the hovered attribute directly, use that first and fall back to registry lookup.

### 4) Import link handling needs explicit validation and clearer invariants

- Location: `node_editor/node_editor.py` (`_cntrl_import_setting_dict_body`)
- Detail: import should not partially preserve model-rejected links in `_node_link_list`.
- Current assessment:
  - multiple links to the same destination/input are malformed for this editor,
  - if an imported file contains that case, import should use the same last-link-wins replacement behavior as interactive linking.
- Suggested next step:
  - add focused tests for normal import and duplicate-destination import,
  - keep `_node_link_list` and link registries consistent for every imported link that is accepted.

### 5) History remap (`_history_node_id_remap`) lifecycle is under-specified

- Location: `node_editor/node_editor.py`
- Detail: remap table grows during history re-creation but reset/cleanup boundaries are unclear.
- Risk:
  - stale mappings can make history behavior hard to reason about in long sessions,
  - harder debugging for undo/redo edge cases.
- Suggested fix:
  - explicitly scope and reset remap state during graph reset/import/history invalidation,
  - evaluate whether node identity concerns belong in shared node/editor contract (possibly in `node_abc` integration points).

---

## Low-priority / organization observations

### 6) `DpgNodeEditor` is still a “god class” despite internal `_mdl/_vw/_cntrl` naming split

- Location: `node_editor/node_editor.py`
- Detail: one class still owns graph state, DPG manipulation, history policy, and import/export orchestration.
- Impact:
  - high cognitive load,
  - difficult isolated testing,
  - regressions likely when touching unrelated concerns.
- Practical direction:
  - keep file cohesion for now if preferred,
  - but define clearer section-level boundaries and interfaces first, then split only where pain is highest.

### 7) Runtime loop has unbounded exception retry behavior

- Location: `node_editor/runtime_controller.py`
- Detail: async worker catches exceptions, prints traceback, and immediately continues forever.
- Impact:
  - persistent faults can loop noisily and consume CPU,
  - production debugging becomes harder.
- Suggested fix: add minimal backoff/rate-limiting and optional fail-fast mode for development.

---

## Revised phased cleanup plan

1. **Next step (execute first): Node-side contract for editor-triggered behavior**
   - Define a minimal, common interface in a base all nodes can use (candidate: `node/node_abc.py`).
   - Initial scope:
     - editor-to-node side effects for parameter toggles,
     - optional hooks for result-node behavior,
     - stable identity/metadata helpers where needed.
   - Why first:
     - this removes the highest-coupling logic from controller,
     - unlocks cleaner follow-up refactors without requiring all nodes to migrate to declarative base.

2. **History simplification pass**
   - Replace manual `isinstance` dispatch in undo/redo with direct command invocation.
   - Document and enforce lifecycle rules for `_history_node_id_remap`.

3. **Hover/port lookup pass**
   - Introduce registry-based port lookup and eliminate hardcoded type/index probing.
   - Prefer direct DPG context information when available; fallback to registry.

4. **Import behavior hardening pass**
   - Add tests to validate duplicate-destination import behavior and define canonical invariants.
   - Keep `_node_link_list` and link registries consistent; do not partially preserve rejected links.

5. **Reliability/structure pass**
   - Add runtime exception backoff/fail-fast options.
   - Incrementally formalize internal module boundaries (without mandatory full file split).

---

## Plan update after Phase 1 execution

Phase 1 has been implemented with a minimal cross-node contract:

- Added `on_editor_parameter_value_applied(value_tag, value)` hook to `DpgNodeABC`.
- Updated `node_editor` parameter side-effect path to call that hook instead of hard-coded/private node methods.
- Implemented the hook in `DeclarativeImageProcessNodeBase` for `Cache`, `ResultImage`, and `ResultImageLarge` behavior.

### Impact

- Removes controller dependence on node-private methods (`_on_*_toggle`) for editor-applied values.
- Establishes a common extension point usable by both declarative and non-declarative nodes.
- Keeps migration scope incremental: existing nodes can adopt hook behavior as needed without broad rewrites.

## Plan update after toolbar ownership follow-up

The toolbar follow-up keeps the practical compromise from Phase 1 while making
ownership clearer:

- `DpgNodeABC` now owns a small reusable toolbar helper for node-built editor
  chrome.
- Declarative image process nodes call that helper so their top row contains
  the universal delete button plus image-specific `R`, `RL`, and `Cache`
  controls on a single ASCII-safe row.
- The editor no longer provides a fallback delete button. Nodes that want the
  toolbar button should call the shared helper; non-migrated nodes can still be
  removed with existing editor deletion commands such as the Delete key.

### Next execution step

The next cleanup step should be the **History simplification pass** from the
revised plan: remove manual command-type dispatch from undo/redo and make command
objects responsible for their own `undo()`/`redo()` behavior. That is now the
best next target because the highest-value node-specific UI coupling has been
reduced, and history dispatch remains a localized structural issue with clear
unit-test coverage.

## Plan update after history dispatch simplification

The **History simplification pass** has started with the most localized cleanup:

- `_cntrl_undo()` and `_cntrl_redo()` now dispatch directly to command
  `undo(editor)` / `redo(editor)` methods instead of manually branching over
  every known command class.
- Parameter-history suspension is centralized in one helper so future history
  commands automatically get the same guard behavior.
- A duck-typed test command verifies that history dispatch no longer depends on
  membership in the editor's hard-coded command list.

### Next execution step

Continue the history cleanup by tightening `_history_node_id_remap` lifecycle
rules. The next pass should document when remaps are created/cleared and add
focused tests around undo/redo after import, graph reset, and node recreation so
stale identity mappings cannot leak across unrelated history sessions.

## Plan update after keyboard shortcut normalization

Before leaving undo/redo, the keyboard shortcuts were normalized to common
application conventions:

- `Ctrl+Z` now triggers undo.
- `Ctrl+Shift+Z` now triggers redo.
- Bare `Z` key presses no longer invoke history actions.

## Plan update after import undo/history hardening

Import is now represented as an undoable history command instead of a history
clearing boundary:

- Successful imports push an `ImportGraphCommand` containing the imported node
  payloads and final accepted imported links.
- Undo removes the imported nodes; redo recreates them through the history
  rehydration path.
- Duplicate-destination links in an imported file use last-link-wins replacement
  behavior instead of partially preserving a rejected link in `_node_link_list`.
- Focused tests cover import undo/redo, duplicate-destination import behavior,
  stale remap cleanup when a new command replaces a fully undone redo branch.

### Next execution step

Move to the hover/port lookup pass: reduce ad-hoc port probing by relying more
on the existing node/port registries, with tests around link insertion and
occupied-input replacement.
