# Undo/Redo + Import Sync Notes

When editor behavior changes in a way that affects runtime/editor state, import
logic must be updated to keep imported graphs behaviorally equivalent to
interactively-created graphs.

## Required checks after state-model changes

If you add or change any editor state used by interaction flows (for example:
history stacks, position caches, selection-derived transient data, per-node
runtime caches), update import logic accordingly.

Current required sync points live in `DpgNodeEditor._cntrl_import_setting_dict`
and must include:

1. Rebuild graph model/view objects (nodes, links).
2. Recompute/synchronize derived caches (`_cntrl_sync_position_cache`).
3. Reset transient interaction/history state (`_cntrl_reset_history_state`).

Without this, first interaction after import can diverge from normal behavior
(e.g., first drag undo mismatch).
