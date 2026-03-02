# Async DearPyGui Race Guide

This guide explains intermittent update-loop errors seen during delete/import stress (for example, deleting nodes while the async worker is running).

## What's really happening

The application runs two kinds of work concurrently:

1. **GUI callbacks** (add/delete/import, widget interaction)
2. **Async node updates** in `async_main` (`run_in_executor`)

When both touch DearPyGui state in the same time window, a "time-of-check vs time-of-use" race can happen:

- Item exists when checked
- Item is deleted by callback
- Async update calls `dpg.get_*` and receives `SystemError` / `RuntimeError`

This is why failures often appear around delete/import operations, but the root cause is **concurrent GUI state access**, not deletion alone.

## Best general-purpose solution

The most reliable architecture is:

1. **Single-thread ownership for DearPyGui calls**  
   Restrict raw DPG API calls to the GUI thread.
2. **Snapshot state for async updates**  
   Keep per-node settings/data snapshots updated by GUI callbacks; async update logic reads snapshots instead of DPG widgets.
3. **UI command application on GUI thread**  
   Async workers emit results/intents; GUI thread applies widget updates.
4. **Defensive fallback layer**  
   Keep guarded wrappers and node-level safe parsing as safety nets.

## Practical recommendations for this repository

### Keep (stability-critical)

- Top-level async loop exception guard in `async_main`
- Per-node async exception handling in `update_node_info`
- Deleted-node/stale-connection cleanup
- Guarded DPG helpers in `node_editor/util.py`
- Safe int parsing for node input values (`Blur`, `Resize`)

### Prefer for future implementation

- For code reachable from `Node.update()`, prefer `node_editor.util` guarded wrappers over direct `dpg.get_*` calls.
- Treat node UI reads as optional in async mode:
  - if read fails, use default/fallback values
  - continue processing rather than exiting
- Keep exception logs concise and actionable:
  - node id/name
  - exception type/message
  - traceback

### Debug workflow

1. Reproduce with async mode (default)
2. Confirm behavior in sync mode (`--unuse_async_draw`)
3. Compare outputs/logs to classify race vs logic bug
4. Harden specific node read paths where exceptions repeat

## Why not rely only on deferred deletion?

Mark-for-deletion can reduce one race window, but if async updates still call DPG APIs concurrently, races can still happen. Deferred deletion helps, but **thread ownership and snapshot-based updates** are the durable fix.
