# Node base class refactor notes

This document is the canonical summary of the node base class/helper refactor.
It replaces previous phase-tracking analysis notes.

## Scope

- Standardize shared helper usage across nodes that directly inherit from `DpgNodeABC`.
- Mechanically standardize tag generation and link parsing.

## Shared helper APIs

The following helpers in `node/node_abc.py` are the source of truth:

- `_node_name(node_id)`
- `_port_tag(node_name, value_type, port_name)`
- `_value_tag(port_tag)`
- `_iter_connections(connection_list)`
- `_extract_source_node_key(source_tag)`
- `_extract_port_name(tag)`

## Phase status

- ✅ Phase 1 completed (8/8)
- ✅ Phase 2 completed (10/10)
- ✅ Phase 3 completed (8/8)

All nodes listed in the former phase plan were migrated to the helper-based pattern while preserving existing behavior/result contracts.

## Current architecture guidance

1. Treat tag/link helpers in `DpgNodeABC` as the single source of truth.
2. Keep `DeclarativeImageProcessNodeBase` focused on straightforward image-processing flows.
3. Prefer incremental, mechanical refactoring before unifying into higher-level base classes.

## Candidate future base families

- Source/Capture base
- Model inference base
- Dynamic slot aggregation base
- Sink/output base
- Script execution base

These categories are design guidance for future abstraction after helper standardization.

## Operational notes

- For paths reachable from `Node.update()`, prefer guarded helpers in `node_editor/util.py` (`dpg_get_value`, `dpg_set_value`, `dpg_get_item_children`) over direct `dpg.get_*` calls.
- Parse UI values defensively (`None` and invalid values can occur during async delete/import races).
