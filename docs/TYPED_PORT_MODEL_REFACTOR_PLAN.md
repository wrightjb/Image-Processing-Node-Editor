# Typed Port Model Refactor Plan

This plan captures the proposed sequencing for making `NodeRef` / `PortRef`
authoritative graph model data instead of metadata reconstructed from DearPyGui
string tags.

## Goals

- Keep DearPyGui tags as a UI/backend boundary detail, not the source of truth
  for graph identity.
- Let nodes declare ports through a shared concrete node base so the editor can
  register `PortRef` data at creation time.
- Migrate existing nodes mechanically, without inventing additional family base
  classes unless a later refactor finds real duplicated behavior.
- Preserve import/export compatibility while allowing a future typed schema.

## Current status

Implemented so far:

- `DpgNodeBase` has been introduced as the concrete DearPyGui node base.
- Direct node/base subclasses under `node/` now inherit from `DpgNodeBase`
  instead of `DpgNodeABC`, while preserving existing behavior and tag strings.
- Existing concrete helper methods for tag construction, legacy connection/tag
  parsing, and the standard editor toolbar now live on `DpgNodeBase`.
- `DpgNodeBase` now has composed node/port/value tag helpers that accept a
  `node_id` directly, reducing repeated nested tag construction in node code.
- `node.port_model` defines the first reusable passive `NodeRef` / `PortRef`
  records for node-side declarations.
- `DpgNodeBase` also has the first typed port declaration APIs
  (`input_port`, `output_port`, `parameter_port`) that create passive `PortRef`
  metadata while preserving the compact DPG tag format.
- `DeclarativeImageProcessNodeBase` uses typed `PortRef` declarations for its
  standard image input/output, elapsed-time output, and declared parameter ports;
  cache/result toggles still use value tags because they are toolbar controls,
  not graph ports.
- Direct non-declarative node `add_node()` implementations now declare their DPG
  input/output graph attributes through `input_port()` / `output_port()` while
  preserving existing compact tag strings.
- `DpgNodeABC` is back to the abstract lifecycle contract, shared metadata, shared
  type constants, and the optional editor hook.
- The editor imports the shared `node.port_model` records, registers node-owned
  declared ports during node creation, and now prefers registered `PortRef` data
  for right-click hovered-port discovery and node-port lookup. Legacy compact tag
  parsing remains as a compatibility boundary for imported graphs, callback
  aliases, and undo/redo data.

Important limitation of the current implementation:

- The editor now has typed `LinkRef` records for canonical link registry entries,
  but history/import/export still expose compact string pairs. The next step is
  to migrate those boundaries toward a typed serialized schema so old compact
  strings remain only a compatibility adapter.

## Next work

1. Migrate history, runtime, import, and export from string pairs toward typed
   refs behind compatibility adapters.
2. Add a typed serialized link schema and import/export round-trip coverage for
   both legacy compact strings and typed ref-backed internal state.
3. Keep legacy parsing only for imported graphs, callback aliases, undo/redo data,
   and other compatibility boundaries until the checked-in graph fixtures have
   been converted.

## Step 1: Split the abstract interface from concrete node behavior

Create a concrete base node, for example `DpgNodeBase`, and move implementation
helpers out of `DpgNodeABC` when they are not abstract-interface concerns.

`DpgNodeABC` should keep only the minimum interface contract:

- node identity metadata (`node_label`, `node_tag`, `_ver`) if still useful as
  part of the plugin contract,
- shared value-type constants if keeping them on the interface avoids churn,
- abstract lifecycle methods (`add_node`, `update`, `get_setting_dict`,
  `set_setting_dict`, `close`),
- optional editor hook method signatures such as
  `on_editor_parameter_value_applied`.

Move concrete behavior to `DpgNodeBase`:

- tag construction helpers (`_node_name`, `_port_tag`, `_value_tag`),
- legacy tag parsing helpers (`_extract_source_node_key`, `_extract_port_name`,
  `_extract_node_id`, `_iter_connections`),
- standard editor toolbar helpers (`_editor_toolbar_attr_tag`,
  `_editor_toolbar_group_tag`, `_editor_delete_button_tag`,
  `add_editor_toolbar`),
- new typed port declaration/registration helpers.

The new base should provide explicit port declaration APIs that create both the
compact DearPyGui tag and typed metadata in one place. Suggested shape:

```python
image_in = self.input_port(node_id, self.TYPE_IMAGE, 'Input01')
image_out = self.output_port(node_id, self.TYPE_IMAGE, 'Output01')
threshold = self.parameter_port(node_id, self.TYPE_INT, 'Input02')
```

Each returned object should expose at least:

- `node_ref`,
- `direction`,
- `data_type`,
- `index` / port name,
- `dpg_tag`,
- optional value/control tags for parameter widgets.

The editor should receive these declarations through a registration callback or
node-owned registry during `add_node()`, so normal graph operations do not need
to parse tags after creation.

## Step 2: Move `DeclarativeImageProcessNodeBase` onto the concrete base

Refactor `DeclarativeImageProcessNodeBase` to inherit from `DpgNodeBase` and use
its typed port helpers for:

- standard image input,
- standard image output,
- elapsed-time output,
- declared parameter ports,
- cache/result-image toggle controls.

This is the highest-leverage first migration because declarative process nodes
already centralize a large amount of repeated node UI and setting behavior.

## Step 3: Use registered ports for hovered-port discovery

After `DpgNodeBase` can register typed ports and
`DeclarativeImageProcessNodeBase` has moved onto it, fold audit item 2
(hovered-port discovery) into this refactor. This should happen after
creation-time registration is available, because pre-refactor `_port_registry`
data was populated lazily from parsed strings rather than authoritatively during
node creation.

Done: `_cntrl_get_hovered_output_port_tag()` and
`_cntrl_get_hovered_input_port_tag()` now iterate registered `PortRef` data
before falling back to synthesized DearPyGui tags for legacy/imported graphs.
`_cntrl_find_node_port()` also prefers registered refs so newly created nodes can
be connected through the typed registry rather than through fixed index scans.

Implemented incremental shape:

1. public helpers still return compact DPG tags, so surrounding context-menu code
   remains compatible,
2. returned tags are sourced from registered `PortRef` objects when available,
3. legacy tag synthesis remains only as a fallback for imported/test graphs that
   have not gone through creation-time registration,
4. behavior tests cover registered high-index ports, legacy fallback registration,
   add-node-from-hovered-output, add-node-to-hovered-input, insert-node-between-links,
   and occupied input replacement.

This makes hovered-port lookup the first editor-side proof that typed port
registration is reliable, without forcing the entire graph model to change at
once.

## Step 4: Update editor graph internals to prefer typed refs

Once hovered-port lookup is backed by creation-time port registration, continue
shifting editor internals from string pairs toward typed objects.

Implemented intermediate model:

- keep `_node_link_list` as exported legacy string pairs until import/export and
  runtime are migrated,
- add `LinkRef` as the canonical typed link record stored in `_link_registry` and
  `_link_by_dest_port_ref`,
- make `_mdl_add_link()` accept string aliases or `PortRef` endpoints on the normal
  path and normalize successful links to `LinkRef`,
- keep string-to-`PortRef` resolution available at boundaries.

Boundary-only parsing is still acceptable for:

- DearPyGui callback aliases,
- old import files,
- undo/redo payloads until history commands are migrated,
- tests that intentionally exercise malformed serialized data.

## Step 5: Migrate all remaining nodes to the concrete base in one mechanical wave

Do one broad, mechanical pass changing direct `DpgNodeABC` subclasses to inherit
from `DpgNodeBase` and use the typed port helpers.

There is no need to introduce extra source/capture/model/sink/dynamic-slot base
families as part of this step. The current nodes already work because they share
the same tag construction convention, so this migration should be mostly
risk-controlled and repetitive:

1. replace direct `_port_tag(...)` / `_value_tag(...)` construction with concrete
   base port helper calls,
2. keep existing UI layout and business logic intact,
3. register static ports during `add_node()`,
4. register dynamic ports at the same time they are added to DearPyGui,
5. preserve existing external tag strings for compatibility during the first
   pass.

If family-specific abstractions are useful later, add them only after this
migration exposes real duplication. They are not required to make `PortRef`
authoritative.

## Step 6: Move runtime/history/import/export to typed graph data

After nodes reliably register ports, migrate graph consumers in this order:

1. history link commands store typed endpoint data or typed serializable endpoint
   records instead of raw strings,
2. graph sorting and cycle detection use `NodeRef` / `PortRef` directly,
3. runtime connection lists consume typed links or a typed adapter rather than
   repeatedly splitting source/destination tags,
4. import parses serialized endpoints once at the file boundary,
5. export serializes typed links from the canonical graph model.

At the end of this step, parsing a port tag should be a compatibility adapter,
not part of normal graph mutation.

## Serialization and DearPyGui tag format notes

The official DearPyGui documentation says item IDs can be integers or strings,
and describes string aliases as usable anywhere UUIDs are used. It documents the
requirement that user-provided aliases are unique, but does not document a
specific maximum alias length.

Recommendation:

- Keep the current compact colon-separated tag format for DearPyGui aliases and
  for backward-compatible serialized files initially.
- Do not switch to JSON-style strings embedded inside DPG tags; that would be
  longer, harder to read in debugging output, and not necessary once the live
  graph model is typed.
- Treat compact strings as a boundary encoding only. Internally, use typed
  `NodeRef`, `PortRef`, and `LinkRef` objects.

If a versioned compact format is needed later, prefer a small URI/URN-like or
version-prefixed delimiter format over JSON-in-a-string, for example:

```text
v1:node_id:node_tag:data_type:Output01
```

or keep the current form unchanged:

```text
node_id:node_tag:data_type:Output01
```

The current form is acceptable as long as each field comes from controlled node
metadata and cannot contain the delimiter. If arbitrary user-defined names ever
become part of tags, add percent-encoding or another explicit escaping rule at
the serialization boundary.

Common compact alternatives such as MessagePack, CBOR, or base64-encoded binary
payloads are useful for storage/transmission, but they make DearPyGui aliases and
logs less readable. For this project, the better tradeoff is typed internal data
plus a readable compact boundary string.

## Suggested checkpoints

- Done: introduce `DpgNodeBase`, migrate direct node subclasses to inherit from
  it, and move existing concrete helper methods from `DpgNodeABC` into
  `DpgNodeBase`.
- Done: migrate declarative and direct node static graph attributes to typed
  `PortRef` declarations while preserving existing compact DPG tags.
- Done: wire node-owned declared ports into the editor registry during node
  creation, with callback support for dynamic ports created later.
- Done: replace hovered-port discovery and node-port lookup with registered
  `PortRef` iteration first, with legacy compact-tag scanning as fallback.
- Done: add typed `LinkRef` registry entries while preserving legacy `_node_link_list`
  pairs for existing export/history/runtime code.
- Next: add a typed import/export schema and conversion path for checked-in graph
  fixtures.
- Add import/export round-trip tests covering both legacy compact strings and any
  future typed endpoint schema.
