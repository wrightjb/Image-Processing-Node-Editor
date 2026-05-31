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
- `DpgNodeBase` also has typed port declaration APIs (`input_port`,
  `output_port`, `parameter_port`) that create passive `PortRef` metadata using
  `PortDirection` enum values while preserving the compact DPG tag format.
- `DeclarativeImageProcessNodeBase` uses typed `PortRef` declarations for its
  standard image input/output, elapsed-time output, and declared parameter ports;
  cache/result toggles still use value tags because they are toolbar controls,
  not graph ports.
- Direct non-declarative node `add_node()` implementations now declare their DPG
  input/output graph attributes through `input_port()` / `output_port()` while
  preserving existing compact tag strings.
- Direct non-declarative node `update()` implementations now iterate typed
  connection info records through `_iter_connection_infos()` instead of the legacy
  `_iter_connections()` adapter. Source value nodes and the still-image node now
  keep their declared output `PortRef` handles and read value tags from those
  handles in setting/update paths.
- `DpgNodeABC` is back to the abstract lifecycle contract, shared metadata, shared
  type constants, and the optional editor hook.
- The editor imports the shared `node.port_model` records, registers node-owned
  declared ports during node creation, and now prefers registered `PortRef` data
  for right-click hovered-port discovery and node-port lookup. Legacy compact tag
  parsing remains as a compatibility boundary for imported graphs, callback
  aliases, and undo/redo data.

Current implementation note:

- The editor now exports and imports typed `link_refs` directly; graph sorting,
  cycle detection, delete-through-node reconnection, the runtime scheduler,
  undo/redo link history, and the node `update()` connection boundary all consume
  or preserve typed `LinkRef` data. The editor stores canonical links in
  `_link_refs`; `_node_link_list` remains only as a legacy compact-pair adapter
  for compatibility with older tests/integrations and DearPyGui boundary code. The
  declarative image process base and direct non-declarative nodes read typed
  connection-info records, while `_iter_connections()` remains only as a legacy
  compatibility adapter for external callers and tests that explicitly cover that
  boundary.

## Chosen near-term architecture: Option A

After discussion, the refactor will **not** switch immediately to one Python
object per graph node. The current plugin/editor architecture keeps one node
object per node type and passes `node_id` into lifecycle methods. That shape is
confusing for per-node state, but rewriting it now would touch node loading, the
runtime loop, history, import/export, and callbacks all at once.

Instead, use Option A as the near-term plan:

- Keep the existing editor/runtime lifecycle and one-node-object-per-node-type
  plugin model for now.
- Stop doing broad mechanical conversions that merely replace one compact string
  lookup with another compact or semantic string lookup.
- Add a base-owned, typed per-node port-handle layer so node implementations can
  keep using `node_id` lifecycle methods while accessing ports through handles
  created by `DpgNodeBase`, not through node-local dictionaries.
- Introduce typed `PortSpec` / `PortDataType` declarations before migrating more
  nodes, so `Input01` / `Output01` and data-type strings are derived at the
  DearPyGui/import/export boundary instead of being authored throughout node
  implementations.
- Keep compact DPG aliases and legacy pair adapters only at explicit boundary
  functions: DearPyGui item creation/callbacks, import/export, compatibility
  tests, and old history payload replay.

The intended node authoring style after this foundation is closer to:

```python
class Node(DpgNodeBase):
    port_specs = PortSpecs(
        value=OutputPort(PortDataType.INT),
    )

    def add_node(...):
        ports = self.create_ports(node_id)
        dpg.add_input_int(tag=ports.value.value_tag)

    def get_setting_dict(self, node_id):
        ports = self.ports(node_id)
        return {ports.value.value_tag: dpg_get_value(ports.value.value_tag)}
```

Normal node code should not repeatedly call string-keyed helpers such as
`port_handle(node_id, "value")`, and it should not define ad hoc maps such as
`self._output_ports[str(node_id)]`. A string key may appear once in a declaration
(`value=...` or equivalent), but regular node logic should use generated handle
attributes (`ports.value`).

## Next work

1. Add `PortDataType` and keep compatibility aliases for the existing
   `TYPE_*` constants.
2. Add `PortSpec` / `InputPort` / `OutputPort` declarations and a base-owned
   per-`node_id` port-handle registry on `DpgNodeBase`.
3. Move compact tag build/parse functions into an explicit serialization
   boundary module.
4. Convert one small node (for example `IntValue`) to the new handle style and
   evaluate ergonomics before continuing the broader node migration.

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
compact DearPyGui tag and typed metadata in one place. The initial helpers accept
legacy `TYPE_*` data-type values and optional compatibility port names, but the
Option A target is to declare ports by semantic spec/handle and let the base
derive legacy names from direction plus numeric index:

```python
ports = self.create_ports(node_id)
image_in = ports.image
image_out = ports.result
threshold = ports.threshold
```

Each returned or generated port handle should expose at least:

- `node_ref`,
- typed `direction`,
- typed `data_type`,
- numeric `index`,
- boundary `dpg_tag`,
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
`_cntrl_get_hovered_input_port_tag()` now iterate registered `PortRef` data only;
legacy DearPyGui tag synthesis is no longer used for hovered-port discovery.
`_cntrl_find_node_port()` also requires registered refs, so newly created nodes are
connected through the typed registry rather than through fixed index scans.

Implemented incremental shape:

1. public helpers still return compact DPG tags, so surrounding context-menu code
   remains compatible,
2. returned tags are sourced from registered `PortRef` objects,
3. behavior tests cover registered high-index ports, missing-registration behavior,
   add-node-from-hovered-output, add-node-to-hovered-input, insert-node-between-links,
   and occupied input replacement.

This makes hovered-port lookup the first editor-side proof that typed port
registration is reliable, without forcing the entire graph model to change at
once.

## Step 4: Update editor graph internals to prefer typed refs

Once hovered-port lookup is backed by creation-time port registration, continue
shifting editor internals from string pairs toward typed objects.

Implemented intermediate model:

- keep `_node_link_list` as an internal string-pair adapter until history/runtime
  are migrated,
- add `LinkRef` as the canonical typed link record stored in `_link_registry` and
  `_link_by_dest_port_ref`,
- make `_mdl_add_link()` accept string aliases or `PortRef` endpoints on the normal
  path and normalize successful links to `LinkRef`,
- export typed `link_refs` from `LinkRef` data and import `link_refs` without
  normal-path fallback to legacy `link_list`.

Boundary-only parsing is still acceptable for:

- DearPyGui callback aliases,
- old import files,
- undo/redo payloads until history commands are migrated,
- tests that intentionally exercise malformed serialized data.

## Step 5: Add typed port specs and base-owned per-node handles before more node migration

Do **not** continue a broad mechanical pass that only swaps compact string
helpers for another lookup helper. Before migrating more nodes, add the small
foundation needed for consistent node authoring inside the current architecture.

Near-term target:

1. Add `PortDataType` for `Int`, `Float`, `Image`, `TimeMS`, and `Text`, while
   keeping existing `TYPE_*` constants as compatibility aliases during migration.
2. Add `PortSpec` declarations for static graph ports. A spec should describe a
   semantic handle name, direction, data type, optional index override, and UI
   label/control metadata if needed.
3. Add base-owned per-`node_id` handle storage on `DpgNodeBase`, so node code can
   call `ports = self.create_ports(node_id)` in `add_node()` and
   `ports = self.ports(node_id)` in later lifecycle methods.
4. Generate attribute-style handles from declarations (`ports.value`,
   `ports.image`, `ports.elapsed`) rather than requiring repeated string-keyed
   calls such as `port_handle(node_id, "value")`.
5. Derive legacy `Input01` / `Output01` names from `PortDirection` plus numeric
   index. `port_name` should become a compatibility/serialization value, not the
   node-authored source of truth.
6. Move compact DPG tag serialization/deserialization into an explicit boundary
   helper/module. Normal node logic should work with `PortRef` handles; only
   DearPyGui aliases, import/export, and legacy compatibility paths should build
   or parse compact strings.

Only after this foundation exists should remaining nodes be migrated. The first
validation migration should be a small node such as `IntValue`; if the resulting
node code is not simpler than the current node-local `_output_ports` map, pause
and revisit the design before touching more files.

Family-specific abstractions can still be added later if repeated UI/state
patterns emerge, but they should build on the same typed spec/handle layer rather
than introducing separate graph identity conventions.

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

Current typed graph-consumer checkpoint:

- Graph sorting iterates typed `LinkRef` records and uses `NodeRef` metadata for
  dependency ordering while preserving legacy connection-list values for node
  `update()` calls.
- Cycle detection validates the candidate link once into a `LinkRef` and builds
  adjacency from typed registered links.
- Delete-through-node reconnection reads source/destination node IDs and link data
  types from `LinkRef` endpoints.
- Runtime scheduling prefers typed sorted connection refs and wraps typed links in
  a compatibility adapter for node `update()` calls; the adapter exposes typed
  endpoints while still iterating like the legacy source/destination tag pair.
- `DeclarativeImageProcessNodeBase` consumes typed adapter fields for its image
  source lookup and linked-parameter synchronization while keeping legacy pair
  fallback behavior. `_iter_connections()` now yields tag-compatible strings that
  retain `PortRef` metadata, so existing non-declarative nodes using shared source,
  port-name, node-id, and value-tag helpers also avoid reparsing typed endpoints.
- History replay normalizes stored link entries through a link-pair adapter, so
  commands can carry `LinkRef` payloads while older string-pair commands continue
  to replay through the same ID-remapping path. New link-add, link-replace,
  delete/reconnect, import, and add-node history payloads now preserve `LinkRef`
  values when the editor has a registered link.

Current typed import/export checkpoint:

- New exports include `link_refs` records with typed source/destination endpoint
  metadata and no longer write legacy `link_list`.
- Imports require `link_refs` and remap endpoint node IDs through the same import
  ID map as nodes.
- The checked-in exported graph fixtures have been converted externally, normal
  import no longer falls back to legacy compact-link payloads, and import/export
  tests now construct typed `link_refs` payloads directly instead of building
  compact `link_list` inputs and converting them inside the test.

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
  `PortRef` iteration; legacy compact-tag scanning has been removed from these
  normal paths.
- Done: add canonical typed `LinkRef` storage in `_link_refs` while preserving
  legacy `_node_link_list` compact pairs as a compatibility adapter.
- Done: add typed `link_refs` import/export schema and remove normal-path legacy
  `link_list` import/export fallback after fixture conversion.
- Done: move graph sorting, cycle detection, delete-through-node reconnection, and
  runtime scheduling onto typed `LinkRef` iteration.
- Done: migrate history payloads and node `update()` connection consumers from
  string pairs to typed refs.
- In progress: adopt Option A before further node migration: keep the current
  editor/runtime lifecycle, add `PortDataType`, typed `PortSpec` declarations,
  base-owned per-node port handles, and explicit compact tag serialization
  boundaries. Then rework one small node to prove the authoring style before
  touching the remaining compact-tag-heavy node implementations.
