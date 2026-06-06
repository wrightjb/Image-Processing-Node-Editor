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
  parsing, and the standard editor toolbar now live on `DpgNodeBase`; non-graph
  UI control aliases now have explicit `_control_tag()` / `_control_value_tag()`
  helpers so graph-port tags and toolbar/control tags are separated in code even
  while they share the compact DearPyGui alias format.
- `DpgNodeBase` now has composed node/port/value tag helpers that accept a
  `node_id` directly, reducing repeated nested tag construction in node code.
- `node.port_model` defines the first reusable passive `NodeRef` / `PortRef`
  records for node-side declarations.
- `DpgNodeBase` creates typed `PortRef` metadata through `PortSpecs` /
  `create_ports()` and dynamic/data-driven `create_port()` calls using
  `PortDirection` and `PortDataType` enum values while preserving compact DPG
  aliases at the boundary.
- `node.port_model` now includes `PortSpec`, `InputPort` / `OutputPort` /
  `ParameterPort`, `PortSpecs`, and `PortHandles` for the Option A typed handle
  layer.
- `DpgNodeBase.create_port()` can now create one data-driven or dynamic
  `PortSpec` at runtime and store it either as a named handle or in a per-node
  handle collection, reusing the same declaration/registration path as static
  `create_ports()`. `PortSpec` also supports parameter-style default control
  tags for value widgets through `ParameterPort`.
- Compact port tag construction/parsing has started moving into the
  `node.port_serialization` boundary module.
- `DeclarativeImageProcessNodeBase` now creates its standard image
  input/output, optional elapsed-time output, and data-driven parameter graph
  ports through `create_port()` handles. Parameter handles are stored under a
  `ports(node_id).parameters` collection, while cache/result toggles still use
  value tags because they are toolbar controls, not graph ports.
- Direct non-declarative node `add_node()` implementations now declare graph
  attributes through class-level `PortSpecs` and `create_ports()` handles while
  preserving existing compact DearPyGui aliases at the UI boundary.
- Direct non-declarative node `update()` implementations now iterate typed
  connection info records through `_iter_connection_infos()` instead of the legacy
  `_iter_connections()` adapter. All active direct `DpgNodeBase` node classes now
  declare their static graph ports with base-owned `PortSpecs` / `PortHandles`
  and read value tags from generated handles in setting/update paths; dynamic
  slot creation and data-driven declarative parameters use `create_port()`.
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
  `_link_refs`; old compact-pair link lists are no longer stored or seeded in
  the editor model. DearPyGui callback aliases are parsed only at explicit
  boundary points. The
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

## What is left

The graph-port part of the refactor is effectively complete for active nodes.
Static graph ports use `PortSpecs`/`PortHandles`, dynamic graph slots and
declarative process parameters use `create_port()`, editor graph internals use
typed `LinkRef` storage, and import/export writes typed `link_refs`. It is
reasonable to stop the broad migration here if the remaining compact strings are
kept at explicit boundaries.

The remaining work is cleanup and boundary-hardening, not another mechanical
node-by-node graph-port migration:

1. **Finish non-graph control alias cleanup.** Many remaining compact tags are
   UI controls rather than graph endpoints: model selectors, provider selectors,
   buttons, color editors, cache/result toggles, and similar DearPyGui widgets.
   Continue moving those call sites from generic `_port_tag()` / `_value_tag()`
   calls to `_control_tag()` / `_control_value_tag()` so control aliases are
   explicit. Do not convert them to `PortSpec` unless they become graph ports.
2. **Keep graph declaration adapter APIs removed.** Active nodes no longer use
   `input_port()`, `output_port()`, or `parameter_port()`, so those public
   adapters have been removed from `DpgNodeBase`. New static graph ports should
   use `PortSpecs`; new dynamic or data-driven graph ports should use
   `create_port()`. Disabled nodes must migrate before being re-enabled.
3. **Optionally improve declarative parameter metadata.** Declarative process
   graph identity now flows through `ParameterPort`/`create_port()`, but the
   parameter definitions are still dictionaries (`type`, `port`, `widget`,
   `default`, etc.). If those dictionaries remain annoying, introduce a typed
   `ParameterSpec`/widget metadata object; otherwise leave them alone.
4. **Keep tests focused on typed links.** `_link_refs` is canonical. Core
   editor and import/export tests should seed links through `_mdl_add_link()` or
   typed import payloads, then assert typed link pairs for graph operations.
5. **Audit disabled/legacy nodes when re-enabling them.** `*.py.disable` files
   such as the disabled QR-code node are outside the active migration. If one is
   re-enabled, migrate its graph ports to `PortSpecs` first.

Suggested next implementation step: continue item 1 by migrating direct node UI
controls such as model/provider selectors and color editors to the explicit
control-tag helpers.

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

Implemented model:

- `_link_refs` is the canonical in-memory list of typed `LinkRef` records,
- `_mdl_add_link()` accepts string aliases or `PortRef` endpoints and normalizes
  successful links to `LinkRef`,
- graph sorting, cycle detection, delete/reconnect, runtime scheduling, history
  replay, and node connection adapters consume or preserve typed links,
- export writes typed `link_refs`, and import expects/remaps `link_refs` rather
  than performing normal-path legacy `link_list` conversion.

Boundary-only parsing is still acceptable for:

- DearPyGui callback aliases,
- legacy compatibility adapters and tests,
- dynamic UI/port creation boundaries,
- tests that intentionally exercise malformed serialized data.

## Step 5: Add typed port specs and base-owned per-node handles before more node migration

Do **not** continue a broad mechanical pass that only swaps compact string
helpers for another lookup helper. Before migrating more nodes, add the small
foundation needed for consistent node authoring inside the current architecture.

Near-term target:

1. Done initially: add `PortDataType` for `Int`, `Float`, `Image`, `TimeMS`, and
   `Text`, while keeping existing `TYPE_*` constants as compatibility aliases
   during migration.
2. Done initially: add `PortSpec` declarations for static graph ports. A spec
   describes a semantic handle name, direction, data type, optional index
   override, and UI label/control metadata if needed.
3. Done initially: add base-owned per-`node_id` handle storage on `DpgNodeBase`,
   so node code can call `ports = self.create_ports(node_id)` in `add_node()` and
   `ports = self.ports(node_id)` in later lifecycle methods.
4. Done initially: generate attribute-style handles from declarations
   (`ports.value`, `ports.image`, `ports.elapsed`) rather than requiring repeated
   string-keyed calls such as `port_handle(node_id, "value")`.
5. Done for static graph ports: derive legacy `Input01` / `Output01` names from
   `PortDirection` plus numeric index. `port_name` is now primarily a
   compatibility/serialization value for static `PortSpecs`; dynamic/data-driven
   declaration paths may still pass explicit legacy names.
6. Done initially: add `DpgNodeBase.create_port()` for one-off data-driven and
   dynamic `PortSpec` declarations. It stores generated refs either as named
   handles or in a per-node handle collection while reusing the same typed
   declaration/registration path as `create_ports()`.
7. In progress: compact DPG tag serialization/deserialization lives partly in the
   explicit `node.port_serialization` boundary module, but many static/control UI
   aliases still call base tag helpers. Normal graph-port logic should work with
   `PortRef` handles; only DearPyGui aliases, import/export, legacy compatibility
   paths, tests, and dynamic UI boundaries should build or parse compact strings.

All active direct `DpgNodeBase` node classes now serve as the validation set for
this style. Active dynamic-slot nodes and the declarative process-node base now
declare runtime/data-driven graph ports through `create_port()`. New
dynamic/data-driven graph ports should use `create_port()`, and static graph
ports should be authored through `PortSpecs` going forward.

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
- Done: add canonical typed `LinkRef` storage in `_link_refs`. Normal
  editor-core and import/export link assertions now prefer typed link pairs.
- Done: add typed `link_refs` import/export schema and remove normal-path legacy
  `link_list` import/export fallback after fixture conversion.
- Done: move graph sorting, cycle detection, delete-through-node reconnection, and
  runtime scheduling onto typed `LinkRef` iteration.
- Done: migrate history payloads and node `update()` connection consumers from
  string pairs to typed refs.
- Done: Option A static graph-port foundation is in place. `PortDataType`, typed
  `PortSpec` declarations, base-owned per-node port handles, and the
  `node.port_serialization` boundary module have been introduced; all active
  direct `DpgNodeBase` node classes now use generated handles for static graph
  ports.
- Done: add `DpgNodeBase.create_port()` for one-off dynamic and data-driven
  `PortSpec` declarations.
- Done: migrate active dynamic-slot graph inputs (`ImageConcat` and `FPS`) to
  `create_port(..., collection=...)`.
- Done: migrate `DeclarativeImageProcessNodeBase` standard graph ports and
  parameter graph ports to `create_port()` handles, including parameter
  collection handles.
- In progress: migrate non-graph static/control aliases to `_control_tag()` /
  `_control_value_tag()` helpers; declarative process toolbar controls and the
  `VideoWriter`/`OnOffSwitch` control widgets are now on that explicit boundary.
- Done: remove the public graph declaration adapters (`input_port()`,
  `output_port()`, `parameter_port()`) after active nodes and tests moved to
  `PortSpecs`/`create_port()`.
- Done: remove normal import/export test coupling to old compact link-list
  storage; tests now seed links through `_mdl_add_link()` or typed import
  payloads.
- Remaining: finish reviewing compact static/control tags and compatibility
  helper usage.
