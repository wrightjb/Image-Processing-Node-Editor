# Graph Runtime Performance Plan

## Purpose

The graph runtime currently polls every node and uses cache signatures to avoid
re-running unchanged image operations. This preserves compatibility with live
sources and mutable node outputs, but a cache hit can still perform expensive
work: GUI settings reads, signature construction, full-image fallback hashing,
and deep copies of cached images.

This plan improves that design incrementally. Each stage should preserve still
image, video, live-source, import/export, preview, and custom-node behavior. The
stages deliberately establish explicit revisions and ownership rules before
removing the conservative fallbacks that currently provide correctness.

## Current execution model

On every runtime tick, the scheduler walks all nodes in topological order. For
each cacheable node it reads settings and builds a signature from connections,
upstream results or revisions, and node settings. Video sources add stream and
frame tokens. When no upstream revision is available, array contents are hashed
as a fallback. Cached images and results have historically been deep-copied back
into the active dictionaries even when a static signature is unchanged.

The existing output-version dictionary already prevents repeated image hashing
through most cached static pipelines. Still-image inputs opt into source caching,
and video inputs publish frame tokens. The remaining work is to make the normal
path revision-driven rather than content-inspection-driven.

## Stage 1: Make static cache hits no-ops

**Status: implemented.**

For a non-video cache hit whose signature is unchanged, retain the image and
result objects already stored in the active runtime dictionaries. Do not replace
them with deep copies of the cache snapshot. If a caller provides cache state
without the corresponding active image or result, restore only the missing
value defensively.

Video frame-cache hits remain unchanged in this stage. Seeking or revisiting a
frame can require replacing the currently active frame with a different cached
frame, so that path needs the ownership work in Stage 4 before its copies can be
removed safely.

### Validation

- An unchanged static node is not updated twice.
- Active image and result object identities survive an unchanged second tick.
- Missing active values can still be restored from an existing cache.
- Cached preview rendering behavior remains unchanged.
- Video frame-cache behavior remains unchanged.

### Node migration scope

No node changes are required. This is isolated to the graph runtime and its
tests.

## Stage 2: Separate compute parameters from persistence and presentation state

**Status: partially implemented.** The runtime now excludes the universal
presentation-only keys `pos`, `__result_image_enabled__`, and
`__result_large_image_enabled__` from cache signatures. The dedicated API and
legacy/custom-node migration described below remain future work.

Introduce a cache-specific API, for example `get_compute_setting_dict(node_id)`,
that contains only values capable of changing a node's returned image or result.
Keep `get_setting_dict()` as the import/export representation. Node position,
preview state, and other presentation-only values must not invalidate computed
outputs.

Provide a compatibility default that derives compute settings from existing
settings while removing universally non-computational keys such as position.
Declarative nodes should implement the new behavior once in their shared base.
Legacy nodes with custom settings should migrate incrementally, with tests for
settings whose classification is not obvious.

### Node migration scope

This does **not** require an atomic update to every node. Runtime and base-class
changes can provide a safe default. The declarative base covers its subclasses.
Legacy/custom nodes need individual updates only to obtain the complete benefit
or when their compute and persistence settings cannot be classified centrally.

## Stage 3: Give every output an authoritative revision

Assign a small immutable revision token whenever a node produces a new output:

- cached deterministic nodes use their compute-input signature;
- uncached or stateful nodes receive a monotonically increasing generation;
- still images use a file/source generation;
- video files use stream and frame tokens;
- webcams, RTSP, and screen capture use source/frame generations; and
- scalar and callback-driven sources increment when their output value changes.

Once every active upstream output has a revision, cache signatures no longer
need to inspect or hash image bytes. Keep content hashing temporarily as an
assertion, diagnostic mode, or compatibility fallback until revision coverage
is verified.

### Node migration scope

Most processing nodes do **not** need changes because the runtime can assign a
generation after `update()` returns. Source families that change independently
of graph inputs need explicit revision semantics. Shared input/base classes can
cover some of them, but video, webcam, RTSP, screen capture, file, scalar, and
unusual stateful/custom sources require review and possibly targeted changes.

## Stage 4: Define output ownership and reduce remaining copies

Adopt and document an ownership contract: a node owns the image it returns, and
downstream nodes must treat upstream images as read-only. A node that performs
in-place work must copy its input first. Audit built-in nodes and add mutation
tests around representative pipelines and custom-code boundaries.

After the contract is established:

- share immutable references between active state and cache where safe;
- avoid the second copy made during cache insertion;
- restore cached video frames by reference where safe; and
- retain copies at explicitly unsafe or external mutation boundaries.

Read-only NumPy flags may be used in tests or selected paths, but should not be
enabled globally until OpenCV and custom-node compatibility is confirmed.

### Node migration scope

Every node that consumes images should be **audited**, but every node should not
need a code change. Only nodes that mutate an upstream array, retain a mutable
borrowed reference, or cross an unsafe custom/external boundary need changes.
Declarative and utility helpers should encode the common safe behavior.

## Stage 5: Cache normalized topology

Add a graph-structure revision to the editor. Rebuild filtered, normalized
connection adapters and topological metadata only when nodes, links, or ports
change. Ordinary runtime ticks reuse that structure.

### Node migration scope

No ordinary node changes are required. Work is confined to the editor model,
runtime, typed-port/link infrastructure, and tests. Nodes with genuinely dynamic
port definitions may need to notify the editor when their port structure changes.

## Stage 6: Dirty-subgraph scheduling

Track dirty nodes explicitly and propagate invalidation only to reachable
downstream nodes. Dirty causes include compute-parameter callbacks, topology
changes, new live-source frames, file revisions, and explicit stateful-node
requests. Coalesce rapid UI changes and evaluate the dirty set in topological
order. Retain a compatibility polling mode while migration is incomplete.

### Node migration scope

Core scheduling belongs in the runtime and editor, and shared declarative
callbacks can cover many nodes. Nodes with custom callbacks or independently
changing external state must emit invalidation events, so this is the stage most
likely to require broad node-level integration. It still need not modify every
node if base classes and standard callback helpers provide the default behavior.

## Implementation order and safeguards

1. Land the static cache-hit no-op with identity and restoration tests.
2. Add lightweight runtime timing instrumentation for actual node updates, slow
   signature construction (including fallback image hashing), and slow cache
   restoration. **Implemented:** launch with `--runtime_trace`; reports are
   throttled per node and operation, and slow-operation reports default to a
   50 ms threshold configurable with `--runtime_trace_threshold_ms`. Repeated
   reports for the same node default to a 30-second interval, configurable with
   `--runtime_trace_repeat_seconds`, so video graphs do not flood the terminal.
   Slow settings reads are traced separately from signature construction so GUI
   polling costs are visible.
   Update lines identify whether work is an intentionally uncached source, a
   disabled cache, a cold cache, or a cache miss. With tracing enabled, cache
   misses also name changed signature components (`connections`, `upstream`,
   `frames`, or `settings`) to make repeated invalidation diagnosable.
3. Introduce compute-only settings with compatibility fallbacks.
4. Complete revision coverage and measure any remaining image-hash fallback.
5. Audit mutation behavior and reduce cache-insertion/video copies.
6. Cache normalized topology.
7. Add dirty-subgraph scheduling behind a compatibility switch.
8. Remove legacy polling and content hashing only after tests and instrumentation
   show that no supported source depends on them.

Each stage should test still images, changed parameters, node movement, branching
graphs, disabled caches, video playback and seeking, live sources where feasible,
node deletion/import races, cached previews, and custom/stateful nodes.
