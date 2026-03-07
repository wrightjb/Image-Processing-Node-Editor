# Node Base Class Refactor Analysis and Implementation Plan

## Why this is feasible

A metadata-driven base class is highly feasible in this repository because almost every node already follows the same lifecycle and naming conventions:

- All node classes subclass `DpgNodeABC` and implement the same five methods (`add_node`, `update`, `get_setting_dict`, `set_setting_dict`, `close`).
- Nodes consistently build tag strings as `"{node_id}:{node_tag}:{type}:{port}"` and then derive `...Value` tags.
- Most visual nodes allocate a black placeholder texture, register it via `dpg.add_raw_texture`, then display/update it with `convert_cv_to_dpg` + `dpg_set_value`.
- Many nodes repeat connection parsing logic to locate upstream image sources and to forward scalar values.
- Many nodes repeat optional elapsed-time instrumentation when `use_pref_counter` is enabled.

In short: the current code is already using a de-facto template; it is just duplicated manually.

## What should be in a new base class

A practical refactor is to add a **new, optional convenience base class** (for example `DeclarativeNodeBase`) that still satisfies `DpgNodeABC`, and supports:

1. **Port/field declarations**
   - Inputs, outputs, and static controls declared in a Python list/dict schema.
   - Automatic tag generation from declarations.

2. **Default node rendering**
   - Generic `add_node()` that renders DPG widgets from the declarations.
   - Built-in preview texture setup for image outputs.

3. **Common update pipeline helpers**
   - Resolve connected input image(s) and scalar forwarding from links.
   - Standard elapsed-time wrapper.
   - Standard output texture update.

4. **Settings persistence helpers**
   - Save/restore declared control values + node position automatically.

5. **Hook points for one-off behavior**
   - `before_process(...)`, `process(...)`, `after_process(...)`
   - Optional override for `build_custom_ui(...)` to inject non-standard controls.
   - Optional override for connection parsing if a node needs unusual link semantics.

This gives a high-value default while preserving flexibility for unusual nodes.

## What should stay custom (and why)

Some nodes are not purely declarative and should keep custom logic (possibly still inheriting helper utilities):

- **Resource/lifecycle nodes** (e.g., camera/video/model nodes) that hold long-lived external resources and use custom `close()` behavior.
- **Deep learning nodes** that dynamically initialize model instances/provider choices and maintain internal model caches.
- **Nodes with bespoke UI interactions** (file dialogs, multiline code editor, record toggles, concat slot controls) where a simple field schema is insufficient.
- **Output/action nodes** where behavior is side-effect-driven (e.g., writing files), not just transforming an image.

## Recommended architecture

Use a layered approach rather than one giant superclass:

- `DpgNodeABC` (existing): minimal contract.
- `NodeTagMixin`: tag building/parsing helpers.
- `NodeSettingMixin`: generic get/set setting serialization for declared controls.
- `NodePreviewMixin`: image texture create/update helpers.
- `NodeTimingMixin`: elapsed-time measurement helper.
- `DeclarativeNodeBase(DpgNodeABC, ...)`: integrates mixins and supports declaration-driven nodes.

This avoids forcing every node into the same mold and minimizes base-class bloat.

## Mixin applicability: what *would not* use each mixin

Your intuition is mostly right: **many basic nodes will use almost all mixins**. The exceptions are mainly non-visual scalar/action nodes and specialized UI/resource nodes.

### 1) `NodeTagMixin`
Likely non-users: **none** (practically every node should use this).

- Reason: all nodes rely on consistent tag composition/parsing for link/update behavior.
- Recommendation: make this near-universal and low-risk.

### 2) `NodeSettingMixin`
Likely non-users: **very few**; mostly nodes with intentionally custom persistence semantics.

- Candidate partial/non-users:
  - `node/preview_release_node/node_code_exec.py` (multiline code text may need custom migration/version handling).
  - Any node that stores transient runtime-only handles should keep those out of generic serialized settings.
- Recommendation: default-on, but allow per-node `serialize_fields` opt-out and custom hooks.

### 3) `NodePreviewMixin` (texture create/update for image previews)
Likely non-users: nodes without image preview textures.

- Clear non-users/candidates:
  - Scalar source nodes: `node/input_node/node_int_value.py`, `node/input_node/node_float_value.py`.
  - Control/toggle nodes: `node/other_node/node_on_off_switch.py`.
  - Analysis nodes that may output metrics/text only (depends on concrete UI path): `node/analysis_node/node_fps.py`.
- Partial users:
  - Nodes with image output but unusual preview sizing/placement can still use the mixin with override hooks.

### 4) `NodeTimingMixin`
Likely non-users: nodes where elapsed-time output is not rendered/enabled or timing is semantically misleading.

- Candidate non-users:
  - Scalar/control nodes (`Int/Float/OnOff`).
  - Nodes doing mainly side effects rather than image transform display timing (for example, writer/toggle-only flows).
- Recommendation: keep opt-in via declaration flag (e.g., `has_elapsed_output`).

### Bottom line for “basic nodes”

For classic “image in -> small parameter UI -> image out” nodes (blur/brightness/contrast/etc.), you are correct: they should generally use **all four** mixins. The mixin decomposition is mostly for:

- keeping complex nodes from inheriting irrelevant behavior,
- avoiding base-class growth,
- making adoption incremental instead of all-or-nothing.

### Quick applicability matrix (pragmatic default)

- **Process nodes (basic filters/transforms):** Tag ✅ / Setting ✅ / Preview ✅ / Timing ✅
- **Deep-learning visual nodes:** Tag ✅ / Setting ✅ / Preview ✅ / Timing ✅ (with custom process/model lifecycle)
- **Input image/video/webcam nodes:** Tag ✅ / Setting ✅ / Preview ✅ / Timing optional
- **Scalar/control nodes (`Int`, `Float`, `OnOff`):** Tag ✅ / Setting ✅ / Preview ❌ / Timing ❌
- **Action/output side-effect nodes (e.g., VideoWriter):** Tag ✅ / Setting ✅ / Preview optional / Timing optional

## Migration estimate and blast radius

### Candidate groups

1. **Low-risk, high-duplication nodes** (best first wave)
   - Basic process nodes: blur, brightness, contrast, gamma, threshold, grayscale, etc.
   - Many of these follow one image input + slider(s) + image output + optional elapsed time.

2. **Medium-risk nodes**
   - Input nodes with combo/select but still simple frame output.
   - Draw/analysis nodes with small variations.

3. **High-risk nodes (defer)**
   - Video writer, code exec, concat/alpha blend, deep-learning nodes, MOT, RTSP-specific controls.

### Practical rollout

- **Phase 1:** Add mixins/base class without changing existing nodes.
- **Phase 2:** Migrate 3–5 simple process nodes and validate parity.
- **Phase 3:** Expand to remaining simple/medium nodes incrementally.
- **Phase 4:** Decide case-by-case whether complex nodes should partially adopt only mixins.

## Compatibility constraints to preserve

- Keep current public node class shape (`class Node(DpgNodeABC): ...`) valid for dynamic discovery.
- Preserve tag naming scheme because graph connection logic depends on string format.
- Keep `get_setting_dict`/`set_setting_dict` output compatible with existing import/export JSON.
- Do not break `node_editor` discovery/update loops.

## Risk register

- **Risk: Base class becomes too feature-heavy.**
  - Mitigation: split functionality into mixins and keep hook surface small.
- **Risk: Hidden per-node assumptions in tag naming and connection parsing.**
  - Mitigation: preserve format exactly and migrate in small batches.
- **Risk: Regression in import/export settings.**
  - Mitigation: add/extend tests for serialization parity before broad migration.
- **Risk: Hard-to-debug GUI regressions.**
  - Mitigation: test migrated nodes with deterministic unit tests where possible and manual smoke checks.

## Concrete implementation plan

1. Create `node/base/` helpers (`tags.py`, `settings.py`, `preview.py`, `timing.py`).
2. Add `declarative_node_base.py` implementing:
   - declaration schema
   - default `add_node/update/get_setting_dict/set_setting_dict/close`
   - hook methods for custom behavior.
3. Add tests for:
   - tag generation/parsing
   - settings serialization compatibility
   - timing wrapper and texture update helper behavior (with stubs/mocks).
4. Migrate a pilot set of simple nodes (e.g., Brightness, Contrast, Blur).
5. Validate with `pytest --use-cv2-stub` and quick manual smoke test.
6. Iterate node-by-node; keep complex nodes on legacy path until clear benefit.

## Suggested declaration shape (example)

```python
class Node(DeclarativeNodeBase):
    node_label = "Brightness"
    node_tag = "Brightness"

    spec = {
        "inputs": [
            {"name": "image", "type": "Image", "kind": "input"},
            {"name": "beta", "type": "Int", "kind": "input", "widget": "slider_int", "min": 0, "max": 255, "default": 0},
        ],
        "outputs": [
            {"name": "image", "type": "Image"},
            {"name": "elapsed", "type": "TimeMS", "optional": "use_pref_counter"},
        ],
    }

    def process(self, frame, values, context):
        return cv2.convertScaleAbs(frame, alpha=1.0, beta=values["beta"]), None
```

This addresses your idea directly: most nodes can become declarations + core processing logic, while one-off nodes can override hooks or stay custom.

## Implementation progress (ongoing)

### Completed waves

- **Wave 1 (already completed)**
  - Base class introduced at `node/base/declarative_node_base.py`.
  - Pilot nodes migrated: `Brightness`, `Contrast`, `Blur`.

- **Wave 2 (this update)**
  - Additional simple process nodes migrated to declarative base:
    - `Grayscale`
    - `EqualizeHist`
    - `GammaCorrection`
    - `Flip`
    - `ApplyColorMap`
    - `Threshold`
  - Base class hardened for async GUI race resilience guidance:
    - defensive connection parsing for malformed/stale links
    - per-port parameter matching (avoids cross-updating same-typed controls)
    - safe default fallback when UI values are missing/invalid during async races
  - Declarative test coverage expanded for new behaviors.

- **Wave 3 (completed)**
  - Additional process nodes migrated to declarative base:
    - `GaussianBlur`
    - `Canny`
    - `Resize`
  - Base class enhanced with small extensibility hooks:
    - `normalize_parameter_values(...)` for per-node defensive normalization in `update()`
    - `on_node_added(...)` / `on_settings_applied(...)` for UI-only post setup (e.g., `Auto Sigma` enable/disable)
    - declarative `input_int` widget support


- **Wave 4 (completed)**
  - Migrated additional nodes to declarative base:
    - `Crop`
    - `SimpleFilter`
  - Fixed `SimpleFilter` settings/caching mismatch by declaratively persisting all kernel fields (`Input02`..`Input11`) instead of only first four.
  - Expanded tests for:
    - Crop crossed-bounds normalization behavior
    - SimpleFilter full settings coverage and linked `K` clamp range


- **Wave 5 (completed)**
  - Migrated `Curves` to declarative base using a hybrid approach:
    - kept drag-point plot UI/callbacks as node-local custom logic
    - adopted shared image I/O and elapsed-time output behavior from base class
    - persisted/restored curve points via declarative custom-settings hooks

### Next suggested wave

- **Wave 6 candidate: `OmnidirectionalViewer` (stateful migration)**
  - Preserve the node-local internal map cache (`phi/theta`) optimization because recomputation is expensive.
  - Review lifecycle/state cleanup (`close`) and per-node cache invalidation behavior under delete/import races.
  - Adopt declarative base incrementally after cache/lifecycle semantics are explicitly covered by tests.

### Why these are deferred from simple-wave migrations

- `OmnidirectionalViewer` has non-trivial internal state/caching semantics beyond declarative slider-driven transforms.

### Entry criteria for starting each deferred wave

- Add focused tests for node-specific behavior first (interaction/state invariants).
- Keep migration PRs isolated (one complex node per PR) to simplify regression triage.
- Confirm async mode behavior with `pytest --use-cv2-stub` and a manual smoke check in the editor.
