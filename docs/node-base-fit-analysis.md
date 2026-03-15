# Node base fit analysis (remaining non-declarative nodes)

This review covers the 28 node modules that still inherit directly from `DpgNodeABC`.

## Current helper-adoption snapshot (as of this branch)

`DpgNodeABC` now includes shared helpers for tag generation and connection parsing:

- `_node_name(node_id)`
- `_port_tag(node_name, value_type, port_name)`
- `_value_tag(port_tag)`
- `_iter_connections(connection_list)`
- `_extract_source_node_key(source_tag)`
- `_extract_port_name(tag)`

Adoption status across node modules (`node/**/node_*.py`, excluding base files):

- **44 total node modules**
- **16 nodes** inherit `DeclarativeImageProcessNodeBase` (and therefore use the shared helpers through that base)
- **2 nodes** (`Int Value`, `Float Value`) now call the helpers directly
- **26 nodes** still use legacy manual tag/connection string handling and are pending mechanical refactor

So, helper centralization is in place in `DpgNodeABC`; the remaining work is migration/adoption in legacy nodes.

## Recommendation summary

### Good candidates for `DeclarativeImageProcessNodeBase` with small base extensions

These nodes follow a mostly “single image in → image out (+ optional scalar/text params)” shape and can likely migrate with minimal extra hooks:

- `node/other_node/node_on_off_switch.py`
- `node/draw_node/node_draw_information.py`
- `node/draw_node/node_puttext.py` (needs a small custom widget hook for color editor/group)
- `node/draw_node/node_image_alpha_blend.py` (if base gains explicit multi-image-input support)
- `node/analysis_node/node_BRISQUE.py` (if base gains support for a static result text panel)

### Better fit for a **different** dedicated base (not current declarative image-process base)

- **Source/Capture base** (stateful producers, file dialogs, capture handles, optional worker process):
  - `node/input_node/node_webcam_input.py`
  - `node/input_node/node_video_input.py`
  - `node/input_node/node_video_set_frame_pos_input.py`
  - `node/input_node/node_still_image.py`
  - `node/input_node/node_rtsp_input.py`
  - `node/preview_release_node/node_screen_capture.py`
- **Model inference base** (model registry, lazy instance cache, provider/device toggle, score slider, result dict formatting):
  - `node/deep_learning_node/node_classification.py`
  - `node/deep_learning_node/node_object_detection.py`
  - `node/deep_learning_node/node_semantic_segmentation.py`
  - `node/deep_learning_node/node_face_detection.py`
  - `node/deep_learning_node/node_pose_estimation.py`
  - `node/deep_learning_node/node_monocular_depth_estimation.py`
  - `node/deep_learning_node/node_low_light_image_enhancement.py`
  - `node/preview_release_node/node_mot.py`
- **Dynamic slot aggregation base** (runtime Add Slot behavior and generated ports):
  - `node/draw_node/node_image_concat.py`
  - `node/analysis_node/node_fps.py`
- **Sink/output base** (consume stream and render/store, no real transformed output contract):
  - `node/draw_node/node_result_image.py`
  - `node/draw_node/node_result_large_image.py`
  - `node/other_node/node_video_writer.py`
- **Script execution base** (user-provided code text execution):
  - `node/preview_release_node/node_code_exec.py`

### Probably keep simple direct `DpgNodeABC` (or add a tiny “value output” base later)

- `node/input_node/node_int_value.py`
- `node/input_node/node_float_value.py`

These two are very small output-only scalar nodes; migration value is limited unless you introduce a tiny `DeclarativeValueOutputNodeBase`.

## Why mix-ins were not necessary for already-converted process nodes

A separate mix-in layer was likely unnecessary because `DeclarativeImageProcessNodeBase` already absorbed the repeated concerns into overridable hooks:

- Parameter declaration and UI generation (`parameters`, `_add_parameter_ui`).
- Shared update flow: input syncing, parameter cast/clamp, elapsed-time handling, texture update.
- Extension points (`build_custom_ui`, `normalize_parameter_values`, `on_node_added`, `on_settings_applied`).

In short, the current base is already acting as a “composed behavior surface.” For this specific set of converted process nodes, mix-ins would add class hierarchy complexity with little functional gain.

## Concrete standardization approach (recommended)

If the goal is to standardize **all** nodes around tag generation and connection parsing, the safest approach is:

1. Keep tag/connection helpers in `DpgNodeABC` as the single source of truth.
2. Keep `DeclarativeImageProcessNodeBase` focused on the simple image-process workflow, while relying on those shared helpers.
3. Migrate legacy nodes incrementally as “mechanical refactors” (replace string concatenation/split logic first, behavior unchanged).

This gives you consistency across every node without forcing all node types into a single base class too early.

## Refactor plan for remaining helper adoption

### Status update (as of this branch)

- ✅ **Phase 1** helper adoption completed (8/8 modules).
- ✅ **Phase 2** helper adoption completed (10/10 modules).
- 🔄 Remaining legacy set is now the original **Phase 3** modules (8 modules).

### Phase 1 (low risk, quick wins) — 8 modules **(completed)**

- `node/draw_node/node_result_image.py`
- `node/draw_node/node_result_large_image.py`
- `node/draw_node/node_draw_information.py`
- `node/other_node/node_on_off_switch.py`
- `node/analysis_node/node_rgb_histgram.py`
- `node/analysis_node/node_BRISQUE.py`
- `node/input_node/node_still_image.py`
- `node/preview_release_node/node_code_exec.py`

Goal: adopt `_node_name/_port_tag/_value_tag` + `_extract_source_node_key` where applicable, no functional change.

### Phase 2 (moderate complexity) — 10 modules **(completed)**

- `node/input_node/node_webcam_input.py`
- `node/input_node/node_video_input.py`
- `node/input_node/node_video_set_frame_pos_input.py`
- `node/input_node/node_rtsp_input.py`
- `node/other_node/node_video_writer.py`
- `node/draw_node/node_puttext.py`
- `node/draw_node/node_image_alpha_blend.py`
- `node/draw_node/node_image_concat.py`
- `node/analysis_node/node_fps.py`
- `node/preview_release_node/node_screen_capture.py`

Goal: same helper adoption plus normalizing connection iteration to `_iter_connections` where safe.

Result: migrated to `_node_name/_port_tag/_value_tag` and standardized connection iteration with `_iter_connections` / `_extract_source_node_key` in these modules while preserving behavior.

### Phase 3 (highest complexity) — 8 modules

- `node/deep_learning_node/node_classification.py`
- `node/deep_learning_node/node_object_detection.py`
- `node/deep_learning_node/node_semantic_segmentation.py`
- `node/deep_learning_node/node_face_detection.py`
- `node/deep_learning_node/node_pose_estimation.py`
- `node/deep_learning_node/node_monocular_depth_estimation.py`
- `node/deep_learning_node/node_low_light_image_enhancement.py`
- `node/preview_release_node/node_mot.py`

Goal: helper adoption while preserving model/provider/runtime behavior and existing result dict contracts.
