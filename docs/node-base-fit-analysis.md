# Node base fit analysis (remaining non-declarative nodes)

This review covers the 28 node modules that still inherit directly from `DpgNodeABC`.

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

## Suggested next step order

1. Add a small **source/capture base** and migrate `still_image` first (lowest risk), then `video_input`/`webcam_input`.
2. Add a **dynamic-slot base** and migrate `image_concat` and `fps` together.
3. Add a **model inference base** and migrate one DL node (`semantic_segmentation` or `classification`) as template.
4. Re-evaluate whether explicit mix-ins are still needed after those three bases; likely still optional.

## Concrete standardization approach (recommended)

If the goal is to standardize **all** nodes around tag generation and connection parsing, the safest approach is:

1. Put tag/connection helpers in `DpgNodeABC` (single source of truth):
   - `_node_name(node_id)`
   - `_port_tag(node_name, value_type, port_name)`
   - `_value_tag(port_tag)`
   - `_iter_connections(connection_list)`
   - `_extract_source_node_key(source_tag)` / `_extract_port_name(tag)`
2. Keep `DeclarativeImageProcessNodeBase` focused on the simple image-process workflow, but rely on those shared helpers from `DpgNodeABC`.
3. Migrate legacy nodes incrementally as “mechanical refactors” (replace string concatenation/split logic first, behavior unchanged).

This gives you consistency across every node without forcing all node types into a single base class too early.
