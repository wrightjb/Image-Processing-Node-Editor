# Repository Guidelines

This project is an image processing application built with DearPyGui. It allows users to build pipelines by connecting nodes. Each node is placed under `node/` and is implemented using the interface defined in `node/node_abc.py`. The GUI logic resides in `node_editor/` and the entry point is `main.py`.

## Current project goal and product direction
- The near-term project goal is to help recreate image-editing workflows originally made on Android phone editors such as Polish, Image Editor, Snapseed, and Google Photos editor.
- A typical target workflow may combine blur, curves, hue adjustment, and other tools in sequence, sometimes repeated a dozen or more times. The original intermediate images may be available, but the exact settings often were not saved.
- Because many recreated workflows have a dozen or more steps, small method or parameter mismatches can accumulate across the graph and produce large final-image differences. Favor exact tool semantics, parameter ranges, and tuner accuracy over merely approximate matches when the goal is faithful recreation.
- Prioritize non-destructive node workflows that can recreate favorite edited images at higher resolution and make the recovered workflow easy to apply to other images or videos.
- Tuner nodes are important because they help discover likely original parameters, or practical equivalents, for missing settings from earlier mobile-editor workflows.
- Expect future work to add niche or rarely used editing tools when needed to replicate specific images, not only broadly common filters.
- Creative generation of new images and workflows is also a goal, but workflow recreation is the current focus.

## JPEG and compression-aware recreation direction
- Many target/reference images were saved as JPEGs, so pixel diffs against fresh recreations may retain unavoidable compression error even when editing parameters are correct.
- It is useful to expose available metadata from image inputs, including JPEG-related metadata when present. However, note that the exact JPEG quality/compression level is not always stored directly or reliably in the file; estimating it may require heuristics or encoder-specific analysis.
- Consider adding either image-input metadata display or a dedicated metadata/inspection node for compression-related information such as file format, EXIF fields, quantization tables, chroma subsampling, dimensions, and any quality estimate we can infer.
- A JPEG compression process node is desirable both for matching saved references during tuning and as a creative effect, especially for intentionally heavy compression artifacts.
- Compression-aware tuning is tenable and likely useful: tuners could compare against a target after applying matching or estimated compression to the transformed source before scoring error. Prefer explicit graph data flow or metadata outputs over hidden peeking between nodes when practical, but pragmatic tuner access to image-input metadata is acceptable if it keeps workflows simple.
- Longer term, image export/save functionality should expose JPEG and other format-specific compression settings so recreated workflows can be saved consistently.

## Directory overview
- `main.py` – Application entry point creating the node editor; runs the main event loop asynchronously of GUI callbacks.
- `node_editor/` – GUI and common utilities.
- `node/` – Collection of node modules grouped in subdirectories such as `input_node`, `process_node`, and `deep_learning_node`.
- `docker/` – Example container image for running with GPU support.


## Refactor reference
- Node base/helper refactor canonical notes: `docs/NODE_BASE_CLASS_REFACTOR.md`.

## Coding style
- Follow PEP 8 and the settings in `.editorconfig` (4‑space indentation for Python files).
- New nodes should subclass `DpgNodeABC` and be placed in an appropriate subfolder inside `node/` using the naming pattern `node_<name>.py`.
- Keep public documentation bilingual (Japanese and English) when updating README files.

## Testing
There is an automated test suite using pytest. Before committing, run the test suite:

First, make sure you have activated the virtual environment:
```bash
.venv\Scripts\activate
```

Then run the tests:
```bash
python -m pytest
```

If OpenCV cannot be imported in your environment (for example, missing
`libGL.so.1` in headless CI), run pytest with the repository's official cv2
stub mode:
```bash
python -m pytest --use-cv2-stub
```

## Async DearPyGui safety notes
- In this repository, `Node.update()` runs in an async worker by default (`main.py` loop). GUI callbacks can mutate/delete DPG items concurrently.
- For code paths reachable from `Node.update()`, prefer guarded helpers in `node_editor/util.py` (`dpg_get_value`, `dpg_set_value`, `dpg_get_item_children`) instead of direct `dpg.get_*` calls.
- Parse node inputs defensively (`None` / malformed UI values can occur during delete/import races).
- See `docs/async-dpg-race-guide.md` for details and architecture recommendations.

## Commit messages
Write clear commit messages in English. Use a short summary line followed by a blank line and additional details if necessary.
