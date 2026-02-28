# Repository Guidelines

This project is an image processing application built with DearPyGui. It allows users to build pipelines by connecting nodes. Each node is placed under `node/` and is implemented using the interface defined in `node/node_abc.py`. The GUI logic resides in `node_editor/` and the entry point is `main.py`.

## Directory overview
- `main.py` – Application entry point creating the node editor; runs the main event loop asynchronously of GUI callbacks.
- `node_editor/` – GUI and common utilities.
- `node/` – Collection of node modules grouped in subdirectories such as `input_node`, `process_node`, and `deep_learning_node`.
- `docker/` – Example container image for running with GPU support.

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

## Commit messages
Write clear commit messages in English. Use a short summary line followed by a blank line and additional details if necessary.
