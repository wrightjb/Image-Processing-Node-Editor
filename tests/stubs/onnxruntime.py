"""Minimal onnxruntime stub for import-only unit tests.

The project has node declaration tests that import deep-learning node modules to
inspect their ports. Those tests do not run model inference, so they only need
onnxruntime imports to succeed in environments without the native runtime.
"""


class _TensorInfo:
    name = 'stub_tensor'


class InferenceSession:
    def __init__(self, *args, **kwargs):
        del args, kwargs

    def get_inputs(self):
        return [_TensorInfo()]

    def get_outputs(self):
        return [_TensorInfo()]

    def run(self, *args, **kwargs):
        del args, kwargs
        return []
