import numpy as np

from node_editor import image_metadata


def test_summarize_metadata_multiline_uses_labels():
    metadata = {
        'format': 'JPEG',
        'jpeg': {
            'width': 4000,
            'height': 3000,
            'subsampling': '4:2:0',
            'estimated_quality': 92,
            'progressive': False,
        },
    }

    summary = image_metadata.summarize_metadata(metadata, multiline=True)

    assert summary.splitlines() == [
        'Format: JPEG',
        'Dim: 4000x3000',
        'JPEG: Q≈92, 4:2:0',
        'Type: baseline',
    ]


def test_compression_roundtrip_filters_jpeg_only_parameters(monkeypatch):
    calls = {}

    def fake_jpeg_roundtrip(image, **kwargs):
        calls['image'] = image
        calls['kwargs'] = kwargs
        return image, {'codec': 'JPEG'}

    monkeypatch.setattr(image_metadata, 'jpeg_roundtrip', fake_jpeg_roundtrip)
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    output, metadata = image_metadata.compression_roundtrip(
        image,
        'JPEG',
        quality=81,
        subsampling='4:2:0',
        progressive=True,
        optimize=True,
        png_compression=9,
        generation=2,
    )

    assert output is image
    assert metadata == {'codec': 'JPEG'}
    assert calls['kwargs'] == {
        'quality': 81,
        'subsampling': '4:2:0',
        'progressive': True,
        'optimize': True,
        'generation': 2,
    }
