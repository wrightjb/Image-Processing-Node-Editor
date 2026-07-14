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


def test_compression_parameters_accept_direct_compression_metadata():
    assert image_metadata.compression_parameters_from_metadata({
        'codec': 'JPEG',
        'quality': 77,
        'subsampling': '4:4:4',
        'progressive': True,
        'optimize': False,
    }) == {
        'codec': 'JPEG',
        'quality': 77,
        'subsampling': '4:4:4',
        'progressive': True,
        'optimize': False,
        'generation': 1,
    }


def test_format_metadata_report_explains_progressive_and_optimize():
    report = image_metadata.format_metadata_report({
        'format': 'JPEG',
        'jpeg': {
            'width': 640,
            'height': 480,
            'progressive': True,
            'subsampling': '4:2:0',
            'estimated_quality': 82,
            'quality_confidence': 'heuristic',
            'quantization_tables': {'0': {}},
        },
    })

    assert 'Progressive JPEG: yes' in report
    assert 'Optimize JPEG: not stored in JPEG metadata' in report
