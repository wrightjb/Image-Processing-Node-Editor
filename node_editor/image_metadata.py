#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Image file metadata and compression helpers."""
import os
import cv2
import numpy as np


JPEG_STD_LUMA_Q50 = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
]

JPEG_STD_CHROMA_Q50 = [
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
]

JPEG_SUBSAMPLING_PARAM = getattr(cv2, 'IMWRITE_JPEG_SAMPLING_FACTOR', None)
JPEG_SUBSAMPLING_VALUES = {
    '4:4:4': getattr(cv2, 'IMWRITE_JPEG_SAMPLING_FACTOR_444', None),
    '4:2:2': getattr(cv2, 'IMWRITE_JPEG_SAMPLING_FACTOR_422', None),
    '4:2:0': getattr(cv2, 'IMWRITE_JPEG_SAMPLING_FACTOR_420', None),
    '4:1:1': getattr(cv2, 'IMWRITE_JPEG_SAMPLING_FACTOR_411', None),
    '4:4:0': getattr(cv2, 'IMWRITE_JPEG_SAMPLING_FACTOR_440', None),
}


def inspect_image_file(path):
    """Return stable, JSON-like metadata for an image file."""
    metadata = {
        'path': path,
        'filename': os.path.basename(path) if path else None,
        'exists': bool(path and os.path.isfile(path)),
    }
    if not metadata['exists']:
        return metadata

    metadata['file_size_bytes'] = os.path.getsize(path)
    with open(path, 'rb') as file_obj:
        data = file_obj.read()

    metadata['format'] = detect_format(data, path)
    if metadata['format'] == 'JPEG':
        metadata['jpeg'] = parse_jpeg_metadata(data)
    else:
        metadata['jpeg'] = None
    return metadata


def detect_format(data, path=None):
    if data.startswith(b'\xff\xd8\xff'):
        return 'JPEG'
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'PNG'
    if data.startswith(b'RIFF') and data[8:12] == b'WEBP':
        return 'WEBP'
    if data[:6] in (b'GIF87a', b'GIF89a'):
        return 'GIF'
    extension = os.path.splitext(path or '')[1].lower().lstrip('.')
    return extension.upper() if extension else 'Unknown'


def parse_jpeg_metadata(data):
    metadata = {
        'quantization_tables': {},
        'components': [],
        'subsampling': None,
        'progressive': False,
        'estimated_quality': None,
        'quality_confidence': 'unknown',
        'warnings': [],
    }
    index = 2
    data_len = len(data)
    while index + 4 <= data_len:
        if data[index] != 0xFF:
            index += 1
            continue
        while index < data_len and data[index] == 0xFF:
            index += 1
        if index >= data_len:
            break
        marker = data[index]
        index += 1
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        if index + 2 > data_len:
            break
        segment_length = int.from_bytes(data[index:index + 2], 'big')
        segment_start = index + 2
        segment_end = index + segment_length
        if segment_length < 2 or segment_end > data_len:
            break
        segment = data[segment_start:segment_end]
        if marker == 0xDB:
            _parse_dqt_segment(segment, metadata)
        elif marker in (0xC0, 0xC2):
            metadata['progressive'] = marker == 0xC2
            _parse_sof_segment(segment, metadata)
        index = segment_end

    metadata['estimated_quality'] = estimate_jpeg_quality(
        metadata['quantization_tables']
    )
    if metadata['estimated_quality'] is not None:
        metadata['quality_confidence'] = 'heuristic'
    return metadata


def _parse_dqt_segment(segment, metadata):
    index = 0
    while index < len(segment):
        table_info = segment[index]
        index += 1
        precision = table_info >> 4
        table_id = table_info & 0x0F
        count = 64
        if precision == 0:
            if index + count > len(segment):
                return
            values = list(segment[index:index + count])
            index += count
        else:
            byte_count = count * 2
            if index + byte_count > len(segment):
                return
            values = [
                int.from_bytes(segment[i:i + 2], 'big')
                for i in range(index, index + byte_count, 2)
            ]
            index += byte_count
        metadata['quantization_tables'][str(table_id)] = {
            'precision': precision,
            'values': values,
        }


def _parse_sof_segment(segment, metadata):
    if len(segment) < 6:
        return
    metadata['precision'] = segment[0]
    metadata['height'] = int.from_bytes(segment[1:3], 'big')
    metadata['width'] = int.from_bytes(segment[3:5], 'big')
    component_count = segment[5]
    components = []
    index = 6
    for _ in range(component_count):
        if index + 3 > len(segment):
            break
        component_id = segment[index]
        sampling = segment[index + 1]
        quant_table_id = segment[index + 2]
        components.append({
            'id': component_id,
            'horizontal_sampling': sampling >> 4,
            'vertical_sampling': sampling & 0x0F,
            'quantization_table_id': quant_table_id,
        })
        index += 3
    metadata['components'] = components
    metadata['subsampling'] = infer_jpeg_subsampling(components)


def infer_jpeg_subsampling(components):
    if len(components) < 3:
        return None
    y_component = components[0]
    h = y_component.get('horizontal_sampling')
    v = y_component.get('vertical_sampling')
    if (h, v) == (1, 1):
        return '4:4:4'
    if (h, v) == (2, 1):
        return '4:2:2'
    if (h, v) == (2, 2):
        return '4:2:0'
    if (h, v) == (4, 1):
        return '4:1:1'
    if (h, v) == (1, 2):
        return '4:4:0'
    return f'{h}x{v}'


def estimate_jpeg_quality(quantization_tables):
    if not quantization_tables:
        return None
    table = quantization_tables.get('0')
    reference = JPEG_STD_LUMA_Q50
    if table is None:
        table = next(iter(quantization_tables.values()))
        reference = JPEG_STD_CHROMA_Q50
    values = table.get('values') or []
    if len(values) != 64:
        return None
    scale_values = []
    for value, ref_value in zip(values, reference):
        if ref_value <= 0:
            continue
        scale_values.append((int(value) * 100 - 50) / ref_value)
    if not scale_values:
        return None
    scale = sorted(scale_values)[len(scale_values) // 2]
    if scale <= 0:
        return 100
    if scale < 100:
        quality = 5000 / scale
    else:
        quality = (200 - scale) / 2
    return int(max(1, min(100, round(quality))))


def summarize_metadata(metadata, multiline=False):
    if not isinstance(metadata, dict) or not metadata.get('exists', True):
        return 'Status: No metadata' if multiline else 'No metadata'

    image_format = metadata.get('format') or 'Unknown'
    jpeg = metadata.get('jpeg') or {}
    width = jpeg.get('width') or metadata.get('width')
    height = jpeg.get('height') or metadata.get('height')
    dimensions = f'{width}x{height}' if width and height else None

    if multiline:
        lines = [f'Format: {image_format}']
        if dimensions:
            lines.append(f'Dim: {dimensions}')
        if image_format == 'JPEG':
            subsampling = jpeg.get('subsampling') or 'unknown'
            quality = jpeg.get('estimated_quality')
            jpeg_type = 'progressive' if jpeg.get('progressive') else 'baseline'
            quality_text = f'Q≈{quality}' if quality is not None else 'Q≈unknown'
            lines.append(f'JPEG: {quality_text}, {subsampling}')
            lines.append(f'Type: {jpeg_type}')
        return '\n'.join(lines)

    parts = [str(image_format)]
    if dimensions:
        parts.append(dimensions)
    if image_format == 'JPEG':
        subsampling = jpeg.get('subsampling')
        quality = jpeg.get('estimated_quality')
        if subsampling:
            parts.append(subsampling)
        if quality is not None:
            parts.append(f'Q≈{quality}')
        parts.append('progressive' if jpeg.get('progressive') else 'baseline')
    return ', '.join(parts)


def format_metadata_report(metadata):
    if not isinstance(metadata, dict):
        return 'No metadata available.'
    lines = [f"Format: {metadata.get('format', 'Unknown')}"]
    if metadata.get('filename'):
        lines.append(f"File: {metadata.get('filename')}")
    if metadata.get('file_size_bytes') is not None:
        lines.append(f"Size: {metadata['file_size_bytes']} bytes")
    jpeg = metadata.get('jpeg') or {}
    if jpeg:
        lines.extend([
            f"Dimensions: {jpeg.get('width', '?')}x{jpeg.get('height', '?')}",
            f"JPEG type: {'progressive' if jpeg.get('progressive') else 'baseline'}",
            f"Subsampling: {jpeg.get('subsampling') or 'unknown'}",
            f"Estimated quality: {jpeg.get('estimated_quality') or 'unknown'}",
            f"Quality confidence: {jpeg.get('quality_confidence', 'unknown')}",
            f"Quantization tables: {len(jpeg.get('quantization_tables') or {})}",
        ])
    return '\n'.join(lines)


def jpeg_roundtrip(
    image,
    quality=90,
    subsampling='Auto',
    progressive=False,
    optimize=False,
    generation=1,
):
    quality = int(max(1, min(100, quality)))
    generation = int(max(1, generation))
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    if hasattr(cv2, 'IMWRITE_JPEG_PROGRESSIVE'):
        params.extend([int(cv2.IMWRITE_JPEG_PROGRESSIVE), int(bool(progressive))])
    if hasattr(cv2, 'IMWRITE_JPEG_OPTIMIZE'):
        params.extend([int(cv2.IMWRITE_JPEG_OPTIMIZE), int(bool(optimize))])
    if (
        JPEG_SUBSAMPLING_PARAM is not None
        and subsampling in JPEG_SUBSAMPLING_VALUES
        and JPEG_SUBSAMPLING_VALUES[subsampling] is not None
    ):
        params.extend([int(JPEG_SUBSAMPLING_PARAM), int(JPEG_SUBSAMPLING_VALUES[subsampling])])

    result = image.copy()
    encoded_size = None
    encoded_bytes = None
    for _ in range(generation):
        ok, encoded = cv2.imencode('.jpg', result, params)
        if not ok:
            raise RuntimeError('JPEG encode failed')
        encoded_bytes = encoded.tobytes()
        encoded_size = len(encoded_bytes)
        decoded = cv2.imdecode(np.frombuffer(encoded_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise RuntimeError('JPEG decode failed')
        result = decoded
    encode_metadata = parse_jpeg_metadata(encoded_bytes or b'')
    return result, {
        'codec': 'JPEG',
        'quality': quality,
        'subsampling': subsampling,
        'progressive': bool(progressive),
        'optimize': bool(optimize),
        'generation': generation,
        'encoded_size_bytes': encoded_size,
        'jpeg': encode_metadata,
    }


def png_roundtrip(image, compression=3, generation=1):
    compression = int(max(0, min(9, compression)))
    generation = int(max(1, generation))
    params = [int(cv2.IMWRITE_PNG_COMPRESSION), compression]
    result = image.copy()
    encoded_size = None
    for _ in range(generation):
        ok, encoded = cv2.imencode('.png', result, params)
        if not ok:
            raise RuntimeError('PNG encode failed')
        encoded_size = len(encoded.tobytes())
        decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise RuntimeError('PNG decode failed')
        result = decoded
    return result, {
        'codec': 'PNG',
        'compression': compression,
        'generation': generation,
        'encoded_size_bytes': encoded_size,
    }


def webp_roundtrip(image, quality=90, generation=1):
    quality = int(max(1, min(100, quality)))
    generation = int(max(1, generation))
    if not hasattr(cv2, 'IMWRITE_WEBP_QUALITY'):
        raise RuntimeError('OpenCV WebP encoding is not available')
    params = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
    result = image.copy()
    encoded_size = None
    for _ in range(generation):
        ok, encoded = cv2.imencode('.webp', result, params)
        if not ok:
            raise RuntimeError('WebP encode failed')
        encoded_size = len(encoded.tobytes())
        decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise RuntimeError('WebP decode failed')
        result = decoded
    return result, {
        'codec': 'WEBP',
        'quality': quality,
        'generation': generation,
        'encoded_size_bytes': encoded_size,
    }


def compression_roundtrip(image, codec, **params):
    codec_normalized = str(codec or 'JPEG').upper()
    if codec_normalized == 'JPEG':
        return jpeg_roundtrip(
            image,
            quality=params.get('quality', 90),
            subsampling=params.get('subsampling', 'Auto'),
            progressive=params.get('progressive', False),
            optimize=params.get('optimize', False),
            generation=params.get('generation', 1),
        )
    if codec_normalized == 'PNG':
        return png_roundtrip(
            image,
            compression=params.get('png_compression', params.get('quality', 3)),
            generation=params.get('generation', 1),
        )
    if codec_normalized == 'WEBP':
        return webp_roundtrip(
            image,
            quality=params.get('quality', 90),
            generation=params.get('generation', 1),
        )
    raise ValueError(f'Unsupported compression codec: {codec}')
