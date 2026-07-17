#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import hashlib
import pickle

from node.port_model import LinkConnectionAdapter


class GraphRuntime:
    """Owns graph update state (images/results/cache) and executes update ticks."""

    def __init__(self, cache_enabled=True, cache_source_nodes=False):
        self.node_image_dict = {}
        self.node_result_dict = {}
        self.node_cache_dict = {}
        self.node_version_dict = {}
        self.cache_enabled = cache_enabled
        self.cache_source_nodes = cache_source_nodes

    def step(self, node_editor, mode_async=True):
        update_node_info(
            node_editor,
            self.node_image_dict,
            self.node_result_dict,
            node_cache_dict=self.node_cache_dict,
            node_version_dict=self.node_version_dict,
            mode_async=mode_async,
            cache_enabled=self.cache_enabled,
            cache_source_nodes=self.cache_source_nodes,
        )


def _freeze_cache_value(value):
    if isinstance(value, (str, int, float, bool, type(None))):
        return value

    if isinstance(value, bytes):
        return hashlib.sha1(value).hexdigest()

    if isinstance(value, (list, tuple)):
        return tuple(_freeze_cache_value(item) for item in value)

    if isinstance(value, set):
        return tuple(sorted(_freeze_cache_value(item) for item in value))

    if isinstance(value, dict):
        return tuple(
            sorted(
                (
                    _freeze_cache_value(key),
                    _freeze_cache_value(item),
                )
                for key, item in value.items()
            )
        )

    if (
        hasattr(value, 'shape') and
        hasattr(value, 'dtype') and
        hasattr(value, 'tobytes')
    ):
        try:
            return (
                'ndarray',
                tuple(value.shape),
                str(value.dtype),
                hashlib.sha1(value.tobytes()).hexdigest(),
            )
        except TypeError:
            pass

    return repr(value)


def _build_node_signature(
    node_id,
    connection_list,
    node_image_dict,
    node_result_dict,
    node_setting,
    node_version_dict=None,
):
    if node_version_dict is None:
        node_version_dict = {}
    upstream_values = []
    upstream_frame_tokens = []
    for source_tag, _ in connection_list:
        source_node_id_name = ':'.join(source_tag.split(':')[:2])
        source_result = node_result_dict.get(source_node_id_name)
        frame_token = _extract_frame_token(source_result)
        if frame_token is not None:
            upstream_frame_tokens.append((source_tag, frame_token))
        source_version = node_version_dict.get(source_node_id_name)
        if source_version is not None:
            upstream_image_value = ('version', _freeze_cache_value(source_version))
        else:
            upstream_image_value = _freeze_cache_value(
                node_image_dict.get(source_node_id_name)
            )
        upstream_values.append((
            source_tag,
            upstream_image_value,
            _freeze_cache_value(_strip_cache_meta(source_result)),
        ))

    signature_payload = {
        'node_id': node_id,
        'connection_list': connection_list,
        'upstream_values': upstream_values,
        'upstream_frame_tokens': upstream_frame_tokens,
        'node_setting': _freeze_cache_value(node_setting),
    }
    payload_bytes = pickle.dumps(signature_payload)
    return hashlib.sha1(payload_bytes).hexdigest()


def _build_pipeline_signature_for_video(
    node_id,
    connection_list,
    upstream_frame_tokens,
    node_result_dict,
    node_setting,
):
    """
    Build a stable signature for video pipelines that is invariant to frame index.
    """
    upstream_result_values = []
    for source_tag, _ in connection_list:
        source_node_id_name = ':'.join(source_tag.split(':')[:2])
        source_result = node_result_dict.get(source_node_id_name)
        upstream_result_values.append((
            source_tag,
            _freeze_cache_value(_strip_cache_meta(source_result)),
        ))

    signature_payload = {
        'node_id': node_id,
        'connection_list': connection_list,
        'upstream_result_values': upstream_result_values,
        'upstream_stream_tokens': [
            (source_tag, stream_id)
            for source_tag, (stream_id, _) in upstream_frame_tokens
        ],
        'node_setting': _freeze_cache_value(node_setting),
    }
    payload_bytes = pickle.dumps(signature_payload)
    return hashlib.sha1(payload_bytes).hexdigest()


def _extract_frame_token(result):
    if not isinstance(result, dict):
        return None

    if result.get('__cache_kind__') != 'video_frame':
        return None

    return (
        result.get('__cache_stream__'),
        result.get('__cache_frame__'),
    )


def _strip_cache_meta(result):
    if not isinstance(result, dict):
        return result

    return {
        key: value
        for key, value in result.items()
        if not str(key).startswith('__cache_')
    }


def _connection_info_to_update_connection(connection_info):
    if isinstance(connection_info, LinkConnectionAdapter):
        return connection_info
    if (
        hasattr(connection_info, 'source')
        and hasattr(connection_info, 'destination')
        and hasattr(connection_info, 'legacy_pair')
    ):
        return LinkConnectionAdapter(connection_info)
    return connection_info


def _connection_info_node_names(connection_info):
    if (
        hasattr(connection_info, 'source')
        and hasattr(connection_info, 'destination')
    ):
        return (
            connection_info.source.node_ref.node_id_name,
            connection_info.destination.node_ref.node_id_name,
        )

    source_tag, dest_tag = connection_info
    return ':'.join(source_tag.split(':')[:2]), ':'.join(dest_tag.split(':')[:2])


def update_node_info(
    node_editor,
    node_image_dict,
    node_result_dict,
    node_cache_dict=None,
    node_version_dict=None,
    mode_async=True,
    cache_enabled=True,
    cache_source_nodes=False,
):
    """
    Update all nodes in topological order with optional in-memory caching.

    Cache path (enabled nodes):
      1) build a signature from upstream outputs + node settings
      2) if signature matches previous run, reuse cached output and skip update()

    Non-cache path:
      always run update() so UI/callback-driven state changes are picked up.
    """
    if node_cache_dict is None:
        node_cache_dict = {}
    if node_version_dict is None:
        node_version_dict = {}

    if not cache_enabled and node_cache_dict:
        node_cache_dict.clear()

    def _is_valid_connection(connection_info, valid_nodes):
        if len(connection_info) != 2:
            return False

        source_node_id_name, dest_node_id_name = _connection_info_node_names(
            connection_info
        )
        if source_node_id_name not in valid_nodes:
            return False
        if dest_node_id_name not in valid_nodes:
            return False

        return True

    node_list = list(node_editor.get_node_list())
    active_node_set = set(node_list)

    deleted_image_node_id_name_list = [
        node_id_name for node_id_name in node_image_dict.keys()
        if node_id_name not in active_node_set
    ]
    for deleted_node_id_name in deleted_image_node_id_name_list:
        del node_image_dict[deleted_node_id_name]

    deleted_result_node_id_name_list = [
        node_id_name for node_id_name in node_result_dict.keys()
        if node_id_name not in active_node_set
    ]
    for deleted_node_id_name in deleted_result_node_id_name_list:
        del node_result_dict[deleted_node_id_name]
        node_version_dict.pop(deleted_node_id_name, None)

    if hasattr(node_editor, 'get_sorted_node_connection_refs'):
        sorted_node_connection_dict = node_editor.get_sorted_node_connection_refs()
    else:
        sorted_node_connection_dict = node_editor.get_sorted_node_connection()

    for node_id_name in node_list:
        has_active_image = node_id_name in node_image_dict
        has_active_result = node_id_name in node_result_dict
        if not has_active_image:
            node_image_dict[node_id_name] = None

        node_id, node_name = node_id_name.split(':')

        if hasattr(node_editor, 'is_node_active'):
            try:
                if not node_editor.is_node_active(node_id_name):
                    continue
            except Exception:
                pass

        connection_refs = sorted_node_connection_dict.get(node_id_name, [])
        connection_refs = [
            connection_info for connection_info in connection_refs
            if _is_valid_connection(connection_info, active_node_set)
        ]
        connection_list = [
            _connection_info_to_update_connection(connection_info)
            for connection_info in connection_refs
        ]

        node_instance = node_editor.get_node_instance(node_name)
        if node_instance is None:
            node_image_dict[node_id_name] = None
            node_result_dict[node_id_name] = None
            continue

        cache_signature = None
        upstream_frame_tokens = []
        use_cache = cache_enabled and (
            len(connection_list) > 0 or cache_source_nodes
        )
        node_setting = {}
        if cache_enabled and hasattr(node_instance, 'get_setting_dict'):
            if mode_async:
                try:
                    node_setting = node_instance.get_setting_dict(node_id)
                except Exception as e:
                    print(
                        'WARNING: failed to read node settings in '
                        f'update_node_info ({node_id_name}) '
                        f'{type(e).__name__}: {e}'
                    )
                    import traceback
                    traceback.print_exc()
                    use_cache = False
            else:
                node_setting = node_instance.get_setting_dict(node_id)

        if use_cache and isinstance(node_setting, dict):
            if node_setting.get('__cache_enabled__') is False:
                use_cache = False

        if (
            cache_enabled and
            isinstance(node_setting, dict) and
            node_setting.get('__cache_source_enabled__') is True
        ):
            use_cache = True

        if use_cache:
            for source_tag, _ in connection_list:
                source_node_id_name = ':'.join(source_tag.split(':')[:2])
                source_result = node_result_dict.get(source_node_id_name)
                frame_token = _extract_frame_token(source_result)
                if frame_token is not None:
                    upstream_frame_tokens.append((source_tag, frame_token))

            cache_signature = _build_node_signature(
                node_id,
                connection_list,
                node_image_dict,
                node_result_dict,
                node_setting,
                node_version_dict=node_version_dict,
            )
            cached_result = node_cache_dict.get(node_id_name)
            if upstream_frame_tokens:
                pipeline_signature = _build_pipeline_signature_for_video(
                    node_id,
                    connection_list,
                    upstream_frame_tokens,
                    node_result_dict,
                    node_setting,
                )
                frame_key = tuple(upstream_frame_tokens)
                if (
                    cached_result is not None and
                    cached_result.get('pipeline_signature') == pipeline_signature
                ):
                    cached_frame_results = cached_result.get('frame_results', {})
                    frame_cached_result = cached_frame_results.get(frame_key)
                    if frame_cached_result is not None:
                        node_image_dict[node_id_name] = copy.deepcopy(
                            frame_cached_result['image']
                        )
                        node_result_dict[node_id_name] = copy.deepcopy(
                            frame_cached_result['result']
                        )
                        rendered_frame_keys = cached_result.setdefault(
                            'rendered_frame_keys', set()
                        )
                        if (
                            frame_key not in rendered_frame_keys and
                            hasattr(node_instance, 'render_cached_output')
                        ):
                            try:
                                node_instance.render_cached_output(
                                    node_id,
                                    node_image_dict[node_id_name],
                                )
                                rendered_frame_keys.add(frame_key)
                            except Exception:
                                pass
                        node_version_dict[node_id_name] = (
                            'video_frame',
                            pipeline_signature,
                            frame_key,
                        )
                        continue
            elif (
                cached_result is not None and
                cached_result.get('signature') == cache_signature
            ):
                # A static cache hit normally means these dictionaries already
                # contain this exact output from the previous tick.  Keep those
                # objects in place rather than copying full-resolution images
                # on every graph traversal.  Restore from the cache only when
                # a caller supplied cache state without the active outputs.
                if not has_active_image:
                    node_image_dict[node_id_name] = copy.deepcopy(
                        cached_result['image']
                    )
                if not has_active_result:
                    node_result_dict[node_id_name] = copy.deepcopy(
                        cached_result['result']
                    )
                if (
                    cached_result.get('rendered_signature') != cache_signature and
                    hasattr(node_instance, 'render_cached_output')
                ):
                    try:
                        node_instance.render_cached_output(
                            node_id,
                            node_image_dict[node_id_name],
                        )
                        cached_result['rendered_signature'] = cache_signature
                    except Exception:
                        pass
                node_version_dict[node_id_name] = cache_signature
                continue

        if mode_async:
            try:
                image, result = node_instance.update(
                    node_id,
                    connection_list,
                    node_image_dict,
                    node_result_dict,
                )
            except Exception as e:
                print(
                    'WARNING: node update exception '
                    f'({node_id_name}) {type(e).__name__}: {e}'
                )
                import traceback
                traceback.print_exc()
                image, result = None, None
        else:
            image, result = node_instance.update(
                node_id,
                connection_list,
                node_image_dict,
                node_result_dict,
            )

        node_image_dict[node_id_name] = copy.deepcopy(image)
        node_result_dict[node_id_name] = copy.deepcopy(result)
        if use_cache:
            if upstream_frame_tokens:
                pipeline_signature = _build_pipeline_signature_for_video(
                    node_id,
                    connection_list,
                    upstream_frame_tokens,
                    node_result_dict,
                    node_setting,
                )
                frame_key = tuple(upstream_frame_tokens)
                cache_entry = node_cache_dict.get(node_id_name, {})
                if cache_entry.get('pipeline_signature') != pipeline_signature:
                    cache_entry = {
                        'pipeline_signature': pipeline_signature,
                        'frame_results': {},
                    }
                cache_entry['frame_results'][frame_key] = {
                    'image': copy.deepcopy(image),
                    'result': copy.deepcopy(result),
                }
                rendered_frame_keys = cache_entry.setdefault(
                    'rendered_frame_keys', set()
                )
                rendered_frame_keys.add(frame_key)
                node_cache_dict[node_id_name] = cache_entry
                node_version_dict[node_id_name] = (
                    'video_frame',
                    pipeline_signature,
                    frame_key,
                )
            else:
                node_cache_dict[node_id_name] = {
                    'signature': cache_signature,
                    'image': copy.deepcopy(image),
                    'result': copy.deepcopy(result),
                    'rendered_signature': cache_signature,
                }
                node_version_dict[node_id_name] = cache_signature
        elif node_id_name in node_cache_dict:
            del node_cache_dict[node_id_name]
            node_version_dict.pop(node_id_name, None)

        if not use_cache:
            frame_token = _extract_frame_token(result)
            if frame_token is not None:
                node_version_dict[node_id_name] = ('frame', frame_token)
            else:
                node_version_dict.pop(node_id_name, None)

    deleted_node_id_name_list = [
        node_id_name for node_id_name in node_cache_dict.keys()
        if node_id_name not in node_list
    ]
    for deleted_node_id_name in deleted_node_id_name_list:
        del node_cache_dict[deleted_node_id_name]
        node_version_dict.pop(deleted_node_id_name, None)

    return
