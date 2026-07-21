from node.node_abc import DpgNodeABC, DpgNodeBase
from node.port_model import LinkConnectionAdapter, LinkRef, NodeRef, PortRef
from node.input_node.node_int_value import Node as IntValueNode
from node.process_node.node_blur import Node as BlurNode


def test_dpg_node_base_is_concrete_helper_subclass():
    assert issubclass(DpgNodeBase, DpgNodeABC)


def test_abstract_interface_no_longer_owns_concrete_helpers():
    for helper_name in (
        '_node_name',
        '_port_tag',
        '_value_tag',
        '_iter_connections',
        'add_editor_toolbar',
    ):
        assert helper_name not in DpgNodeABC.__dict__
        assert helper_name in DpgNodeBase.__dict__


def test_representative_direct_nodes_inherit_concrete_base():
    assert isinstance(IntValueNode(), DpgNodeBase)
    assert isinstance(BlurNode(), DpgNodeBase)


def test_concrete_base_preserves_existing_tag_helpers():
    node = IntValueNode()
    node_name = node._node_name(3)
    port_tag = node._port_tag(node_name, node.TYPE_INT, 'Output01')

    assert node_name == '3:IntValue'
    assert port_tag == '3:IntValue:Int:Output01'
    assert node._value_tag(port_tag) == '3:IntValue:Int:Output01Value'


def test_concrete_base_node_tag_helpers_compose_port_and_value_tags():
    node = IntValueNode()

    assert node._node_port_tag(3, node.TYPE_INT, 'Output01') == (
        '3:IntValue:Int:Output01'
    )
    assert node._port_value_tag('3:IntValue', node.TYPE_INT, 'Output01') == (
        '3:IntValue:Int:Output01Value'
    )
    assert node._node_value_tag(3, node.TYPE_INT, 'Output01') == (
        '3:IntValue:Int:Output01Value'
    )



def test_concrete_base_control_tag_helpers_mark_non_graph_alias_boundary():
    node = IntValueNode()

    assert node._control_tag('3:IntValue', node.TYPE_TEXT, 'Toggle') == (
        '3:IntValue:Text:Toggle'
    )
    assert node._control_value_tag('3:IntValue', node.TYPE_TEXT, 'Toggle') == (
        '3:IntValue:Text:ToggleValue'
    )
    assert node._node_control_tag(3, node.TYPE_TEXT, 'Toggle') == (
        '3:IntValue:Text:Toggle'
    )
    assert node._node_control_value_tag(3, node.TYPE_TEXT, 'Toggle') == (
        '3:IntValue:Text:ToggleValue'
    )

def test_concrete_base_preserves_connection_iteration_guards():
    node = IntValueNode()

    assert list(node._iter_connections([
        ['1:Src:Int:Output01', '2:IntValue:Int:Input01'],
        ['malformed-source', '2:IntValue:Int:Input01'],
        ['1:Src:Int:Output02'],
        [None, '2:IntValue:Int:Input02'],
    ])) == [
        ('1:Src:Int:Output01', '2:IntValue:Int:Input01', 'Int'),
    ]


def test_concrete_base_iter_connections_accepts_typed_link_adapter():
    node = IntValueNode()
    link_ref = LinkRef(
        PortRef(
            node_ref=NodeRef('1', 'Src'),
            direction='Output',
            data_type='Int',
            index=1,
            port_name='Output01',
            dpg_tag='1:Src:Int:Output01',
        ),
        PortRef(
            node_ref=NodeRef('2', 'IntValue'),
            direction='Input',
            data_type='Int',
            index=1,
            port_name='Input01',
            dpg_tag='2:IntValue:Int:Input01',
        ),
    )

    assert list(node._iter_connections([LinkConnectionAdapter(link_ref)])) == [
        ('1:Src:Int:Output01', '2:IntValue:Int:Input01', 'Int'),
    ]


def test_concrete_base_iter_connection_infos_exposes_typed_adapter():
    node = IntValueNode()
    link_ref = LinkRef(
        PortRef(
            node_ref=NodeRef('1', 'Src'),
            direction='Output',
            data_type='Int',
            index=1,
            port_name='Output01',
            dpg_tag='1:Src:Int:Output01',
        ),
        PortRef(
            node_ref=NodeRef('2', 'IntValue'),
            direction='Input',
            data_type='Int',
            index=1,
            port_name='Input01',
            dpg_tag='2:IntValue:Int:Input01',
        ),
    )
    adapter = LinkConnectionAdapter(link_ref)

    assert list(node._iter_connection_infos([adapter])) == [
        (adapter, '1:Src:Int:Output01', '2:IntValue:Int:Input01', 'Int'),
    ]


def test_concrete_base_iter_connections_tags_preserve_typed_metadata():
    node = IntValueNode()
    source = PortRef(
        node_ref=NodeRef('1', 'Src'),
        direction='Output',
        data_type='Int',
        index=1,
        port_name='Output01',
        dpg_tag='legacy:source:Int:Output99',
        value_tag='1:Src:Int:Output01Value',
    )
    destination = PortRef(
        node_ref=NodeRef('2', 'IntValue'),
        direction='Input',
        data_type='Int',
        index=1,
        port_name='Input01',
        dpg_tag='legacy:dest:Int:Input99',
        value_tag='2:IntValue:Int:Input01Value',
    )
    source_tag, destination_tag, connection_type = next(
        node._iter_connections([LinkConnectionAdapter(LinkRef(source, destination))])
    )

    assert connection_type == 'Int'
    assert str(source_tag) == 'legacy:source:Int:Output99'
    assert node._extract_source_node_key(source_tag) == '1:Src'
    assert node._extract_node_id(source_tag) == '1'
    assert node._extract_port_name(destination_tag) == 'Input01'
    assert node._value_tag(source_tag) == '1:Src:Int:Output01Value'


def test_direct_node_add_node_graph_attributes_use_port_declarations():
    import re
    from pathlib import Path

    node_root = Path(__file__).parents[2] / 'node'
    failures = []
    for path in sorted(node_root.rglob('*')):
        if not (path.name.endswith('.py') or path.name.endswith('.py.disable')):
            continue
        source = path.read_text(encoding='utf-8')
        if 'class Node(DpgNodeBase)' not in source:
            continue
        add_node_match = re.search(r'(?m)^    def add_node\(', source)
        if add_node_match is None:
            continue
        next_method_match = re.search(
            r'(?m)^    def \w+\(',
            source[add_node_match.end():],
        )
        if next_method_match is None:
            add_node_source = source[add_node_match.start():]
        else:
            add_node_source = source[
                add_node_match.start():add_node_match.end() + next_method_match.start()
            ]
        declared_tag_variables = {
            declaration.group(1)
            for declaration in re.finditer(
                r'(\w+)_port = self\.(?:input_port|output_port)\(',
                add_node_source,
            )
        }
        declared_tag_variables.update(
            declaration.group(1)
            for declaration in re.finditer(
                r'(\w+)_port = ports\.\w+',
                add_node_source,
            )
        )
        lines = add_node_source.splitlines()
        index = 0
        while index < len(lines):
            if 'with dpg.node_attribute(' not in lines[index]:
                index += 1
                continue
            call_lines = [lines[index]]
            if '):' not in lines[index]:
                index += 1
                while index < len(lines):
                    call_lines.append(lines[index])
                    if lines[index].strip() == '):':
                        break
                    index += 1
            call_source = '\n'.join(call_lines)
            tag_match = re.search(r'tag=(\w+)', call_source)
            attr_match = re.search(
                r'attribute_type=dpg\.mvNode_Attr_(Input|Output)',
                call_source,
            )
            if (
                tag_match is not None
                and attr_match is not None
                and tag_match.group(1) not in declared_tag_variables
            ):
                failures.append(
                    f'{path.relative_to(node_root.parent)}: '
                    f'{attr_match.group(1)} {tag_match.group(1)}'
                )
            index += 1

    assert failures == []


def test_direct_node_updates_use_typed_connection_info_iteration():
    from pathlib import Path

    node_root = Path(__file__).parents[2] / 'node'
    failures = []
    for path in sorted(node_root.rglob('*')):
        if path.name == 'node_abc.py':
            continue
        if not (path.name.endswith('.py') or path.name.endswith('.py.disable')):
            continue
        source = path.read_text(encoding='utf-8')
        if 'class Node(DpgNodeBase)' not in source:
            continue
        if 'self._iter_connections(' in source:
            failures.append(str(path.relative_to(node_root.parent)))

    assert failures == []


def test_curve_points_port_data_type_round_trips():
    from node.port_model import PortDataType
    from node.port_serialization import port_ref_from_tag

    port = port_ref_from_tag('1:CurvesPoints:CurvePoints:Output01')

    assert port.data_type is PortDataType.CURVE_POINTS
    assert port.value_tag == '1:CurvesPoints:CurvePoints:Output01Value'


def test_curve_points_dragging_dynamic_point_outside_deletes_it(monkeypatch):
    from node import curves_points_ui as curves_ui
    from node.curves_points_ui import CurvesPointsEditorMixin

    class TestEditor(CurvesPointsEditorMixin):
        def __init__(self):
            self.deleted = []
            self.changed = []
            self.emitted = []
            self.points = [[0, 0], [128, 128], [255, 255]]

        def _node_name(self, node_id):
            return f'{node_id}:TestCurves'

        def _get_drag_points(self, node_id):
            del node_id
            return self.points

        def _redraw_line(self, node_id):
            del node_id

        def _on_points_changed(self, node_id, points):
            self.changed.append((node_id, points))

        def _emit_points_changed(self, node_id, before_points, after_points, coalesce=False):
            self.emitted.append((node_id, before_points, after_points, coalesce))

    editor = TestEditor()
    monkeypatch.setattr(curves_ui, 'dpg_get_value', lambda tag: [-1, 128])
    monkeypatch.setattr(
        curves_ui.dpg,
        'delete_item',
        lambda tag: editor.deleted.append(tag),
    )

    editor._callback_moved_point('point-tag', None, (7, None))

    assert editor.deleted == ['point-tag']
    expected_curve_set = {
        'White': editor.points,
        'Red': [[0, 0], [255, 255]],
        'Green': [[0, 0], [255, 255]],
        'Blue': [[0, 0], [255, 255]],
    }
    assert editor.changed == [(7, expected_curve_set)]
    assert editor.emitted == [
        (7, editor.points, editor.points, False),
    ]


def test_curve_points_clear_channel_and_all_reset_curve_sets():
    from node.curves_points_ui import CurvesPointsEditorMixin

    class TestEditor(CurvesPointsEditorMixin):
        def __init__(self):
            self.changed = []

        def _node_name(self, node_id):
            return f'{node_id}:TestCurves'

        def _on_points_changed(self, node_id, curve_set):
            self.changed.append((node_id, curve_set))

    editor = TestEditor()
    editor._set_active_channel(7, 'Red')
    editor._set_curve_set(7, {
        'White': [[0, 0], [255, 100]],
        'Red': [[0, 0], [255, 50]],
        'Green': [[0, 0], [255, 80]],
        'Blue': [[0, 0], [255, 30]],
    })

    editor._callback_clear_channel(None, None, 7)

    assert editor._curve_set(7)['White'] == [[0.0, 0.0], [255.0, 100.0]]
    assert editor._curve_set(7)['Red'] == [[0, 0], [255, 255]]

    editor._callback_clear_all(None, None, 7)

    assert editor._curve_set(7) == editor._default_curve_set()
    assert editor.changed[-1] == (7, editor._default_curve_set())


def test_curve_points_editor_defaults_to_compact_canvas_with_large_mode():
    from node.curves_points_ui import CurvesPointsEditorMixin

    editor = CurvesPointsEditorMixin()

    assert editor._plot_width == 240
    assert editor._plot_height == 180
    assert editor._large_plot_width >= 800
    assert editor._large_plot_height >= 600
    assert editor._axis_padding > 0
    assert editor._keyboard_nudge_step <= 0.1


def test_curve_points_keyboard_nudges_last_touched_point(monkeypatch):
    from node import curves_points_ui as curves_ui
    from node.curves_points_ui import CurvesPointsEditorMixin

    class TestEditor(CurvesPointsEditorMixin):
        def __init__(self):
            self.values = {'point-tag': [128.0, 128.0]}
            self.changed = []
            self.emitted = []
            self._last_touched_point_by_node = {'7': 'point-tag'}

        def _node_name(self, node_id):
            return f'{node_id}:TestCurves'

        def _get_drag_points(self, node_id):
            del node_id
            return [self.values['point-tag']]

        def _redraw_line(self, node_id):
            del node_id

        def _on_points_changed(self, node_id, points):
            self.changed.append((node_id, points))

        def _emit_points_changed(self, node_id, before_points, after_points, coalesce=False):
            self.emitted.append((node_id, before_points, after_points, coalesce))

    editor = TestEditor()
    monkeypatch.setattr(curves_ui.dpg, 'does_item_exist', lambda tag: tag == 'point-tag')
    monkeypatch.setattr(curves_ui.dpg, 'get_item_user_data', lambda tag: (7, None))
    monkeypatch.setattr(curves_ui, 'dpg_get_value', lambda tag: editor.values[tag])
    monkeypatch.setattr(
        curves_ui.dpg,
        'set_value',
        lambda tag, value: editor.values.__setitem__(tag, value),
    )

    editor._callback_nudge_last_point(None, None, (7, 0.1, -0.1))

    assert editor.values['point-tag'] == [128.1, 127.9]
    assert editor.changed[-1][0] == 7
    assert editor.emitted[-1][3] is False


def test_curve_points_keyboard_handlers_use_global_handler_registry():
    from pathlib import Path

    source = Path('node/curves_points_ui.py').read_text(encoding='utf-8')
    key_handler_block = source.split('dpg.add_key_press_handler', maxsplit=1)[1]

    assert 'with dpg.handler_registry()' in source
    assert 'parent=handler' not in key_handler_block.split(')', maxsplit=1)[0]


def test_curve_points_spline_line_samples_lut_without_extra_drag_points():
    from node.curves_points_ui import CurvesPointsEditorMixin

    editor = CurvesPointsEditorMixin()
    points = [[0, 0], [64, 230], [128, 40], [255, 255]]

    x_values, y_values = editor._line_values(points, interpolation='spline')

    assert list(x_values) == list(range(256))
    assert len(y_values) == 256
    assert [x for x, _y in points] == [0, 64, 128, 255]


def test_curve_points_linear_line_uses_only_drag_points():
    from node.curves_points_ui import CurvesPointsEditorMixin

    editor = CurvesPointsEditorMixin()
    points = [[0, 0], [64, 230], [128, 40], [255, 255]]

    x_values, y_values = editor._line_values(points, interpolation='linear')

    assert list(x_values) == [0, 64, 128, 255]
    assert list(y_values) == [0, 230, 40, 255]


def test_curve_points_large_editor_toggle_changes_plot_size(monkeypatch):
    from node import curves_points_ui as curves_ui
    from node.curves_points_ui import CurvesPointsEditorMixin

    class TestEditor(CurvesPointsEditorMixin):
        def _node_name(self, node_id):
            return f'{node_id}:TestCurves'

    configured = {}
    editor = TestEditor()
    monkeypatch.setattr(curves_ui.dpg, 'does_item_exist', lambda tag: True)
    monkeypatch.setattr(
        curves_ui.dpg,
        'configure_item',
        lambda tag, **kwargs: configured.update({tag: kwargs}),
    )

    editor._callback_toggle_large_editor(None, None, 7)

    assert configured['7:TestCurves:plot'] == {'width': 900, 'height': 650}
    assert editor._plot_size(7) == (900, 650)

    editor._callback_toggle_large_editor(None, None, 7)

    assert configured['7:TestCurves:plot'] == {'width': 240, 'height': 180}
    assert editor._plot_size(7) == (240, 180)
