#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import OrderedDict
import os

from node_editor.node_editor import DpgNodeEditor


def build_default_menu_dict():
    return OrderedDict({
        'InputNode': 'input_node',
        'ProcessNode': 'process_node',
        # 'DeepLearningNode': 'deep_learning_node',
        'AnalysisNode': 'analysis_node',
        'DrawNode': 'draw_node',
        'OtherNode': 'other_node',
        # 'PreviewReleaseNode': 'preview_release_node'
    })


def create_node_editor(current_path, editor_width, editor_height,
                       opencv_setting_dict, use_debug_print=False,
                       menu_dict=None):
    if menu_dict is None:
        menu_dict = build_default_menu_dict()

    return DpgNodeEditor(
        width=editor_width - 15,
        height=editor_height - 40,
        opencv_setting_dict=opencv_setting_dict,
        menu_dict=menu_dict,
        use_debug_print=use_debug_print,
        node_dir=os.path.join(current_path, 'node'),
    )


def import_startup_json(node_editor, import_json):
    if import_json is None:
        return

    print('**** Import JSON ********')
    try:
        node_editor.import_setting_file(import_json)
    except Exception as e:
        print('ERROR: failed to import startup JSON file')
        print(f'\tpath                 : {import_json}')
        print(f'\terror                : {type(e).__name__}: {e}')
        import traceback
        traceback.print_exc()
        print()
