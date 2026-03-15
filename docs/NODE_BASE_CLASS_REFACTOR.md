# Node base class refactor notes / ノード基底クラス整理メモ

This document is the canonical summary of the Node base-class/helper refactor.
It replaces the previous phase-tracking analysis memo.

このドキュメントはノード基底クラス/ヘルパー整理の正本です。
過去のフェーズ分析メモを置き換えます。

## Scope / 対象

- Shared helper adoption for nodes that still directly inherit `DpgNodeABC`.
- Mechanical standardization of tag construction and connection parsing.

- `DpgNodeABC` を直接継承するノードでの共通ヘルパー利用の統一。
- タグ生成・接続解析の機械的な標準化。

## Shared helper APIs / 共通ヘルパーAPI

The following helpers in `node/node_abc.py` are the source of truth:

- `_node_name(node_id)`
- `_port_tag(node_name, value_type, port_name)`
- `_value_tag(port_tag)`
- `_iter_connections(connection_list)`
- `_extract_source_node_key(source_tag)`
- `_extract_port_name(tag)`

## Phase status / フェーズ状況

- ✅ Phase 1 completed (8/8)
- ✅ Phase 2 completed (10/10)
- ✅ Phase 3 completed (8/8)

All nodes listed in the former phase plan were migrated to the helper-based
pattern while preserving existing behavior/result contracts.

旧フェーズ計画に含まれていたノードは、既存挙動/結果契約を維持したまま、
ヘルパー利用パターンへ移行済みです。

## Current architecture guidance / 現在の設計ガイド

1. Keep tag/connection helpers in `DpgNodeABC` as the single source of truth.
2. Keep `DeclarativeImageProcessNodeBase` focused on simple image-process flow.
3. Prefer incremental, mechanical helper adoption before higher-level base moves.

1. タグ・接続ヘルパーは `DpgNodeABC` を唯一の正として扱う。
2. `DeclarativeImageProcessNodeBase` は単純な画像処理フローに集中させる。
3. 上位ベースへの統合より先に、段階的な機械的リファクタを優先する。

## Candidate future base families / 今後のベース候補

- Source/Capture base
- Model inference base
- Dynamic slot aggregation base
- Sink/output base
- Script execution base

These categories are useful when introducing deeper abstractions beyond helper
normalization.

これらの分類は、ヘルパー統一後にさらに抽象化を進める際の設計指針です。

## Operational notes / 運用メモ

- For code reachable from `Node.update()`, prefer guarded DPG helpers in
  `node_editor/util.py` (`dpg_get_value`, `dpg_set_value`,
  `dpg_get_item_children`) over direct `dpg.get_*` calls.
- Parse UI-driven inputs defensively (`None`/malformed values can occur during
  async delete/import races).

- `Node.update()` から到達する処理では、`dpg.get_*` 直接呼び出しより
  `node_editor/util.py` のガード付きヘルパー利用を優先する。
- UI入力値は防御的にパースする（非同期削除/Import競合で `None` や不正値が起こりうる）。
