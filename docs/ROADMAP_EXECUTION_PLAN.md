# Project Roadmap & Execution Plan

This document consolidates the feature ideas and architecture discussions into a practical, implementation-first plan.

---

## 1) Goals

### Product goals
- Expand core image processing functionality (e.g., sharpen, hue rotation, selective hue adjustment).
- Make graph editing less finicky (connect/reconnect/swap/insert flows).
- Enable parameter animation workflows for still images and videos.
- Add ML-assisted parameter tuning for metric matching and target-image matching.

### Engineering goals
- Reduce coupling in `main.py` by moving runtime update/caching orchestration into a dedicated runtime layer.
- Improve testability and maintainability without forcing broad node API changes.

---

## 2) Guiding Principles

- **Ship in small slices:** avoid long-lived mega-refactors.
- **Behavior-preserving first:** move orchestration before redesigning behavior.
- **Node compatibility:** keep existing node interface stable unless there is clear ROI.
- **Async safety:** preserve guarded access patterns and defensive input parsing.
- **User workflow first:** prioritize friction points in graph editing and parameter workflows.

---

## 3) Phased Plan (Recommended Order)

## Phase A — Runtime refactor (foundation)

### Objective
Extract update/caching orchestration from `main.py` into a dedicated graph runtime module.

### Scope
- Create `node_editor/graph_runtime.py`.
- Move current orchestration responsibilities:
  - node traversal/update scheduling
  - cache signature generation
  - cache hit/miss behavior
  - stale data cleanup for deleted nodes
  - exception handling policy for async update loop integration
- Keep `Node.update(...)` contracts unchanged.
- Keep existing editor UI behavior unchanged.

### Deliverables
- New runtime module and wiring from `main.py`.
- Unit tests for runtime behavior parity.
- No behavior regressions in existing tests.

### Why first
- Reduces future feature risk.
- Makes later graph UX/optimization features easier to implement and test.

---

## Phase B — Graph UX improvements (rewiring quality of life)

### Objective
Make connecting and modifying pipelines faster and less error-prone.

### Candidate features (MVP first)
1. **Auto-reconnect on occupied input**
   - Drop a new source onto an already-connected input to replace old link.
2. **Insert node into link**
   - Select link, choose node type, auto-wire source → new node → destination.
3. **Quick reconnect interaction**
   - Light-weight connect mode for reduced drag precision.
4. **Actionable link feedback**
   - Surface reason for rejected link attempts.

### Stretch goals
- Replace node type in place (implemented as “safe replace + reconnect”).

---

## Phase C — New processing nodes

### Objective
Add practical color and sharpening operations you can immediately use.

### Initial node set
- `Sharpen`
- `Hue Rotate`
- `Selective Hue Adjustment` (aka hue-range adjustment / hue-sector adjustment)

### Implementation approach
- Prefer `DeclarativeImageProcessNodeBase` for consistency and speed.
- Keep parameters linkable so control nodes can drive them.
- Add focused unit tests for processing correctness and parameter bounds.

### Optional expansion
- Advanced sharpen variants (unsharp mask controls).
- Multi-range selective hue controls.

---

## Phase D — Animation & video parameter workflows

### Objective
Support “animate a parameter over time” for still images and existing videos.

### Core features
1. `Frame Index / Time` source node(s)
2. `Parameter Ramp` (or curve) control node
3. Connect control output to process node parameters
4. Export via existing video writer path

### Expected outcomes
- Parameter sweeps (e.g., hue angle increments over frames)
- Repeatable, scriptable animation pipelines in the editor

---

## Phase E — ML tuning nodes

### Objective
Recover/optimize processing parameters from metrics or target images.

### Stage 1 (recommended first)
- Metric-based tuning node
- Objective: optimize scalar metric(s)
- Search methods: random/grid/coarse-to-fine/coordinate descent

### Stage 2
- Target-image matching node
- Objective: minimize output-vs-target distance
- Add optional constraints and warm starts

### Stage 3 (optional)
- Multi-step reconstruction using intermediate saved images
- Step-wise fitting with temporal/sequence regularization

### Risks
- Compute cost grows quickly with parameter dimensionality.
- Need robust stop criteria and iteration budgets.

---

## 4) Backlog Template (without GitHub Issues)

Use this template in local markdown files to track work:

```md
## [ID] Title
- Status: Todo / In Progress / Done
- Priority: P0 / P1 / P2
- Estimate: S / M / L
- Depends on: [IDs]

### Goal
...

### Acceptance Criteria
- [ ] ...
- [ ] ...

### Technical Notes
...

### Test Plan
- [ ] ...
```

---

## 5) Suggested Execution Backlog

## Epic A: Runtime foundation

### A1. Extract graph runtime module
- Priority: P0
- Estimate: M
- Depends on: none
- Acceptance:
  - [ ] Runtime logic moved out of `main.py`
  - [ ] Existing update behavior preserved
  - [ ] No node API changes required

### A2. Runtime behavior parity tests
- Priority: P0
- Estimate: M
- Depends on: A1
- Acceptance:
  - [ ] Cache hit/miss parity tests pass
  - [ ] Deleted-node cleanup tested
  - [ ] Async exception behavior tested

### A3. Runtime policy switches (optional)
- Priority: P1
- Estimate: S
- Depends on: A1
- Acceptance:
  - [ ] Configurable cache enable/disable
  - [ ] Source-node cache policy explicit

---

## Epic B: Graph UX

### B1. Replace existing link on occupied input
- Priority: P0
- Estimate: M
- Depends on: A1 preferred

### B2. Insert-node-into-link action
- Priority: P1
- Estimate: M
- Depends on: B1

### B3. Rejected-link reason feedback
- Priority: P1
- Estimate: S
- Depends on: none

### B4. Safe node-type replace workflow
- Priority: P2
- Estimate: L
- Depends on: B1

---

## Epic C: Processing nodes

### C1. Sharpen node
- Priority: P0
- Estimate: S
- Depends on: none

### C2. Hue Rotate node
- Priority: P0
- Estimate: S
- Depends on: none

### C3. Selective Hue Adjustment node
- Priority: P1
- Estimate: M
- Depends on: C2

### C4. Regression tests for new process nodes
- Priority: P0
- Estimate: M
- Depends on: C1, C2, C3

---

## Epic D: Animation

### D1. Frame Index/Time control node
- Priority: P1
- Estimate: S
- Depends on: none

### D2. Parameter Ramp node
- Priority: P1
- Estimate: M
- Depends on: D1

### D3. End-to-end animation pipeline sample
- Priority: P1
- Estimate: S
- Depends on: D2

---

## Epic E: ML tuning

### E1. Metric optimizer node (v1)
- Priority: P2
- Estimate: L
- Depends on: A1 preferred

### E2. Target image matching node (v1)
- Priority: P2
- Estimate: L
- Depends on: E1

### E3. Multi-step reconstruction workflow
- Priority: P3
- Estimate: XL
- Depends on: E2

---

## 6) Milestone plan (example)

### Milestone 1 (2–3 weeks)
- A1, A2, B1, C1, C2

### Milestone 2 (2–3 weeks)
- B2, B3, C3, C4, D1

### Milestone 3 (3–4 weeks)
- D2, D3, E1 prototype

### Milestone 4 (R&D)
- E2, E3 exploration

---

## 7) Definition of Done

A task is done when:
- Code is merged with tests (or explicit rationale for no test).
- Behavior is documented if user-facing.
- Async safety considerations are reviewed for affected update paths.
- Import/export compatibility impact is checked.

---

## 8) Practical next step (immediate)

Start with **A1 (runtime extraction)** and **C1/C2 (sharpen + hue rotate)** in parallel branches:
- Branch 1: architecture foundation
- Branch 2: user-visible feature delivery

This balances long-term maintainability and near-term feature momentum.
