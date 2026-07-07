# Node Parameter Auto-Tuning Design

This document describes a staged approach for matching a source photo to a target photo by automatically tuning the parameters of selected process nodes. The first implementation should be small, but the architecture should keep room for multi-node tuning and richer objective functions.

## Goal

Given:

- a source image,
- a target image, and
- one or more process nodes explicitly selected by the user,

find parameter values that minimize the visual difference between the processed source image and the target image. The tuner does not initially need to decide which nodes to use; the user chooses the candidate node or short selected chain.

## Recommended starting point

Start with a general-purpose **Auto Tune node plus optimizer service**, not an `Auto Tune` button copied into every supported process node. A toolbar/menu action can still be useful later, but the reusable primitive should be a node that receives images and emits tuned parameter values.

The first implementation should tune one downstream process node, with Gaussian Blur as the first target because it has a small search space and clear parameter bounds. The same architecture should then scale to multiple connected process nodes by adding more parameter outputs and evaluating the selected subgraph as a chain.

The minimum viable flow is:

1. User adds an Auto Tune node.
2. User connects the source image and target image to the Auto Tune node.
3. User connects Auto Tune parameter outputs to parameter input ports on the process node being tuned, for example Gaussian Blur `kernel_size` and `sigma`.
4. Auto Tune evaluates candidate parameter sets by running the selected process node or selected process chain on the source image.
5. Auto Tune scores each processed output against the target image.
6. Auto Tune writes the best values to its output ports, which drive the connected process-node parameter inputs.
7. Optionally, the user can commit the emitted values into the tuned node's own widget settings so the graph no longer depends on the Auto Tune node.

## Core abstraction

Add an auto-tuning service that is independent of DearPyGui widgets wherever possible. The Auto Tune node should be a UI/graph wrapper around this service, while the service itself only needs images, a parameter schema, a way to evaluate candidate parameters, and a metric:

```text
AutoTuneRequest
  source_image: np.ndarray
  target_image: np.ndarray
  evaluation_plan: EvaluationPlan
  tunable_parameters: list[TunableParameter]
  metric: ObjectiveMetric
  budget: EvaluationBudget

AutoTuneResult
  best_parameters: dict[node_id, dict[parameter_name, value]]
  best_score: float
  evaluations: int
  preview_image: np.ndarray
```

For most process nodes, the Auto Tune node can infer tunable metadata from the existing parameter input ports and declarative `parameters` entries. This means each process node does not need its own Auto Tune button or custom UI. Nodes only need extra code when their tunable state is not represented as normal parameter ports, such as Curves drag points.

A suggested `TunableParameter` shape is:

```text
name: string
kind: int | float | bool | categorical | custom
min_value: optional number
max_value: optional number
step: optional number
choices: optional list
scale: linear | log
port_ref: optional graph output/input mapping
apply: callback for custom non-port controls
```

For declarative nodes, this metadata can usually come from `name`, `type`, `min`, `max`, `items`, `default`, and `cast`. The graph connection from an Auto Tune output to a process node's parameter input identifies which parameters are actively tuned. Custom controls such as Curves can provide an adapter that converts a compact parameter vector into node settings and, if needed, exposes a serialized setting output/input rather than a simple scalar port.

Important limitation: the parameter ports themselves identify type and graph connectivity, but they should not be treated as the only source of tuning bounds. The optimizer also needs each parameter's domain, such as slider min/max, allowed combo choices, integer-vs-float stepping, and node-specific constraints like odd Gaussian kernels. For existing declarative nodes, those bounds already live in the node `parameters` definitions, so the tuner should combine port connectivity with node metadata. For custom nodes or external parameter sources, the UI must ask the user for missing bounds before running.


## Auto Tune node graph model

The Auto Tune node should behave like a controller node:

- **Inputs**
  - `source_image`: the original image before the selected process node or chain.
  - `target_image`: the image to match.
  - optional `run` / `enabled` / `budget` controls.
- **Outputs**
  - one output per tunable parameter, such as `kernel_size`, `sigma`, or `hue_shift_degrees`; these connect to existing process-node parameter input ports.
  - `best_score` and optional `preview_image` outputs for debugging.

A simple MVP can let the user add outputs manually by choosing from a list of connected/selected parameter ports. A later UI can inspect the selected node or selected chain and auto-create matching Auto Tune outputs.

The important point is that the Auto Tune node should not need to be embedded into every process node. Existing parameter ports already provide the control surface for many nodes. The process node stays responsible for processing; the Auto Tune node stays responsible for search.

The tuner should **not** tune by repeatedly writing output values into the live graph and waiting for normal graph propagation to produce a new processed image. That would create a feedback-loop problem and would be slow/non-deterministic because the optimizer would depend on GUI update timing. Instead, the Auto Tune node should use the live graph only to discover the selected process node/chain and current fixed parameter values. During a tuning run, the optimizer should evaluate candidates internally by calling the selected nodes' pure processing functions against an in-memory copy of the source image. Only after the best candidate is found should the Auto Tune node publish output values to the graph.

Because candidate evaluation is internal, the Auto Tune node does not need an exemption from the normal no-loops rule. Its parameter outputs are final/best-so-far control values, not part of the image path being scored for each candidate.

## How multi-node co-tuning works

Co-tuning multiple nodes requires an `EvaluationPlan` that represents the chosen process chain and the parameter ports controlled by Auto Tune outputs:

```text
EvaluationPlan
  source_node_or_image
  process_steps:
    - node_id: GaussianBlur
      parameters:
        kernel_size: TunableParameter(output_port=A)
        sigma: TunableParameter(output_port=B)
    - node_id: HueRotation
      parameters:
        hue_shift_degrees: TunableParameter(output_port=C)
  target_image
```

For each candidate vector, the optimizer:

1. maps vector values to the connected/tuned parameter names,
2. fills every untuned parameter in the chain from the node's current fixed settings,
3. runs the first process node on the source image,
4. passes that output into each subsequent process node,
5. repeats until the endpoint, and
6. scores the final output against the target.

This is joint optimization when the vector contains parameters from more than one node. For example, Gaussian Blur `kernel_size`, Gaussian Blur `sigma`, and Hue Rotation `hue_shift_degrees` can be searched together. To keep the search tractable, the UI should default to a small budget and use staged strategies for larger chains: tune one node at a time, keep the best candidates, then do a short joint refinement pass.

The chain should be explicit. The Auto Tune node does not need to infer which process nodes to use; it only needs to evaluate the node or chain selected by the user, or the process nodes between its source input and the final preview/result chosen as the tuning endpoint.

## Subgraph discovery and validation

Normal usage should let the user wire an Auto Tune node into an existing graph without manually rebuilding a hidden graph. For example:

```text
Source -> Gaussian Blur -> Curves -> Result
   \-> Auto Tune.source_image
Target -> Auto Tune.target_image
Auto Tune.kernel_size -> Gaussian Blur.kernel_size
Auto Tune.points -> Curves.points
```

When the user clicks `Run`, the editor-owned evaluator should discover the evaluation chain from the Auto Tune connections:

1. Identify the source image producer connected to `Auto Tune.source_image`.
2. Identify every process-node parameter port controlled by Auto Tune outputs.
3. Find the process node that owns each controlled parameter.
4. Require every controlled process node to be downstream of the source image producer through image links.
5. Choose the scoring endpoint as either an explicit `processed_image`/`endpoint` input on Auto Tune, a user-selected result node, or the lowest common downstream image node that contains all controlled nodes.
6. Build an ordered chain from the source image to that endpoint and evaluate only that chain internally.

The first implementation should avoid guessing when multiple valid endpoints exist. In ambiguous cases, show a validation error and ask the user to select the endpoint/result image explicitly.

### Invalid wiring cases

The tuner should validate wiring before starting optimization and fail fast with user-facing messages:

- **Controlled node is not downstream of source:** error, because changing that parameter cannot affect the source-to-target output being scored.
- **Controlled node is upstream of the source tap:** error, because the selected source image already includes or bypasses that node's output, making the candidate evaluation ill-defined.
- **Controlled node is on a different disconnected branch:** error, because there is no image path from the Auto Tune source to that node.
- **Controlled nodes do not share a single downstream endpoint:** ask the user to select an endpoint or split the tuning run.
- **Controlled parameter has no domain metadata:** ask for bounds/choices before running.
- **Selected chain contains unsupported nodes:** either freeze those nodes if their current output can be treated as fixed, or report that the chain cannot be evaluated internally yet.

A useful error message should name the offending connection and explain the requirement, for example: `Auto Tune output 'kernel_size' is connected to GaussianBlur#7, but GaussianBlur#7 is not downstream of the selected source image.`

### Endpoint selection

For the MVP, add an explicit Auto Tune input or picker for the processed endpoint to avoid ambiguous graph traversal:

```text
source_image: original/source image before tuned nodes
target_image: target image to match
endpoint_node: result/process node whose output should be compared to target
parameter_outputs: values to tune
```

With an explicit endpoint, validation is straightforward: each tuned node must lie on at least one image path from source to endpoint, and the evaluator runs the ordered subgraph between source and endpoint. Later, endpoint auto-detection can be added as a convenience.

All supported process nodes on the source-to-endpoint path should be included in the internal evaluation chain, even if their parameters are not being tuned. Untuned intermediate nodes run with their current fixed settings, while only parameters controlled by Auto Tune outputs vary between candidates. For example, if the path is `Source -> Gaussian Blur -> Curves -> Result` and Auto Tune is only connected to `Gaussian Blur.kernel_size`, Curves still runs during evaluation using its current curve points. If an untuned intermediate node cannot be evaluated internally yet, the tuner should report that the source-to-endpoint chain contains an unsupported node rather than silently skipping it.

## Diff and metric placement

A separate Diff node can still be useful for visualization, for example showing `abs(processed_source - target)` to help users understand what the tuner is optimizing. However, the optimizer should own the scoring metric internally rather than requiring a Diff node as input.

Reasons to keep the metric inside the tuner:

- the tuner needs a scalar objective, while a diff image is only an intermediate representation;
- different tuning tasks need different reductions, such as MSE, MAE, luminance-only error, SSIM, masked error, or color-space-specific error;
- internal metrics avoid adding extra graph cycles or requiring candidate images to propagate through visible nodes;
- the same source/target images can be scored several ways without rewiring the graph.

A good compromise is to make `preview_image` or a paired Diff/Viewer node optional output UI: the tuner computes the best candidate internally, emits the best processed image or diff image for display, and still keeps optimization self-contained.

## Objective metrics

Use a simple metric first, then make it pluggable. The Gaussian Blur tuner now
ships with three metrics because different recovery workflows need different
assumptions:

- `mse`: direct pixel mean-squared error. This is best when the target is
  expected to be only the blurred source image.
- `smoothness`: compares one whole-image luminance gradient-energy value for
  the candidate against one value for the target. This is useful for quick blur
  strength matching, but it can be fooled when later operations such as curves
  or solarization change global contrast.
- `local_smoothness`: compares per-pixel neighboring-gradient-energy maps. This
  is the default and preferred blur-recovery metric when the target may have had
  a curve or solarizing curve applied after blur, because it ignores exact color
  equality while still preserving where local contrast remains.

Future metrics can still be added for other node families:

- luminance-only or color-space-specific scoring for color adjustments,
- structural similarity (SSIM) for perceptual structure matching,
- masked scoring so users can tune only a region of interest,
- node-specific metrics where a generic pixel metric is the wrong objective.

Always normalize source and target before scoring:

- same dimensions,
- same number of channels,
- optional downscale for speed,
- optional conversion to a stable color space for color-focused nodes.

## Initial candidate and search seeding

The default starting point should be the current settings of the process nodes in the source-to-endpoint chain. This makes the workflow natural: the user manually gets the graph close, then clicks `Run` to let Auto Tune refine from there. The tuner should read those settings through formal node/editor APIs, not by ad-hoc widget inspection, and use them as:

- the baseline score shown before optimization,
- the center of local/coarse-to-fine searches,
- fixed values for parameters that are not connected to Auto Tune outputs, and
- fallback values if a tuned parameter's output is disconnected or optimization is cancelled.

The Auto Tune node can later expose optional seed controls for advanced use cases:

- **Use current settings**: default and recommended.
- **Use Auto Tune output values**: resume from the last best result.
- **Use defaults**: start from each node's declared default values.
- **Use random samples around current**: useful for high-dimensional color adjustments.

For the MVP, implement only `Use current settings`. That is enough for Gaussian Blur and avoids introducing confusing optimizer controls before the basic workflow is proven.

## Optimizer strategy

Use different search strategies by parameter count:

1. **Gaussian Blur:** deterministic grid or coarse-to-fine search. Force odd kernel sizes and include `auto_sigma` as a fixed or categorical choice.
2. **Hue Rotation:** one-dimensional grid over hue shift, optionally repeated for each color-space choice.
3. **Hue/Saturation Adjustment:** coordinate descent or random search with bounds; begin with only global blend and one or two active bands.
4. **Curves:** start with a low-dimensional curve model, such as three or five fixed x positions with optimizable y values. Keep endpoints fixed unless the user opts in.
5. **Short chains:** optimize all exposed parameters jointly only for small spaces; otherwise run staged coordinate descent node-by-node, then refine jointly.

A practical default is:

```text
coarse grid/random samples -> keep top K -> local coordinate refinement -> apply best result
```

This avoids pulling in heavy optimization dependencies and works well with discrete UI sliders.

## Node-specific notes

### Gaussian Blur

Tune `kernel_size` and `sigma`. Because the processing function already corrects even kernels to odd kernels, the tuner should still generate odd kernel sizes directly so displayed settings match evaluated settings.

Recommended first search:

- `kernel_size`: odd integers from 1 to a capped maximum, such as 101 for the first pass.
- `auto_sigma`: either fixed to true for the first MVP, or treated as a categorical parameter.
- `sigma`: only tune when `auto_sigma` is false.

### Curves

Curves needs a parameter input before it can participate cleanly in graph-native tuning, because its current control surface is custom drag points rather than ordinary scalar parameter ports. Add a serialized `points` parameter input/output shape, or a small set of scalar point-y inputs, before treating it like the simple declarative nodes.

Do not optimize arbitrary drag points initially. Use a compact editable representation:

- endpoints fixed at `(0, 0)` and `(255, 255)`,
- three interior x positions, such as 64, 128, and 192,
- y values optimized within 0-255.

The cleanest first graph representation is one custom `points` parameter carrying the list of points. If generic port types make that awkward, use fixed scalar ports such as `y64`, `y128`, and `y192`, then adapt those values back into the Curves point list internally. Later, allow user-selected point counts and per-channel curves.

### Hue Rotation

Treat `hue_shift_degrees` as a bounded integer parameter and `color_space` as categorical. For an MVP, tune HSV only; then evaluate all supported color spaces and choose the best pair.

### Hue/Saturation Adjustment

The full parameter set is large. Start by tuning only:

- `blend`,
- the dominant target/source hue band's hue shift,
- the dominant target/source hue band's saturation.

Then add multi-band coordinate descent.

## UI integration

A small first UI should be explicit and graph-native:

1. User adds an Auto Tune node.
2. User connects source and target image inputs.
3. User selects or connects the processed endpoint whose output should be compared to the target.
4. User selects one process node, or a short ordered chain of process nodes.
5. Auto Tune lists tunable parameter ports discovered from the source-to-endpoint path.
6. User chooses which parameters to tune; Auto Tune creates matching output ports.
7. User connects those outputs to the selected process-node parameter inputs, or accepts an auto-wire option.
8. User clicks `Run` on the Auto Tune node.
9. Auto Tune validates that each controlled parameter belongs to a node on the source-to-endpoint path.
10. Auto Tune emits the best values and optionally offers `Bake to nodes` to copy the best values into the target node widgets/settings.

A separate context-menu command like `Create Auto Tune for selection` can automate steps 1-6, but the underlying implementation remains a reusable tuning node rather than per-node button code.

## Implementation phases

### Phase 1: service and Gaussian Blur MVP

- Create an `auto_tune` module with request/result dataclasses, image normalization, metric functions, an `EvaluationPlan`, and a small grid-search optimizer.
- Add an Auto Tune node with `source_image`, `target_image`, an explicit endpoint selector/input, `best_score`, and dynamically-created parameter output ports.
- Add source-to-endpoint subgraph discovery and validation for connected parameter targets.
- Infer tunable slider/combo metadata from declarative node parameter definitions and connected parameter ports; require user-specified bounds when metadata is missing.
- Add Gaussian Blur-specific normalization that searches odd kernel sizes and handles `auto_sigma`.
- Seed the initial candidate and fixed untuned parameters from the current node settings.
- Evaluate candidates internally by calling the selected node/chain processing functions, then emit only the best values through Auto Tune output ports. Add an optional `Bake to nodes` action that writes final settings through existing setting APIs so undo/redo and serialization remain consistent.

### Phase 2: generic declarative node support

- Support integer sliders, float sliders, checkboxes, and combos generically through parameter-port connections.
- Add optimizer policies based on parameter count.
- Add previews and cancellable progress reporting.

### Phase 3: custom tuners for Curves and Hue/Saturation

- Add a Curves parameter input representation, then add custom parameter-vector adapters for Curves.
- Add staged/band-limited tuning for Hue/Saturation Adjustment.
- Add objective presets for blur, luminance, color, and perceptual similarity.

### Phase 4: selected chains

- Represent the selected process chain as an ordered `EvaluationPlan` pipeline.
- Cache intermediate outputs per node where possible.
- Begin with staged node-by-node tuning, then add joint refinement over the best candidates.

## Risks and safeguards

- Avoid calling DearPyGui APIs from worker threads while evaluating candidates. Extract settings before optimization, evaluate candidates against in-memory images, and apply the final settings on the GUI side.
- Limit image size during scoring to keep interactive runs responsive.
- Put hard caps on evaluation budgets, especially for multi-node chains.
- Treat linked parameter inputs carefully: the Auto Tune node should own any parameter ports it is tuning, while unrelated linked inputs should be frozen or clearly shown as excluded.
- Store the final applied change as a single history action rather than one action per candidate evaluation.
- Do not rely on graph feedback loops for candidate evaluation; loops should remain disallowed in the visible graph.

## Why this shape fits the existing code

The process nodes already separate parameter metadata from image processing for many simple nodes. Declarative nodes define parameter dictionaries, read widget values into `parameter_values`, normalize them, and call `process(frame, **parameter_values)`. That is a good foundation for an optimizer that evaluates parameter dictionaries without needing to manipulate widgets for every candidate.

The same architecture also leaves room for non-declarative or custom controls: those nodes can provide adapters that map between optimizer vectors and node settings while the auto-tuning service remains generic.
