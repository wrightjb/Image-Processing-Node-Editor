# Curves node

The Curves node edits and applies a full curve set:

- `White` adjusts all color channels first.
- `Red`, `Green`, and `Blue` adjust their matching channels after `White`.

The effective order is always:

```text
input image -> White curve -> Red/Green/Blue curves -> output image
```

The editor stores all four curves in one payload:

```json
{
  "curves": {
    "White": [[0, 0], [255, 255]],
    "Red": [[0, 0], [255, 255]],
    "Green": [[0, 0], [255, 255]],
    "Blue": [[0, 0], [255, 255]]
  }
}
```

Legacy payloads that contain only a single points list are interpreted as the
`White` curve.

## Editing controls

- **Edit Channel** selects which curve is editable.
- Inactive channels are shown as ghost curves.
- **Clear Channel** resets only the selected channel to identity.
- **Clear All** resets every channel to identity.
- **Copy Curves**, **Import**, and **Export** operate on the full curve set.

## Reusing curves

The Curves node has a `CurvePoints` output port. Connect it to another Curves
node's curves input to reuse the same curve set downstream. If the Curves node
has no image input, it can still be used as a curve-set source; image processing
is simply skipped by the normal node update flow.

## Auto Tune (Curves)

Auto Tune (Curves) can tune either a single channel or `All` channels. In `All`
mode it tunes `White` first, applies that curve to the source, then tunes
`Red`, `Green`, and `Blue` against the white-adjusted source.
