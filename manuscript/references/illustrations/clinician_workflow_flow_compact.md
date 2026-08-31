# Clinician Workflow Flowchart — Build Notes

Companion document for `clinician_workflow_flow_compact.svg` / `.pdf` / `_qa-1.png`.
This is the figure used by the manuscript: `main.tex:1133` includes the `.pdf` at
`width=0.99\textwidth`. Two earlier variants — the linear four-box `clinician_workflow.svg`
and the tall flowchart `clinician_workflow_flow.svg` — were dropped once this one was chosen.

## Source of truth

The node/edge structure transcribes `documentation/notes/clinician_workflow.mmd` (`flowchart TB`),
which is kept in sync with the figure's labels. Mermaid shapes map as follows:

| Mermaid | Shape here | Nodes |
|---|---|---|
| `["..."]` | rect — square corners for data, rounded for process | A, G, G1, O, P |
| `("...")` | rounded rect | B |
| `{"..."}` | diamond | C, H, L |
| `(["..."])` | stadium, blue fill | Z, R, Q — and F |

Branch-label nodes (`D`/`E`/`K`, `I`/`J`, `M`/`N`) are rendered as **edge labels** on the
diamonds' outgoing arrows rather than as boxes — standard flowchart practice, and it removes
three whole levels from the stack.

Content behind the nodes: feature groups from `module/dataload.py`, threshold and fold model
from `module/expreport.py`, the six actionable features from
`module/experiments/bin_cf_final_202608.yml` (`dice.cf_features.actionable`). Recorded here
only; the figure names no script.

## Deviations from the Mermaid source

- `F` ("Low Priority or Routine follow-up") is drawn as a **terminal**, not a process rect.
  It is a leaf in the graph and reads as an outcome.
- **`Calibrated Threshold` is an added node**, feeding `C`. Not in the Mermaid; makes explicit
  that the threshold is a stored artifact fed into the comparison, not derived at the visit.
- `I` ("Few or none") **merges into the `way above` edge** rather than dropping to `Z` on its
  own bend. Same destination, one less vertical in the right margin.
- **`O -> Q` and `P -> R` are drawn horizontally**, mirrored so both terminals sit on the
  outside. Removes one level; no fidelity cost, only two arrow directions change.
- **`Z` sits in the right-hand lane at mid-height**, not in the bottom terminal row — the
  three-terminal row was what set the canvas width.
- No title, subtitle, or caveat callout: all three live in the LaTeX caption instead.
- `B` is capitalized "CatBoost" (vendor spelling).

## Color by role

Color encodes **what kind of thing a node is**, not which pipeline stage it belongs to.
Border and inner text take the role color; fills are the same hue at ~5%.

| Role | Nodes | Border | Text | Fill |
|---|---|---|---|---|
| Data — collected or stored | `A`, Calibrated Threshold | `#12796a` | `#0d5c51` | `#eef6f4` |
| Automated computation | `B`, `G` | `#4a3aa7` | `#332876` | `#f3f1fb` |
| Decision | `C`, `H`, `L` | `#52514e` | `#2e2d2b` | `#ffffff` |
| Clinician action | `G1`, `O`, `P` | `#c2600f` | `#8a4310` | `#fdf4ec` |
| Outcome | `F`, `Q`, `R`, `Z` | `#2a78d6` | `#1c4f8f` | `#eaf2fc` |

Connectors and edge labels stay neutral (`#898781` / `#52514e`) — they are not nodes.

The orange for clinician action deliberately reuses the hue of the retired decision-support
callout: the human-in-the-loop caveat is now carried by the coding rather than asserted in a
box. **In greyscale, purple and the decision grey converge, as do orange and blue** — shape
still separates them, but the coding degrades. The caption defines the colors, since no reader
infers "orange = a clinician does this" unaided.

## Design

Same conventions as `pipeline_overview.svg`: Arial, discrete `<rect>/<polygon>/<line>` shapes,
no `<style>`/`<defs>`/gradients, so the file imports cleanly into Canva.

Canvas `1095 x 1138`. **All strokes are 2px** — node borders, diamonds, and connectors alike —
so nothing carries accidental emphasis through line weight; role is carried by color only.
All three diamonds carry 22px text; all seven edge labels are 21px, set once on their group.
Figure bbox is `x 30..1065`, giving equal 30px side margins.

Shape choices worth recording: `B` and `Calibrated Threshold` were once stadiums, which is the
**terminator** symbol — they were sharing a shape with the four outcome nodes while being the
opposite of an end state. `A` and `Calibrated Threshold` now take square corners (data),
everything procedural is rounded, outcomes are stadiums, decisions are diamonds.

## Print sizing (sn-jnl, measured)

`sn-jnl` with `sn-mathphys-num` gives `\textwidth = 372pt (5.17in)` and
`\textheight = 552.7pt` — narrower than a default article class, which is what governs how
large the lettering renders.

At a fixed share of the page, rendered point size depends **only** on the figure's pixel
height: `pt = px * (figure height in pt) / H`. Canvas width is free provided the figure stays
no wider than about its own height (else page width binds first). Hence a near-square canvas.

**Every text class clears 7pt.** That was the governing constraint and it is why the figure
runs slightly over two thirds of the text block rather than exactly two thirds.

At `width=0.99\textwidth`: **383pt tall = 69.3% of the text block**.

| text class | px | rendered |
|---|---|---|
| node titles | 24 | 8.07pt |
| terminals, `G` title | 23 | 7.74pt |
| diamonds, process body | 22 | 7.40pt |
| `A` bullets, `G` feature list, edge labels, threshold | 21 | 7.06pt |

Nothing is set below 21px. Reaching 7pt cost 34px of height: the `G` feature list went back to
two lines at 21px, and `A` widened to 950px so its four bullet columns still fit on one row.
An earlier build had these classes at 19-20px, rendering 6.3-6.7pt — below the ~7pt normally
wanted for journal figures, and the reason for the change.

## Regenerating

Edit the `.svg` directly, then run `./render.sh clinician_workflow_flow_compact` from this
directory (headless Chrome -> vector PDF, `pdftoppm` -> PNG). Requires `google-chrome` and
`pdftoppm` (poppler-utils). LaTeX includes the **`.pdf`**; the `_qa-1.png` is a QA artifact
only.

## Closed

`Rescreen` stays a terminal. It was queried as a possible loop back to `A`; the decision is to
leave it as an end state.
