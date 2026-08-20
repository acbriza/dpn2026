# Porting the manuscript from Overleaf to VS Code, and re-templating IEEE → Springer Nature

This note records how the manuscript moved from an Overleaf-only workflow to
local editing in VS Code, and how it was re-templated from an IEEE
conference format to a Springer Nature journal format
(`module/legacy/202608/overleaf/main.tex` remains the older, separate
manuscript snapshot referenced in the top-level `CLAUDE.md`; this note is
about the newer `manuscript/` folder at the repo root).

## Layout

- `manuscript-ieee/` — the original manuscript as exported from Overleaf,
  IEEE conference format (`\documentclass[conference]{IEEEtran}`). Kept
  as-is, untouched, for reference.
- `manuscript/` — the active manuscript, re-templated to the Springer
  Nature journal template (`sn-jnl.cls`). This is what gets edited and
  submitted going forward.
- `manuscript/main-claude.tex` — Claude's full first-pass translation of
  the IEEE manuscript into the `sn-jnl` template (everything described
  under "Porting the Overleaf zip" and "Bugs hit and fixed" below). Kept
  as a reference/diff base, not built directly.
- `manuscript/main.tex` — the author's working copy going forward: started
  as a fresh copy of the template's own `sn-article.tex` skeleton, to be
  manually rebuilt using `main-claude.tex` as the source of content and
  fixes to pull from, rather than continuing to build on Claude's version
  directly. **This is the file to open and build in VS Code.**
- `manuscript/sn-article.tex` / `sn-bibliography.bib` — the template's own
  example article and bibliography, kept for reference (macro usage,
  environment examples). Not part of the actual paper.
- Both `main.tex` and `main-claude.tex` are self-contained (own
  `\documentclass` and `\begin{document}`), so neither needs LaTeX
  Workshop root-file configuration — open either and build directly. When
  rebuilding `main.tex` by hand, the fixes listed under "Bugs hit and
  fixed along the way" still apply verbatim (they're template/font-level
  issues, not specific to any particular section's content) — copy those
  preamble lines and the em-dash/bib-`&` fixes across rather than
  rediscovering them.

## VS Code: two LaTeX extensions, pick the right one

The workspace has both `james-yu.latex-workshop` and `mjpvs.latex-previewer`
installed. They don't share settings or build logic:

- **LaTeX Workshop** (`Ctrl+Alt+V`) — respects root-file resolution
  (magic comments, `.vscode/settings.json`), and its default `latexmk`
  recipe runs the full `pdflatex → bibtex → pdflatex → pdflatex` cycle
  needed for citations to resolve. **Use this for `manuscript/main.tex`.**
- **LaTeX Previewer** (`Ctrl+Shift+L`) — always compiles whichever file is
  open, directly, with `dvilualatex`, with no root-file concept and no
  bibtex pass. Fine for standalone fragments, wrong choice for a document
  with a bibliography.

## Porting the Overleaf zip

The project was exported from Overleaf as a source zip (Download → Source)
and unzipped into `manuscript-ieee/`, then rebuilt in the new template:

1. Downloaded the Springer Nature `sn-jnl` template package separately and
   extracted it into `manuscript/`.
2. Copied `figures/`, `figures2/`, and `bibfile.bib` from
   `manuscript-ieee/` into `manuscript/`.
3. Reassembled `manuscript/main.tex` from three pieces:
   - a new **header** (preamble, title/author/affiliation block using
     `sn-jnl`'s `\author*[]{\fnm{}\sur{}}`/`\affil{}` macros, `\abstract{}`,
     `\keywords{}`)
   - the **body**, carried over from the IEEE source largely unchanged —
     tables, figures, and prose are standard LaTeX and needed no
     translation. The only mechanical transform was converting
     `\section*{}` / `\subsection*{}` / `\subsubsection*{}` / `\paragraph*{}`
     (IEEE's unnumbered convention) to their unstarred, numbered
     equivalents (`sed`-based, since it's a pure syntactic swap)
   - a new **footer** (`\backmatter`, `\bmhead{Acknowledgements}`, a
     `Declarations` section, `\bibliography{bibfile}`)

## Bugs hit and fixed along the way

None of these were pre-existing issues — they're specific to the IEEE →
Springer Nature switch:

1. **`\cline` undefined.** `sn-jnl.cls` does `\let\cline\cmidrule`
   internally, before `\usepackage{booktabs}` runs in the document
   preamble, so the alias captures nothing. Fix: redo
   `\let\cline\cmidrule` in the preamble, after loading `booktabs`.
2. **Unicode minus sign (U+2212) unmapped.** A stray `−` (not a plain
   hyphen) in the prose isn't in the default font encoding under
   `sn-jnl.cls`. Fixed by mapping it via `newunicodechar`:
   `\newunicodechar{−}{\ensuremath{-}}`.
3. **Em dash inside `\paragraph{}` headings broke hyperref bookmarks.**
   `\paragraph{Patient 99 — Confident False Negative}` (raw Unicode em
   dash, U+2014) triggered a "Runaway argument" fatal error when hyperref
   tried to write the PDF bookmark string — even though the same character
   compiles fine in ordinary body text. Fixed by swapping the literal `—`
   for the ASCII em-dash ligature `---` specifically inside heading
   commands.
4. **Unescaped `&` in two `bibfile.bib` journal names** (`Journal of
   Neuropathology & Experimental Neurology`, `Nutrition & Metabolism`) —
   fatal for BibTeX (`Misplaced alignment tab character &`). Fixed by
   escaping to `\&` in the `.bib` source.
5. **`sn-vancouver-num.bst` not found by `bibtex`.** The template ships its
   `.bst` files under `manuscript/bst/`, but `bibtex` only looks in the
   working directory. Fixed by copying the one style file actually used
   (`sn-vancouver-num.bst`) up to `manuscript/`.

## Open items (not fixed — flagged for the author)

- **Reference style**: defaulted to `sn-vancouver-num` (medicine's usual
  numbered-citation convention) since the downloaded package is the
  generic multi-journal `sn-jnl` template, not journal-specific. Confirm
  against the actual target journal's instructions-for-authors.
- **`bibfile.bib` data gaps**: 12 entries are missing author/journal
  fields (surfaced now because `sn-vancouver-num.bst` is stricter about
  completeness than the old `plain.bst`), 4 duplicate entries, and several
  `\cite{}` keys used in the text aren't in the `.bib` file at all. All
  pre-existing gaps carried over from the IEEE draft, not introduced by
  the template swap. `latexmk`'s default recipe will report the build as
  "failed" over these even though a correct PDF still comes out.
- **`Declarations` section**: Funding and Ethics-approval were filled in
  from what the IEEE text already stated. Conflict of interest, Consent
  for publication, Data/Materials/Code availability are left as
  `[author to complete]` placeholders — not fabricated.
- Four figure references (`fig:categ`, `fig:ncs`, `fig:profile_mnsi`,
  `fig:sudo`) point at EDA figures that were already commented out in the
  IEEE source — pre-existing, shows as `??` in the compiled PDF.

## `.gitignore`

`manuscript-ieee/` is fully ignored (reference-only snapshot).
`manuscript/` is tracked except for:

```
manuscript/figures/
manuscript/figures2/
manuscript/Download+the+journal+article+template+package+*.zip
manuscript/*.aux
manuscript/*.bbl
manuscript/*.blg
manuscript/*.fdb_latexmk
manuscript/*.fls
manuscript/*.log
manuscript/*.out
manuscript/*.synctex.gz
manuscript/main.pdf
manuscript/main-claude.pdf
```

`figures/`/`figures2/` are ignored for size, which means a fresh clone
cannot fully rebuild `main.pdf` without those images being supplied
separately — a known trade-off, not an oversight.

Deleted (unused Springer Nature template clutter, not referenced by
`main.tex`): `sn-article.pdf`, `user-manual.pdf`, `empty.eps`, `fig.eps`.
(`sn-article.tex` and `sn-bibliography.bib` were deleted in the same pass,
then re-extracted from the template zip on request, since they're useful
as a macro/environment reference.)
