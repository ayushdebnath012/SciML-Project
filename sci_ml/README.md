# NeurIPS 2026 workshop paper

Built on the official template in [`sci_ml/`](../../sci_ml/) — `neurips_2026.sty`
and `checklist.tex` are unmodified copies of it, and `main.tex` follows the
template's structure (track options, author block, `ack`, appendix, checklist).

Every headline result in the text and the tables is generated from result
files, so re-running the generator scripts after more runs land updates the
prose and the tables together.

The fetch order is also deliberate. The original 99-run sweep is pulled first,
then `results_corrected` is overlaid onto matching local run directories. That
second tree contains the 42 published-recipe runs and 15 causal SOAP runs
regenerated after fixing continuation-state handoff; unaffected RBA and
L-BFGS-only arms continue to come from the original sweep. The remote trees are
kept separate so the before/after results remain auditable.
For a local audit copy, pass
`--archive-original results/pinn_pre_contfix` to the fetch script; it copies the
uncorrected tree before applying the corrected overlay.

## Build

```bash
# 1. preserve the uncorrected tree, then apply the corrected overlay
RPASS='...' python wave/server/fetch_results.py \
    --archive-original results/pinn_pre_contfix

# 2. regenerate tables/*.tex (including tables/numbers.tex, the inline macros)
python paper/neurips2026_workshop/make_tables.py

# 3. regenerate figures/*.pdf
python paper/neurips2026_workshop/make_figures.py

# 4. compile
cd paper/neurips2026_workshop
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`make_tables.py` writes `tables/numbers.tex`, a set of `\newcommand`s such as
`\PinnBestErr`, `\AblMinGain` and `\FDFloor`. `main.tex`, `sections.tex` and the
checklist quote those macros rather than literal figures, so the abstract, the
tables and the checklist justifications cannot drift apart.

## Page budget — check this against your workshop

The template's nine-page limit is the **main conference** rule. Most NeurIPS
workshops cap submissions well below it, commonly at four to six pages. Confirm
the target workshop's own limit before submitting.

Current build:

| pages | what |
|---|---|
| 1–8 | content — **8 of 9 used, one page of slack** |
| 9–10 | references (appendix begins near the bottom of page 10) |
| 10–11 | appendix |
| 12–18 | checklist |

Acknowledgments, references, the checklist and technical appendices do not
count toward the limit.

If the paper has to come down, cut in this order — each is self-contained and
none of them carries the main claim:

1. **Table 2** (`\input{tables/reffloor}`, §3) — the reference-floor detail. The
   headline number survives in Table 1's floor column and in the text.
2. **Table 5** (`tables/pinn_variants`, §5) — the SOAP/RBA/L-BFGS-only grid. The
   RBA point that matters is made in prose.
3. **Figure 1** (cost vs accuracy) — Table 1 already carries the cost numbers.
4. **§3's first two paragraphs** — the FD/FEM agreement check is a control, not
   a result.

Keep Table 4 and Figure 2: they are the paper.

## Before submitting

1. **`\workshoptitle{...}`** in `main.tex` is a placeholder. Workshop options
   require it; the proceedings footnote is incomplete without it.
2. **The track option.** `main.tex` uses `[dblblindworkshop]`. Switch to
   `[sglblindworkshop]` for single-blind, and add `final` after acceptance. The
   workshop name appears in the page footer only in `final` mode — at
   submission the style file prints the generic "Submitted to ..." notice for
   every track. That is the style file's own behaviour, not something to patch.
3. **The `\author` block** — replace for any non-anonymous option.
4. **The `ack` environment** — hidden at submission, must be filled in for
   camera-ready with funding sources and competing interests.

## The checklist

`checklist.tex` here is the official file with the instruction block removed
(as the template instructs) and all 16 answers filled in. To regenerate it from
scratch after an upstream change:

```bash
cp sci_ml/checklist.tex paper/neurips2026_workshop/checklist.tex
python paper/neurips2026_workshop/fill_checklist.py
```

`fill_checklist.py` keys each answer to its question heading and aborts if a
heading it does not recognise appears, so an upstream edit fails loudly rather
than silently mis-assigning an answer. Answers live in the `ANSWERS` dict at the
top of that script — edit them there, not in the `.tex`.

## Layout

| path | what it is |
|---|---|
| `main.tex` | preamble, title, abstract, structure |
| `sections.tex` | the body |
| `appendix.tex` | reproduction recipe and the implementation notes that changed a result |
| `checklist.tex` | official checklist, answers filled |
| `refs.bib` | bibliography |
| `neurips_2026.sty` | official style file, unmodified |
| `make_tables.py` | result files → `tables/*.tex` |
| `make_figures.py` | result files → `figures/*.pdf` |
| `fill_checklist.py` | answers → `checklist.tex` |
| `tables/`, `figures/` | generated, safe to delete |

## Where the numbers come from

| table / figure | source |
|---|---|
| Table 1 (classical cost) | `results/classical/classical_benchmark.json` |
| Table 2 (reference floor) | same |
| Table 3 (architectures) | `results/pinn/**/l2_errors.json` |
| Table 4 (weighting ablation) | `results/pinn_ablation/**/l2_errors.json` + Table 3's baseline column |
| Table 5 (variant arms) | `results/pinn/**/l2_errors.json` |
| Figure 1 (cost vs accuracy) | classical JSON + PINN results + `results/pinn/run_seconds.json` |
| Figure 2 (causal frontier) | `results/pinn/**/causal_convergence.json` |

Regenerate the classical side with:

```bash
python wave/numerical/run_classical_benchmark.py \
    --out results/classical/classical_benchmark.json
```

That takes about six minutes on one CPU core, most of it in the two spectral
reference solves per material.
