# docs

LaTeX source for "2D Image Classification with Random Convolutional
Kernels: A ROCKET-Based Approach" (Elsevier `cas-sc` single-file paper,
condensed from `bachelor_thesis_felicitas_lock_36434.pdf`).

## Layout

```
paper.tex           # single-file paper (frontmatter + all sections)
cas-refs.bib        # bibliography
figures/             # not currently used; drop plots here + \includegraphics if needed
bachelor_thesis_felicitas_lock_36434.pdf   # source thesis this paper condenses
```

## Build

```bash
cd docs
make            # -> build/paper.pdf
make clean      # remove build/ artifacts
```

Requires `latexmk` + a LaTeX distribution with the Elsevier `cas-sc` class
and `cas-model2-names.bst` (installed via `texlive-latex-base
texlive-latex-recommended texlive-fonts-recommended texlive-latex-extra
texlive-publishers texlive-science latexmk`; `cas-sc.cls` ships in
`texlive-latex-extra` under `els-cas-templates`).
