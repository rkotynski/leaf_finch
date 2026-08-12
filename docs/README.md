# Documentation source

Project homepage: https://github.com/rkotynski/leaf_finch

The PDF documentation is intentionally English-only. Build it from the repository root with:

```bash
make docs
```

or directly:

```bash
cd docs
pdflatex -interaction=nonstopmode -halt-on-error LEAF_FINCH_Documentation.tex
pdflatex -interaction=nonstopmode -halt-on-error LEAF_FINCH_Documentation.tex
```

## Replacing GUI screenshots

The PDF and repository README use the PNG files in `assets/screenshots/`. The distributed screenshots intentionally show the default English interface. Replace them while retaining the filenames, then rebuild the PDF with `make docs`. See `assets/screenshots/README.md` for the expected content and recommended dimensions.
