# Documentation source

Build the PDF from the repository root with:

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

The PDF and the repository README use the PNG files in `assets/screenshots/`. Replace the four placeholders with real screenshots while retaining the filenames. Rebuild the PDF with `make docs` after replacing them. See `assets/screenshots/README.md` for the expected content and recommended dimensions.

