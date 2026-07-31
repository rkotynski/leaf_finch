.PHONY: test docs clean

test:
	python -m pytest

docs:
	cd docs && pdflatex -interaction=nonstopmode -halt-on-error LEAF_FINCH_Documentation.tex
	cd docs && pdflatex -interaction=nonstopmode -halt-on-error LEAF_FINCH_Documentation.tex

clean:
	rm -rf build dist *.egg-info .pytest_cache
	rm -f docs/*.aux docs/*.log docs/*.out docs/*.toc docs/*.lof docs/*.lot docs/*.fls docs/*.fdb_latexmk docs/*.synctex.gz
