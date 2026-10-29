# Build the paper: `make` (needs latexmk + pdflatex + bibtex), `make clean` to remove build files.
main.pdf: main.tex references.bib neurips_2026.sty $(wildcard ../figs/fig*.png)
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	latexmk -C main.tex
	rm -f main.bbl

.PHONY: clean
