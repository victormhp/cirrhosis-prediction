SRC=docs

latexmk:
	latexmk -pdf -bibtex $(SRC).tex

clean: 
	-rm  -f \
	*.aux *.log *.nav *.log *.toc *.snm *.out *.dvi *.ps \
	*latexmk *.fls *.bbl *.blg *.lof *.lot *.tdo
