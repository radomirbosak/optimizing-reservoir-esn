AF=$(HOME)/diplo/clanky/clanok2/grafika/act-fun

all:
	echo "make copy"

copy:
	cp figure0.pdf $(AF)/iden.pdf
	cp figure1.pdf $(AF)/sigm.pdf
	cp figure2.pdf $(AF)/tanh.pdf
	cp figure3.pdf $(AF)/thresh.pdf

clear:
	rm figure*.pdf
	rm figure*.png

generate:
	python3 afunex.py