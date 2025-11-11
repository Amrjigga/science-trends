.PHONY: all clean

all:
	python src/cli.py

clean:
	rm -rf data/interim outputs
