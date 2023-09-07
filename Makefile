all: clean build publish

build:
	flit build

clean:
	rm -rf dist
	rm -rf build

publish: build
	flit publish

.PHONY: build clean publish all
