clean:
	rm -rf build/

.PHONY: build
build:
	python3 setup.py build_ext --inplace



