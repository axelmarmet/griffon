default: simple

simple: simple.native

test: test.native

clean:
	rm -rf _build/*

%.native:
	ocamlbuild -use-ocamlfind -pkgs lymp -tag thread $@
	mv $@ $*

.PHONY: test default