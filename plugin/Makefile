default: simple

simple: simple.native

test: test.native

%.native:
	ocamlbuild -use-ocamlfind -pkgs lymp -tag thread $@
	mv $@ $*

.PHONY: test default