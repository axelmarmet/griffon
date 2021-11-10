rm -r data
rm -r sexp_cache

(cd coq_projects/test_proj
make clean)

(cd coq_projects/train_proj
make clean)

(cd coq_projects/valid_proj
make clean)

(cd coq_projects/ev
make clean)