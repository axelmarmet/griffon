#!/bin/bash

(
    cd coq_projects/aeval
    make all
)



python ../../../../CoqGym/CoqGym/check_proofs.py --file coq_projects/aexp/aexp.meta
python ../../../../CoqGym/CoqGym/extract_proof.py --file coq_projects/aexp/aexp.meta --proof constant_fold_ok

python ../../../../CoqGym/CoqGym/postprocess.py