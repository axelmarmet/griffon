#!/bin/bash

export PATH="/home/axel/Documents/master_proj/CoqGym/CoqGym/coq/bin:$PATH"

(
    cd coq_projects/test_proj
    make all
)

(
    cd coq_projects/train_proj
    make all
)

(
    cd coq_projects/valid_proj
    make all
)

(
    cd coq_projects/ev
    make all
)

python ../../../../CoqGym/CoqGym/check_proofs.py --file coq_projects/test_proj/test.meta
python ../../../../CoqGym/CoqGym/extract_proof.py --file coq_projects/test_proj/test.meta --proof find_in_hypo
python ../../../../CoqGym/CoqGym/extract_proof.py --file coq_projects/test_proj/test.meta --proof right_zero_neutral
python ../../../../CoqGym/CoqGym/extract_proof.py --file coq_projects/test_proj/test.meta --proof commutative

python ../../../../CoqGym/CoqGym/check_proofs.py --file coq_projects/valid_proj/test.meta
python ../../../../CoqGym/CoqGym/extract_proof.py --file coq_projects/valid_proj/test.meta --proof negation_fn_applied_twice
python ../../../../CoqGym/CoqGym/extract_proof.py --file coq_projects/valid_proj/test.meta --proof evar_check

python ../../../../CoqGym/CoqGym/check_proofs.py --file coq_projects/train_proj/test.meta
python ../../../../CoqGym/CoqGym/extract_proof.py --file coq_projects/train_proj/test.meta --proof vacuous_match
python ../../../../CoqGym/CoqGym/extract_proof.py --file coq_projects/train_proj/test.meta --proof test_cofix
python ../../../../CoqGym/CoqGym/extract_proof.py --file coq_projects/train_proj/test.meta --proof fix_test
python ../../../../CoqGym/CoqGym/extract_proof.py --file coq_projects/train_proj/test.meta --proof de_morgan
python ../../../../CoqGym/CoqGym/extract_proof.py --file coq_projects/train_proj/test.meta --proof other_obvious
python ../../../../CoqGym/CoqGym/extract_proof.py --file coq_projects/train_proj/test.meta --proof multi_fix_thm


python ../../../../CoqGym/CoqGym/check_proofs.py --file coq_projects/ev/ev.meta
python ../../../../CoqGym/CoqGym/extract_proof.py --file coq_projects/ev/ev.meta --proof double_is_ev
python ../../../../CoqGym/CoqGym/extract_proof.py --file coq_projects/ev/ev.meta --proof lambda_test
python ../../../../CoqGym/CoqGym/extract_proof.py --file coq_projects/ev/ev.meta --proof weird_com
python ../../../../CoqGym/CoqGym/extract_proof.py --file coq_projects/ev/ev.meta --proof record_check

python ../../../../CoqGym/CoqGym/postprocess.py