all:
	+make -C coq_projects/semantic_unit_tests -f Makefile all
	python ../../../CoqGym/CoqGym/check_proofs.py --file coq_projects/semantic_unit_tests/inference.meta
	python ../../../CoqGym/CoqGym/extract_proof.py --file coq_projects/semantic_unit_tests/inference.meta --proof app_type_1
	python ../../../CoqGym/CoqGym/extract_proof.py --file coq_projects/semantic_unit_tests/inference.meta --proof app_type_2
	python ../../../CoqGym/CoqGym/extract_proof.py --file coq_projects/semantic_unit_tests/inference.meta --proof app_type_3
	python ../../../CoqGym/CoqGym/extract_proof.py --file coq_projects/semantic_unit_tests/inference.meta --proof app_type_4
	python ../../../CoqGym/CoqGym/check_proofs.py --file coq_projects/semantic_unit_tests/natural_deduction.meta
	python ../../../CoqGym/CoqGym/extract_proof.py --file coq_projects/semantic_unit_tests/natural_deduction.meta --proof assumption
	python ../../../CoqGym/CoqGym/extract_proof.py --file coq_projects/semantic_unit_tests/natural_deduction.meta --proof conj_intro
	python ../../../CoqGym/CoqGym/extract_proof.py --file coq_projects/semantic_unit_tests/natural_deduction.meta --proof conj_elim_left
	python ../../../CoqGym/CoqGym/extract_proof.py --file coq_projects/semantic_unit_tests/natural_deduction.meta --proof conj_elim_right
	python ../../../CoqGym/CoqGym/extract_proof.py --file coq_projects/semantic_unit_tests/natural_deduction.meta --proof disj_intro_left
	python ../../../CoqGym/CoqGym/extract_proof.py --file coq_projects/semantic_unit_tests/natural_deduction.meta --proof disj_intro_right
	python ../../../CoqGym/CoqGym/extract_proof.py --file coq_projects/semantic_unit_tests/natural_deduction.meta --proof disj_elim
	python ../../../CoqGym/CoqGym/extract_proof.py --file coq_projects/semantic_unit_tests/natural_deduction.meta --proof impl_elim
	python ../../../CoqGym/CoqGym/extract_proof.py --file coq_projects/semantic_unit_tests/natural_deduction.meta --proof neg_elim
	python ../../../CoqGym/CoqGym/extract_proof.py --file coq_projects/semantic_unit_tests/natural_deduction.meta --proof aff
	python ../../../CoqGym/CoqGym/extract_proof.py --file coq_projects/semantic_unit_tests/natural_deduction.meta --proof contract
	python ../../../CoqGym/CoqGym/postprocess.py

clean:
	+make -C coq_projects/semantic_unit_tests -f Makefile clean