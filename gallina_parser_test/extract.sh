#!/bin/bash

python ../CoqGymProject/CoqGym/check_proofs.py --file coq_projects/my_test/test.meta
python ../CoqGymProject/CoqGym/extract_proof.py --file coq_projects/my_test/test.meta --proof ev_double
python ../CoqGymProject/CoqGym/postprocess.py