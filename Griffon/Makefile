COQ_GYM_ROOT=$(realpath ../../CoqGym/CoqGym/  )


test: | data/intermediary/corpus
	echo "helooo"

corpus: 
	which python
	python src/griffon/write_corpus.py --coq_gym_root=$(COQ_GYM_ROOT) --corpus_root=data/intermediary/corpus/