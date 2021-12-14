COQ_GYM_ROOT=$(realpath ../../CoqGym/CoqGym/  )

DATA_DIR=$(realpath ./data)
DATA_DIR_STAGE1=$(DATA_DIR)/stage1
DATA_DIR_STAGE2=$(DATA_DIR)/stage2
DATA_DIR_COUNT=$(DATA_DIR)/CounT
DATA_DIR_SEM_TEST=$(DATA_DIR)/semantic_tests
DATA_DIR_SEM_TEST_STAGE1=$(DATA_DIR)/semantic_tests
DATA_DIR_SEM_TEST_STAGE2=$(DATA_DIR)/semantic_tests

data-dirs:
	mkdir -p $(DATA_DIR)
	mkdir -p $(DATA_DIR_STAGE1)
	mkdir -p $(DATA_DIR_STAGE2)
	mkdir -p $(DATA_DIR_COUNT)
	mkdir -p $(DATA_DIR_SEM_TEST)
	mkdir -p $(DATA_DIR_SEM_TEST_STAGE1)
	mkdir -p $(DATA_DIR_SEM_TEST_STAGE2)

data:
	data gdown https://drive.google.com/uc\?export\=download\&id\=1CGQ_XKQKlrVt-grNzVI1x3laxIJYmWft


test: | data/intermediary/corpus
	echo "helooo"

corpus:
	which python
	python src/griffon/write_corpus.py --coq_gym_root=$(COQ_GYM_ROOT) --corpus_root=data/intermediary/corpus/