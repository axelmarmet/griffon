build:
	python src/griffon/preprocessing/stage1/stage1.py --base_root=$(DATA_DIR)/base
	python src/griffon/preprocessing/stage2/stage2.py --base_root=$(DATA_DIR)/base/ --config_path=configs/config.json
	python src/griffon/preprocessing/stage1/create_semantic_test_stage1.py --semantic_test_path=$(DATA_DIR)/semantic_tests
	python src/griffon/preprocessing/stage2/create_semantic_test_stage2.py --data_root=$(DATA_DIR)/
	python src/griffon/dataset/create_count_dataset.py --data_root=$(DATA_DIR)

clean:
	rm -rf $(DATA_DIR)/CounT
	rm -rf $(DATA_DIR)/base/stage1
	rm -rf $(DATA_DIR)/base/stage2
	rm -rf $(DATA_DIR)/semantic_tests/stage1
	rm -rf $(DATA_DIR)/semantic_tests/stage2
	rm -rf $(DATA_DIR)/semantic_tests/new_masks.json