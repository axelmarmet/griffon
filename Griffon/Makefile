build:
	python src/griffon/preprocessing/stage1/stage1.py --base_root=data/small/data/base
	python src/griffon/preprocessing/stage2/stage2.py --base_root=data/small/data/base/ --config_path=configs/config.json
	python src/griffon/preprocessing/stage1/create_semantic_test_stage1.py --semantic_test_path=data/small/data/semantic_tests
	python src/griffon/preprocessing/stage2/create_semantic_test_stage2.py --data_root=data/small/data/
	python src/griffon/dataset/create_count_dataset.py --data_root=data/small/data

clean:
	rm -rf data/small/data/CounT
	rm -rf data/small/data/base/stage1
	rm -rf data/small/data/base/stage2
	rm -rf data/small/data/semantic_tests/stage1
	rm -rf data/small/data/semantic_tests/stage2
	rm -rf data/small/data/semantic_tests/new_masks.json