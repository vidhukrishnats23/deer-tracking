.PHONY: train

train:
	@echo "Starting YOLO training..."
	python scripts/train.py
	@echo "Training pipeline finished."

install:
	pip install -r requirements.txt
