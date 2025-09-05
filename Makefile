.PHONY: train train-detection train-habitat install cleanup

train: train-detection

train-detection:
	@echo "Starting YOLO detection training..."
	python scripts/train.py --mode detection
	@echo "Detection training pipeline finished."

train-habitat:
	@echo "Starting YOLO habitat classification training..."
	python scripts/train.py --mode classification
	@echo "Habitat classification training pipeline finished."

install:
	pip install -r requirements.txt

cleanup:
	@echo "Running cleanup script..."
	python app/cleanup.py
	@echo "Cleanup finished."
