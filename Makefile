.PHONY: install test lint eval docker-test docker-lint

install:
	python -m pip install -e .
	python -m pip install -r requirements.txt

install-log:
	python -m pip install -r requirements-log.txt

test:
	python -m pytest -q

lint:
	python -m ruff check src tests scripts

eval:
	python scripts/eval_pose.py

export:
	python scripts/export_yolo_pose.py

train:
	python scripts/train_pose.py

docker-test:
	docker compose run --rm tests

docker-lint:
	docker compose run --rm lint
