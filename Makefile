.PHONY: install test lint eval

install:
	python -m pip install -e .
	python -m pip install -r requirements.txt

test:
	python -m pytest -q

lint:
	python -m ruff check src tests scripts

eval:
	python scripts/eval_pose.py
