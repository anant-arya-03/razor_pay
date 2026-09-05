.PHONY: all data train experiment sweep clean demo test

all: data train experiment

data:
	python -m data.synthetic_gen --n 3000 --out artifacts/events.jsonl --seed 42

train:
	python -m models.train_classifier --events artifacts/events.jsonl

experiment:
	python -m eval.run_experiment

sweep:
	python -m eval.run_experiment --sweep

demo: all sweep

pitch:
	python -m eval.demo

test:
	pytest tests/ -q

clean:
	rm -f artifacts/*.jsonl artifacts/*.json artifacts/*.joblib
