.PHONY: install test evals demo api ui lint docker clean

install:
	pip install -r requirements-dev.txt

test:
	pytest tests -q

evals:
	python -m evals.run_evals

demo:
	python -m scripts.demo_run

api:
	uvicorn src.api.main:app --reload --port 8000

ui:
	streamlit run src/ui/app.py

lint:
	ruff check src evals tests scripts

docker:
	docker compose up --build

clean:
	rm -rf data evals/report.json .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
