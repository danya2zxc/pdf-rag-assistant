.PHONY: format lint

format:
	poetry run black .
	poetry run isort .
	poetry run ruff check --fix .

lint:
	poetry run black --check .
	poetry run isort --check-only .
	poetry run ruff check .
