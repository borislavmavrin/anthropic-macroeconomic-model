.PHONY: green

green:
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright
	uv run complexipy
	uv run pytest tests/
	uv run pylint src tests
	uv run codespell
