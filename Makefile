
.PHONY: fmt lint test

fmt:
	ruff check --fix .

lint:
	ruff check .
	mypy defor

test:
	pytest -q
