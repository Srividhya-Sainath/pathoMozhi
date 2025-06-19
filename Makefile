install: ## [Local development] Upgrade pip, install requirements, install package.
	python -m pip install -U pip
	python -m pip install -e .

install-dev: ## [Local development] Install test requirements
	python -m pip install -r requirements-dev.txt

lint: ## [Local development] Run mypy, pylint and black
	python -m mypy pathoMozhi
	python -m pylint pathoMozhi
	python -m black --check -l 120 pathoMozhi

black: ## [Local development] Auto-format python code using black
	python -m black -l 120 .

.PHONY: help

help: # Run `make help` to get help on the make commands
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
