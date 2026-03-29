.PHONY: test test-all test-slow lint lint-fix docs docs-serve demo notebook-test clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

test: ## Run fast tests (excludes slow, witness, changepoint)
	python -m pytest tests/ -v --tb=short -k "not slow and not witness and not changepoint"

test-all: ## Run all tests including slow
	python -m pytest tests/ -v --tb=short

test-slow: ## Run only slow validation tests
	python -m pytest tests/ -v --tb=short -k "slow"

lint: ## Run ruff linter
	ruff check .

lint-fix: ## Run ruff with auto-fix
	ruff check --fix .

docs: ## Build Sphinx documentation
	sphinx-build -b html docs docs/_build/html

docs-serve: ## Build and serve docs locally on port 8080
	sphinx-build -b html docs docs/_build/html
	python -m http.server -d docs/_build/html 8080

demo: ## Launch Streamlit demo
	streamlit run demo/app.py

notebook-test: ## Validate tutorial notebooks execute without error
	python -m pytest --nbval-lax notebooks/tutorial_lorenz_walkthrough.ipynb notebooks/tutorial_heterogeneous_timescales.ipynb -v

clean: ## Remove build artifacts
	rm -rf docs/_build dist *.egg-info .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
