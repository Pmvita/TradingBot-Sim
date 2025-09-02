# Trading Bot Simulator Makefile

.PHONY: help setup fmt lint typecheck test smoke docs clean run-example

# Python and Poetry paths
PYTHON := python3
POETRY := /Users/petermvita/.local/bin/poetry

help: ## Show this help message
	@echo "Trading Bot Simulator - Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies and setup development environment
	@echo "Setting up development environment..."
	$(POETRY) install
	@echo "Installing pre-commit hooks..."
	$(POETRY) run pre-commit install
	@echo "Creating necessary directories..."
	mkdir -p data/cache data/sample runs logs
	@echo "Setup complete! Run 'make help' for available commands."

fmt: ## Format code with black and isort
	@echo "Formatting code..."
	$(POETRY) run black src/ tests/ scripts/
	$(POETRY) run isort src/ tests/ scripts/

lint: ## Run linting with ruff
	@echo "Running linting..."
	$(POETRY) run ruff check src/ tests/ scripts/

typecheck: ## Run type checking with mypy
	@echo "Running type checking..."
	$(POETRY) run mypy src/

test: ## Run tests with pytest
	@echo "Running tests..."
	$(POETRY) run pytest tests/ -v

test-cov: ## Run tests with coverage
	@echo "Running tests with coverage..."
	$(POETRY) run pytest tests/ --cov=tbs --cov-report=html --cov-report=term-missing

gui: ## Launch Streamlit GUI
	@echo "Launching Streamlit GUI..."
	$(PYTHON) gui_launcher.py streamlit 8501

gui-dash: ## Launch Dash GUI
	@echo "Launching Dash GUI..."
	$(PYTHON) gui_launcher.py dash 8050

smoke: ## Run smoke test (quick training on sample data)
	@echo "Running smoke test..."
	$(POETRY) run tbs fetch --ticker BTC-USD --start 2023-01-01 --end 2024-01-01 --interval 1d
	$(POETRY) run tbs train --algo ppo --config configs/ppo.yaml --ticker BTC-USD --start 2023-01-01 --end 2024-01-01 --total-timesteps 1000
	@echo "Smoke test completed successfully!"

docs: ## Build documentation
	@echo "Building documentation..."
	$(POETRY) run mkdocs build

docs-serve: ## Serve documentation locally
	@echo "Serving documentation..."
	$(POETRY) run mkdocs serve

clean: ## Clean build artifacts and cache
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf data/cache/
	rm -rf runs/
	rm -rf logs/

run-example: ## Run complete example workflow
	@echo "Running complete example workflow..."
	@echo "1. Fetching sample data..."
	$(POETRY) run tbs fetch --ticker BTC-USD --start 2023-01-01 --end 2024-01-01 --interval 1d
	@echo "2. Training PPO agent..."
	$(POETRY) run tbs train --algo ppo --config configs/ppo.yaml --ticker BTC-USD --total-timesteps 10000
	@echo "3. Evaluating against baselines..."
	$(POETRY) run tbs eval --run runs/ppo/latest --baseline all
	@echo "Example completed! Check runs/ppo/latest/eval/ for results."

check: fmt lint typecheck test ## Run all quality checks

install: ## Install package in development mode
	$(POETRY) install

build: ## Build package
	$(POETRY) build

publish: ## Publish package to PyPI
	$(POETRY) publish

docker-build: ## Build Docker image
	docker compose build

docker-run: ## Run training in Docker
	docker compose run trainer tbs train --algo ppo --config configs/ppo.yaml

docker-shell: ## Open shell in Docker container
	docker compose run trainer bash
