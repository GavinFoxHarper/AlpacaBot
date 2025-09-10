# LAEF AlpacaBot Makefile
# Automated build, test, and deployment commands

.PHONY: help install test lint typecheck build run schedule clean

PYTHON := python
PIP := pip
VENV := venv
PYTEST := pytest
BLACK := black
RUFF := ruff
MYPY := mypy

# Default target
help:
	@echo "LAEF AlpacaBot - Available Commands:"
	@echo "  make install    - Install dependencies and setup environment"
	@echo "  make test       - Run comprehensive test suite"
	@echo "  make lint       - Run code linting (ruff + black)"
	@echo "  make typecheck  - Run type checking (mypy)"
	@echo "  make build      - Build and prepare for deployment"
	@echo "  make run        - Run orchestrator in paper mode"
	@echo "  make schedule   - Setup automated scheduling"
	@echo "  make clean      - Clean generated files and cache"
	@echo "  make report     - Generate system analysis report"
	@echo "  make all        - Run lint, typecheck, test, and build"

# Install dependencies
install:
	@echo "Installing dependencies..."
	$(PYTHON) -m venv $(VENV)
	$(VENV)/Scripts/activate && $(PIP) install --upgrade pip
	$(VENV)/Scripts/activate && $(PIP) install -r requirements.txt
	$(VENV)/Scripts/activate && $(PIP) install pytest pytest-cov hypothesis black ruff mypy
	@echo "Dependencies installed successfully!"

# Run tests with coverage
test:
	@echo "Running comprehensive test suite..."
	$(VENV)/Scripts/activate && $(PYTEST) tests/ \
		--cov=. \
		--cov-report=html \
		--cov-report=term-missing \
		--cov-branch \
		-v \
		--tb=short
	@echo "Test coverage report generated in htmlcov/"

# Lint code
lint:
	@echo "Running code linting..."
	$(VENV)/Scripts/activate && $(RUFF) check . --fix
	$(VENV)/Scripts/activate && $(BLACK) . --line-length 100
	@echo "Linting complete!"

# Type checking
typecheck:
	@echo "Running type checking..."
	$(VENV)/Scripts/activate && $(MYPY) . --strict \
		--ignore-missing-imports \
		--no-implicit-optional \
		--warn-redundant-casts \
		--warn-unused-ignores
	@echo "Type checking complete!"

# Build project
build: lint typecheck test
	@echo "Building project..."
	@mkdir -p dist
	@mkdir -p logs
	@mkdir -p reports
	@mkdir -p metrics
	@echo "Build complete!"

# Run orchestrator
run:
	@echo "Starting LAEF Orchestrator (Paper Mode)..."
	$(VENV)/Scripts/activate && $(PYTHON) orchestrator.py --paper

# Run live monitoring only
monitor:
	@echo "Starting Live Market Monitor..."
	$(VENV)/Scripts/activate && $(PYTHON) orchestrator.py --live-monitor-only

# Setup Windows Task Scheduler
schedule:
	@echo "Setting up Windows Task Scheduler..."
	@cmd /c setup_scheduler.bat
	@echo "Scheduler setup complete!"

# Generate analysis report
report:
	@echo "Generating system analysis report..."
	$(VENV)/Scripts/activate && $(PYTHON) -c "from report_generator import generate_report; generate_report()"
	@echo "Report generated in reports/"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	@if exist __pycache__ rmdir /s /q __pycache__
	@if exist .pytest_cache rmdir /s /q .pytest_cache
	@if exist htmlcov rmdir /s /q htmlcov
	@if exist .coverage del .coverage
	@if exist .mypy_cache rmdir /s /q .mypy_cache
	@if exist *.pyc del /s *.pyc
	@if exist nul del nul
	@echo "Clean complete!"

# Run all checks and build
all: lint typecheck test build
	@echo "All checks passed! System ready for deployment."

# Docker build (optional)
docker-build:
	docker build -t laef-alpacabot:latest .

# Docker run
docker-run:
	docker run -d --name laef-alpacabot \
		--env-file .env \
		-v $(PWD)/logs:/app/logs \
		-v $(PWD)/reports:/app/reports \
		laef-alpacabot:latest