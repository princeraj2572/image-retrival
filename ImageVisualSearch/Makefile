.PHONY: help setup install install-dev test run-ui clean verify docs

help:
	@echo "ImageVisualSearch - Development Commands"
	@echo "=========================================="
	@echo ""
	@echo "setup              Create project directory structure"
	@echo "install            Install dependencies"
	@echo "install-dev        Install with development tools"
	@echo "verify             Verify project setup"
	@echo "test               Run unit tests"
	@echo "test-v             Run tests with verbose output"
	@echo "run-ui             Launch web UI"
	@echo "clean              Clean build and cache files"
	@echo "clean-all          Clean everything including data"
	@echo ""

setup:
	python setup_project.py

install:
	pip install --upgrade pip setuptools wheel
	pip install -r requirements.txt

install-dev:
	pip install --upgrade pip setuptools wheel
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 pylint

verify:
	python verify_installation.py

test:
	pytest tests/ -v

test-v:
	pytest tests/ -vv --tb=short

test-cov:
	pytest tests/ --cov=modules --cov=utils --cov-report=html

run-ui:
	python ui/app.py

run-main:
	python main.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

clean-all: clean
	rm -rf venv/
	rm -rf env/
	rm -f .env

format:
	black modules/ utils/ ui/ tests/ --line-length 100

lint:
	flake8 modules/ utils/ ui/ tests/ --max-line-length=100
	pylint modules/ utils/ ui/ tests/ --max-line-length=100

requirements:
	pip freeze > requirements.txt

docs:
	@echo "Reading documentation..."
	@more README.md
