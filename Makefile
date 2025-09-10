# Makefile

# set default target (goal)
.DEFAULT_GOAL := all

# phony targets
.PHONY = build install all clean test flake8 coverage unittest

all:
	@$(MAKE) clean
	@$(MAKE) build
	@$(MAKE) install-dev
	@$(MAKE) test

help: ## Display help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(CYAN)%-30s$(COFF) %s\n", $$1, $$2}'


build-whl: ## Build the binary
	poetry build

install: ## Install the package
	poetry install --all-extras
	poetry self add poetry-dotenv-plugin

install-dev:  ## Install dev dependencies
	@printf "$(CYAN)Updating deps$(COFF)\n"
	poetry install --only dev
	poetry self add poetry-dotenv-plugin

clean: ## Clean biild, python and test files and artifacts
	@$(MAKE) clean-build 
	@$(MAKE) clean-pyc 
	@$(MAKE) clean-test

clean-build: ## Delete build artifacts
	echo "clean build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf .eggs/
	find . -name '*.egg-info' -exec rm -rf {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## Delete Python file artifacts
	echo "clean Python file artifacts..."
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## Delete test and coverage artifacts
	echo "clean test and coverage artifacts..."
	rm -rf .tox/
	rm -f .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache
	rm -rf flake-report/

lint: ## Lint and reformat the code
	@poetry run autoflake sgs tests --remove-all-unused-imports --recursive --remove-unused-variables --in-place --exclude=__init__.py
	@poetry run black sgs tests --line-length 120 -q
	@poetry run isort sgs tests

lint-check: ## check lint
	poetry run black sgs tests --line-length 120 --check --diff
	poetry run isort sgs tests --check-only --diff
	poetry run autoflake sgs tests --remove-all-unused-imports --recursive --remove-unused-variables --check-diff --exclude=__init__.py

check-types: ## Check data types
	@poetry run mypy sgs tests --ignore-missing-imports --pretty

unittest: ## Run unit-tests
	@poetry run pytest -s -v tests

coverage: ## Run coverage tests
	@mkdir -p .reports
	@poetry run pytest --cov sgs --cov-report=html:.reports/htmlcov --junitxml=.reports/coverage.xml 
	@poetry run coverage report -m

test: ## Run both unit-tests and coverage tests
	echo "run unittest..."
	@$(MAKE) unittest
	echo "run coverage test..."
	@$(MAKE) coverage

deploy-gitlab: ## Deploy to GitLab
	## check if required env variables exists.
	ifndef PROD_360_INFO_GITLAB_DEPLOY_NAME
		$(error PROD_360_INFO_GITLAB_DEPLOY_NAME is undefined)
	endif
	ifndef PROD_360_INFO_GITLAB_TOKEN
		$(error PROD_360_INFO_GITLAB_TOKEN is undefined)
	endif

	$(@error NotImplemented)

deploy-jfrog:
	## check if required env variables exists.
	ifndef JFROG_URL
		$(error JFROG_URL is undefined)
	endif
	ifndef JFROG_USERNAME
		$(error JFROG_USERNAME is undefined)
	endif
	ifndef JFROG_PASSWORD
		$(error JFROG_PASSWORD is undefined)
	endif

	twine upload \
		--repository-url $(JFROG_URL) \
		-u $(JFROG_USERNAME) \
		-p $(JFROG_PASSWORD) \
		dist/*

run-chainlit-ui:
	@poetry run chainlit_ui

run:
	@poetry run cli


