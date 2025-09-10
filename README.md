# smart-grocery-search

Smart Grocery Search is a demo project that combines vector search, knowledge graphs, and DSPy to power smarter, attribute-aware product discovery in grocery retail.

## ℹ️ Features

## ⚡ Getting started

## 🌴 Set up dev environment

### 1. Install `pyenv` to manage Python binaries (versions)

For utilizing `pyenv` to manage your Python versions, follow the steps provided below. However, if you prefer using different Python virtual environment tools, you can skip the `pyenv` setup step and proceed with creating and activating your preferred virtual environment.

> **_NOTE:_**  If you intend to use a local Python virtual environment, make sure the the Python version matches the package Python version.

- Install all required dependencies

    ```bash
    sudo apt-get update; sudo apt-get install make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
    ```

- Download and execute installation script:

    ```bash
    curl https://pyenv.run | bash
    ```

- Add the following entries into your `~/.bashrc` file:

    ```bash
    # pyenv
    export PATH="$HOME/.pyenv/bin:$PATH"
    eval "$(pyenv init --path)"
    eval "$(pyenv virtualenv-init -)"
    ```

- Restart your shell:

    ```bash
    exec $SHELL
    ```

- Validate installation:

    ```bash
    pyenv --version
    ```

- Install latest python `3.12.2`:

    ```bash
    pyenv install 3.12.2
    ```

- Set your local pyenv python version to `3.12.2`:

    ```bash
    pyenv local 3.12.2
    ```

### 2. Install poetry

Install poetry. We utilize `poetry` for managing dependencies in this project.

```bash
pip install --upgrade poetry
```

### 3. Install `sgs`

Install the package alongside its dependencies:

```bash
make install
```

### 4. Activate the Poetry environment

```bash
poetry shell
```

### 5. Add dependencies 

To add dependencies, use the following command:

```bash
poetry add package-name
```

This command will add `package-name` to the project dependecies. Replace `package-name` with the name of the package you want to add. You can also specify version constraints, extras, and more.

### 6. Install the Package Alongsie its Dependencies

Once you've added your dependencies, run the following command to install them:

```bash
make install
```

This make target runs `poetry install` which will automatically generates a `poetry.lock` file to lock the dependencies and their versions. Both `pyproject.toml` and `poetry.lock` will be commited to version control.

## Run Scripts

You can define scripts in your pyproject.toml file under the `[tool.poetry.scripts]` section. To run a script, use:

```bash
poetry run script-name
```

## Build Package

To build your package, run:

```bash
make build-whl
```

This make target runs `poetry build` which will generate distribution files (wheel and source distributions) in the dist directory.

## Update Dependencies

To update your project's dependencies to their latest versions, use:

```bash
poetry update
```
