# Deployment & Publishing Guide

This guide explains how to package the `upscaler` library and publish it so others can install it.

## 1. Build the Package (`.whl`)

To create an installable wheel file:
1.  Install build tools:
    ```bash
    pip install build twine
    ```
2.  Run the build:
    ```bash
    python -m build
    ```
    This will create a `dist/` folder containing:
    -   `upscaler-0.1.0-py3-none-any.whl` (The compiled package)
    -   `upscaler-0.1.0.tar.gz` (Source code)

## 2. Publish to PyPI (Python Package Index)
Publishing to PyPI allows anyone to install with `pip install upscaler`.

1.  Create an account at [pypi.org](https://pypi.org/).
2.  Upload the package:
    ```bash
    python -m twine upload dist/*
    ```
3.  Users can now install:
    ```bash
    pip install upscaler
    ```

## 3. Install from GitHub
If you don't want to use PyPI, users can install directly from your GitHub repository.

```bash
pip install git+https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
```
