# Release Process

This document describes the automated PyPI release process for redback.

## Overview

The project uses GitHub Actions to automatically publish releases to PyPI when version tags are pushed.

## Setup

### Option 1: Trusted Publishing (Recommended)

Trusted publishing is the modern, secure way to publish to PyPI without using API tokens.

1. Go to https://pypi.org/manage/account/publishing/
2. Add a new pending publisher with:
   - **PyPI Project Name**: `redback`
   - **Owner**: `nikhil-sarin`
   - **Repository name**: `redback`
   - **Workflow name**: `pypi-publish.yml`
   - **Environment name**: `pypi`

The workflow is already configured to use trusted publishing by default.

### Option 2: API Token

If you prefer to use an API token:

1. Generate a PyPI API token at https://pypi.org/manage/account/token/
2. Add the token as a repository secret named `PYPI_API_TOKEN`
3. Uncomment the `password` line in `.github/workflows/pypi-publish.yml`:
   ```yaml
   password: ${{ secrets.PYPI_API_TOKEN }}
   ```

## How to Release

### 1. Update Version

Update the version number in `setup.py`:

```python
setup(
    name='redback',
    version='1.12.2',  # Update this
    ...
)
```

### 2. Commit Changes

```bash
git add setup.py
git commit -m "Bump version to 1.12.2"
git push
```

### 3. Create and Push Tag

```bash
git tag v1.12.2
git push origin v1.12.2
```

### 4. Automated Process

Once the tag is pushed, the workflow automatically:
1. Builds the distribution packages (wheel and source distribution)
2. Verifies the packages
3. Publishes to PyPI
4. Creates a GitHub release with the distribution files attached

## Manual Trigger

The workflow can also be triggered manually from the Actions tab on GitHub.

## Monitoring

You can monitor the release process in the "Actions" tab of the repository. The workflow will show:
- Build status
- PyPI publication status
- GitHub release creation status

## Troubleshooting

### Publication Fails

- **Trusted Publishing**: Ensure the publisher is configured correctly on PyPI
- **API Token**: Verify the `PYPI_API_TOKEN` secret is set correctly
- **Version Conflict**: Ensure the version doesn't already exist on PyPI

### Build Fails

- Check that all dependencies are properly specified in `setup.py`
- Verify the package structure is correct
- Review the workflow logs for specific error messages
