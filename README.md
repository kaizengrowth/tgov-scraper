# TGOV Scraper

A set of tools for scraping and analyzing data from the Tulsa Government Access Television (TGOV) website.

## Setup

This project uses Poetry for dependency management.

```bash
# Install dependencies
poetry install --no-root

# Activate the virtual environment
poetry self add poetry-plugin-shell
poetry shell

# Install Jupyter kernel for this environment (needed for Jupyter notebooks)
python -m ipykernel install --user --name=tgov-scraper --display-name="TGOV Scraper"
```

## Running

```bash
poetry run jupyter notebook
```

## Running Tests

```bash
# Run all tests
poetry run pytest

# Run specific tests
poetry run pytest tests/test_meetings.py

# Run tests with verbose output
poetry run pytest -v
```

## Project Structure

- `src/`: Source code for the scraper
  - `models/`: Pydantic models for data representation
- `tests/`: Test files
- `notebooks/`: Jupyter notebooks for analysis and exploration
- `data/`: output from notebooks 
