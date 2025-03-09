# TGOV Scraper

A set of tools for scraping and analyzing data from the Tulsa Government Access Television (TGOV) website.

## Setup

This project uses Poetry for dependency management.

```bash
# Install dependencies
poetry install

# Activate the virtual environment
poetry shell

# Install Jupyter kernel for this environment (needed for Jupyter notebooks)
poetry run python -m ipykernel install --user --name=tgov-scraper --display-name="TGOV Scraper"
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
- 'scripts`: one off scripts for downloading, conversions, etc
- `tests/`: Test files
- `notebooks/`: Jupyter notebooks for analysis and exploration
- `data/`: output from notebooks
  - `audio`: audio output from videos

  pip install assemblyai moviepy