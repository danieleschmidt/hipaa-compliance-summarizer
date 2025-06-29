# Contributing

Thank you for considering contributing to HIPAA-Compliance-Summarizer!

## Setup
1. Clone the repository and create a virtual environment.
2. Install dependencies: `pip install -r requirements.txt && pip install -e .`.
3. Install development tools: `pip install ruff bandit coverage pip-audit`.
4. Run `ruff check .`, `bandit -r src`, `coverage run -m pytest -q`, `coverage xml`,
   `coverage report -m`, and `pip-audit -r requirements.txt` before opening a pull request.

Please open issues or pull requests for discussion.
