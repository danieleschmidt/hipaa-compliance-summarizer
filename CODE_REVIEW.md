# Code Review - Add document type usage example

## Engineer Review
- **ruff check .** shows no issues.
- **bandit -r src** reports no security issues.
- **pytest -q** runs all tests successfully (10 tests).
- Implementation is straightforward with no performance concerns.

## Product Manager Review
- Sprint board shows all tasks completed, including README update for document type handling.
- Acceptance tests confirm document models, parsers, processor routing, and README example.

Overall the feature meets the acceptance criteria and passes all quality checks.
