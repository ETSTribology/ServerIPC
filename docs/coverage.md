# Code Coverage

## Overview

Code coverage is a critical metric that helps us understand how much of our codebase is exercised by our test suite.

## Current Coverage Status

![Coverage Badge](coverage-badge.svg)

### What is Code Coverage?

Code coverage measures the percentage of code that is executed during testing. It helps identify:
- Untested code paths
- Potential areas for additional testing
- Code quality and test comprehensiveness

### Coverage Metrics

- **Total Coverage**: Percentage of code lines executed during tests
- **Module Coverage**: Coverage for individual modules
- **Branch Coverage**: Percentage of decision branches tested

### Interpreting the Badge

- 🟢 Green: Excellent coverage (>90%)
- 🟡 Yellow: Good coverage (80-90%)
- 🔴 Red: Needs improvement (<80%)

### Improving Coverage

1. Write comprehensive unit tests
2. Cover edge cases and error scenarios
3. Use test-driven development (TDD)
4. Regularly review and update tests

## Detailed Reports

For a comprehensive breakdown of code coverage, refer to the generated HTML and XML reports.

### Generating Local Coverage Report

```bash
# Using Poetry
poetry run python scripts/generate_coverage.py

# Alternative method
poetry run pytest --cov=visualization --cov=simulation \
    --cov-report=html --cov-report=term
```

## Coverage Report Locations

- **HTML Report**: `htmlcov/index.html`
- **XML Report**: `coverage.xml`
- **Badge**: `docs/coverage-badge.svg`

## Contributing

When adding new features or fixing bugs, ensure:
- New code is covered by tests
- Existing tests are updated
- Overall coverage does not decrease

## Tools Used

- pytest-cov
- coverage.py
- genbadge
- GitHub Actions
