# Contributing to Data Colada Tools

Thank you for considering contributing to Data Colada Tools! This document provides guidelines for contributing to the project.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## How Can I Contribute?

### Reporting Bugs

Before submitting a bug report:
- Check the issue tracker to see if the bug has already been reported
- If not, create a new issue using the bug report template
- Include detailed information about how to reproduce the bug

### Suggesting Features

- Check if the feature has already been suggested in the issue tracker
- If not, create a new issue using the feature request template
- Describe how your feature would help forensic data analysis

### Code Contributions

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Run the linting script (`./lint.sh`) to ensure code quality
5. Commit your changes (`git commit -m 'Add some feature'`)
6. Push to your branch (`git push origin feature/your-feature-name`)
7. Create a Pull Request

## Development Setup

1. Clone the repository
2. Run `./setup_venv.sh` to set up the virtual environment and dependencies
3. Configure your Claude API key using `python setup_api_key.py` or by editing `config.json`
4. Activate the virtual environment with `source venv/bin/activate`

## Pull Request Process

1. Update the README.md with details of changes if applicable
2. Update documentation as needed
3. Ensure your code works with Python 3.9+
4. Make sure your PR passes all CI checks
5. Be responsive to feedback during the review process

## Coding Conventions

- Use the provided Ruff configuration for code style and linting
- Write docstrings for all functions, classes, and modules
- Include unit tests for new features
- Keep the code modular and maintainable
- Follow data science best practices for numerical analysis

## Focus Areas for Contributions

We especially welcome contributions in these areas:
- Additional statistical methods for detecting data manipulation
- Improved visualization techniques for suspicious data
- Enhanced AI integration for data analysis
- Performance improvements for large datasets
- New detection methods for other types of academic fraud

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT license as specified in the LICENSE file.