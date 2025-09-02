# Contributing to Trading Bot Simulator

Thank you for your interest in contributing to Trading Bot Simulator! This document provides guidelines for contributing to the project.

## Code of Conduct

This project and its participants are governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

- Use the GitHub issue tracker
- Include a clear and descriptive title
- Provide detailed steps to reproduce the bug
- Include system information (OS, Python version, etc.)
- Include error messages and stack traces

### Suggesting Enhancements

- Use the GitHub issue tracker
- Describe the enhancement clearly
- Explain why this enhancement would be useful
- Include mockups or examples if applicable

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`make test`)
6. Run linting (`make lint`)
7. Run type checking (`make typecheck`)
8. Commit your changes (`git commit -m 'Add amazing feature'`)
9. Push to the branch (`git push origin feature/amazing-feature`)
10. Open a Pull Request

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/trading-bot-simulator.git
   cd trading-bot-simulator
   ```

2. Install dependencies:
   ```bash
   make setup
   ```

3. Run tests:
   ```bash
   make test
   ```

## Code Style

We use the following tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **Ruff**: Linting
- **MyPy**: Type checking

Run all checks with:
```bash
make check
```

## Testing

- Write tests for all new functionality
- Ensure test coverage remains above 80%
- Run tests with: `make test`
- Run smoke test with: `make smoke`

## Documentation

- Update documentation for any API changes
- Add docstrings to new functions and classes
- Keep README.md up to date

## Commit Messages

Use clear and descriptive commit messages:

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests after the first line

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
