# Contributing to QPF

First off, thank you for considering contributing to the Quantum Protein Folding Framework! It's people like you that make QPF such a great tool for the research community.

## Code of Conduct
By participating in this project, you are expected to uphold our Code of Conduct. Please be respectful and professional in all interactions.

## How Can I Contribute?

### Reporting Bugs
If you find a bug, please open an issue and include:
*   A clear and descriptive title.
*   Steps to reproduce the problem.
*   The expected vs. actual behavior.
*   Any relevant log files or screenshots.

### Suggesting Enhancements
We welcome ideas for new features or improvements to existing ones! When suggesting an enhancement:
*   Explain the use case and why it would be beneficial.
*   Provide examples of how it might be implemented.

### Pull Requests
1.  **Fork the repo** and create your branch from `main`.
2.  **Ensure the code follows our style**: We use standard Python conventions (PEP 8).
3.  **Add tests**: If you add new functionality, please add corresponding unit tests in `tests/`.
4.  **Verify your changes**: Run the full test suite before submitting.
    ```bash
    PYTHONPATH=./src python -m pytest tests/
    ```
5.  **Documentation**: Update `README.md` or `ABOUT.md` if your changes alter the user-facing API or scientific logic.

## Project Structure
*   `src/quantum_fold/core/`: Physics and biology logic.
*   `src/quantum_fold/algorithms/`: Quantum and hybrid optimization algorithms.
*   `src/quantum_fold/utils/`: I/O, metrics, and visualization.
*   `tests/`: Unit test suite.
*   `notebooks/`: Demo and tutorial notebooks.

## Contact
For any questions, please reach out via GitHub issues or join our research community discussions.
