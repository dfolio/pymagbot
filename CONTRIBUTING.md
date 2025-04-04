# Contributing

Contributions are welcome, and they are greatly appreciated!
Every little bit helps, and credit will always be given.
Whether you are fixing a bug, adding a new feature, improving documentation, or enhancing our test suite, your efforts are greatly appreciated. All contributors will receive credit for their work.
Please follow the guidelines below to ensure a smooth contribution process.

## Codebase structure

The PyMagBot repository is organized to promote modularity and clarity.
The codebase is organized into several modules, each with its own purpose:

- `pymagbot/`  
  - `__init__.py` – Package initialization.
  - `math/` – Mathematical operations, transformations, and geometry.
    - `matrix_base.py` – Base class for matrix and tensor operations.
    - `spatial/` – Spatial transformations (e.g., `RotationMatrix`, `RigidTransform`).
    - `spatial_primitives.py` – Basic spatial primitives such as `Vertex` and `Direction`.
    - `frame.py` – Frame transformations and hierarchy.
    - `geometry/` – Geometric objects:
      - `coordinate.py` – Coordinate system definitions.
      - `span.py` – Span (mesh interval) representation.
      - `mesh.py` – Mesh/grid representation.
      - `vector.py` – Euclidean vector operations.
      - `vector_field.py` – Vector field definitions.
      - `operations.py` – Helper functions for coordinate conversions and vector operations.
  - `utils/` – General utility modules:
    - `io.py` – Serialization/deserialization, save/load functionality.
    - `plotting.py` – Visualization utilities.
    - `config.py` – Global configuration settings.
    - `logging.py` – Logging utilities.
    - `validation.py` – Input validation functions.
  - `physics/` – Physical modeling:
    - `magnetic_field.py` – Magnetic field calculations.
    - `forces.py` – Magnetic forces modeling.
    - `fluid_dynamics.py` – Fluid interactions.
    - `motion_control.py` – Motion planning and control.
  - `devices/` – Device and hardware-related models:
    - `magnetic.py` – Base magnetic object (abstract class).
    - `parameters.py` – Parameter management (`ParamsBase`).
    - `coil.py` – Electromagnetic coil (electromagnet) modeling.
    - `robot.py` – Microrobot dynamic model.
    - `workspace.py` – Simulation workspace definition.
- `docs/` – Documentation (MkDocs).
- `scripts/` – Example scripts and usage demos.
- `tests/` – Unit tests organized to mirror the main codebase structure.
- `mkdocs.yml` – MkDocs configuration.
- `pyproject.toml` – Modern package configuration.
- `README.md` – Project overview.
- `LICENSE` – License file (GPLv3).
- `CHANGELOG.md` – Change history.

## Documentation Guidelines

- **Docstrings:**  
  All code should be documented using the .  
- **API Documentation:**  
  We use [MkDocs](https://www.mkdocs.org/) and [mkdocstrings](https://mkdocstrings.github.io/) to generate API documentation in Markdown format with LaTeX support (via [MathJax](https://www.mathjax.org/)).
- **Style:**  
  Write clear, concise, and descriptive docstrings that explain the purpose, parameters, returns, and any exceptions.
  When applicable, include usage examples.

## Unit testing

- All tests are written using Python’s built-in [unittest](https://docs.python.org/3/library/unittest.html) framework.
- Tests are organized in the `tests/` directory, mirroring the project structure.
- Run tests locally with:
  
  ```bash
  python -m unittest discover -s tests
  ```
  
## Commit message convention

Please follow our commit message guidelines based on the [Karma convention](https://karma-runner.github.io/4.
0/dev/git-commit-msg.html):

```bash
<type>[(scope)]: Subject

[Body]
```

Scope and body are optional.
If you include a body, ensure it’s written in valid Markdown and includes any necessary references (e.g., issue numbers, co-authors).

Type include:
- `feat`: New feature.
- `fix`: Bug fix.
- `docs`: About documentation.
- `style`: A change in code style/format.
- `refactor`: Changes that are not features or bug fixes.
- `perf`: About performance.
- `tests`: About tests.
- `chore`: About packaging or repo/files management.
- `ci`: About Continuous Integration.
- `deps`: Dependencies update.

Subject Guidelines:
- Subject and body must be valid Markdown.
- Use a concise summary of the changes.
- Start with a capital letter; do not end with a period.
- Write in imperative mood (e.g., “Fix bug”, “Add feature”).

## Pull Request Guidelines
- Fork the repository and create your branch from main.
- Write tests for your changes. Ensure all tests pass before submitting your PR.
- Follow the commit message convention and document your changes in the PR description.
- Submit your PR and link any related issues.

# Coding Standards and Best Practices
- Write modular, well-documented code.
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for code style.
  We recommend using tools like `flake8` or `black` for formatting.
- Avoid circular dependencies by designing clear module boundaries.
- Strive for 100% test coverage, and add tests for new features and edge cases.
- Use type hints and modern Python syntax (e.g., f-strings, dataclasses when appropriate) to enhance readability.

Additional Resources
- [PEP 8 – Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)
- [NumPy docstring guide](https://numpydoc.readthedocs.io/en/latest/format.html)
- [MkDocs Documentation](https://www.mkdocs.org/)
- [GitHub Flow](https://guides.github.com/introduction/flow/)