# PyMagBot Roadmap

## Current Status (v0.0.4)
- **Math Submodules Completed:**
  - **MatrixBase/VectorBase:** Core tensor and matrix operations with GPU acceleration.
  - **Spatial Transformations:** `RotationMatrix`, `RigidTransform` and coordinate conversion methods.
  - **Spatial Primitives:** `Vertex` and `Direction`.
  - **Geometry:** Geometric objects such as `Coordinate`, `Span`, `Mesh`, `EuclideanVector`, and `VectorField`.
  - **Frame:** Hierarchical frame system.

- **Testing & Documentation:**
  - Unit tests for math and spatial submodules.
  - API documentation generated with MkDocs and mkdocstrings.

---

## Short-Term Goals (v0.0.5 – v0.0.9)

### v0.0.5: Introduce Parameter Management & Base Magnetic Object
- **New Module:** `pymagbot/devices/parameters.py`  
  - **Implement `ParamsBase`:**  
    - A base (abstract) class for managing variable parameters.
    - Consider inheriting from `dict` to facilitate serialization and easy parameter access.
  
- **New Module:** `pymagbot/devices/magnetic.py`  
  - **Implement `MagneticBase`:**  
    - A base class for magnetic objects.
    - Proposed Inheritance:  
      ```python
      class MagneticBase(GeometryBase, ParamsBase):
          # Implements common magnetic object functionality.
      ```
  
- **New Module:** `pymagbot/physics/magnetic.py`  
  - **Implement Magnetic Field Calculations:**
    - Use approximations such as point-dipole and circular loop.
    - **Anticipated Inheritance:**  
      ```python
      class MagneticField(VectorField, ParamsBase):
          # Combines field representation and parameter management.
      ```
  
- **Unit Testing & Documentation:**  
  - Write tests for `ParamsBase` and ensure that parameter-based objects can be serialized/deserialized.
  - Update API docs with usage examples for managing variable parameters.

### v0.0.6: Electromagnetic Coil Module
- **New Module:** `pymagbot/devices/coil.py`  
  - **Implement `Coil`:**
    - Inherit from `MagneticBase` to reuse geometric and parameter management functionality.
    - Support different coil shapes (cylindrical, rectangular, saddle-like).
    - Define design parameters such as current, radius, length, etc.
  
- **Naming & Inheritance Considerations:**  
  - Ensure that `Coil` only depends on its base classes (e.g., `MagneticBase` and by extension, `GeometryBase` and `ParamsBase`), avoiding circular dependencies.
  
- **Unit Testing & Documentation:**  
  - Write comprehensive tests for coil design and magnetic field generation.
  - Update documentation to show how to create and configure a coil.

### v0.0.7 – v0.0.9: Microrobots, Workspace, and Additional Enhancements
- **Microrobot Module:** `pymagbot/devices/robot.py`  
  - **Implement `Microrobot`:**  
    - A microrobot is a magnetic object with a dynamic model (position, orientation, linear and angular velocities).
    - Inherit from `MagneticBase` to combine geometric and parameter functionality.
  
- **Workspace Module:** `pymagbot/devices/workspace.py`  
  - **Implement `Workspace`:**  
    - Define a simulation environment using a `Mesh` to represent spatial grid and physical properties like density, viscosity, etc.
    - Initially assume uniform properties with potential to support non-uniform fields later.
  
- **Additional Enhancements (v0.0.8 – v0.0.9):**
  - **Plotting and Logging:**  
    - Extend `pymagbot/utils/plotting.py` for advanced visualization of frames, meshes, fields, and device configurations.
    - Enhance logging in `pymagbot/utils/logging.py` for debugging and simulation tracking.
  - **Refactor Inheritance & Naming:**  
    - Ensure all new classes follow consistent naming conventions and inherit from the appropriate base classes.
    - **Anticipated Class Names & Inheritance:**
      - `ParamsBase` (or `ParamsAbstract`) in `pymagbot/devices/parameters.py` as the base for variable parameters.
      - `MagneticBase(GeometryBase, ParamsBase)`: Base class for all magnetic objects.
      - `MagneticField(VectorField, ParamsBase)`: Represents magnetic fields with parameter management.
      - `Coil(MagneticBase)`: Specialized electromagnetic coil class.
      - `Microrobot(MagneticBase)`: Represents a microrobot with dynamic properties.
    - Review potential circular dependencies by decoupling parameters and geometry where possible.

- **Testing & Documentation:**  
  - Extend unit tests in `tests/test_devices/` and `tests/test_physics/` to cover new modules.
  - Update MkDocs documentation to include API references and usage examples for magnetic objects, coils, robots, and workspace.

---

## Medium-Term Goals (v0.1.0 and Beyond)
- **v0.1.0:**  
  - Integration of magnetic force calculations (possibly in a new module, e.g., `pymagbot/physics/forces.py`).
  - Improve simulation accuracy and numerical stability (e.g., handling small angles or near-singular configurations).
  - Begin integration with ROS2 for real-time control and communication.
  
- **v0.2.0:**  
  - Refine visualization and logging tools.
  - Enhance workspace modeling for non-uniform environments.
  - Broaden integration tests and documentation.

- **v1.0.0:**  
  - Stable release with 100% unit test coverage.
  - Complete and polished documentation with tutorials.
  - PyPI package release and official ROS2 integration support.

---

## Versioning Strategy
- **Development Versions (v0.0.x):**  
  - Increment minor versions for significant new modules/features (e.g., v0.0.5, v0.0.6, etc.).
  - Clearly document migration steps and API changes.
  
- **Pre-Releases (v0.1.x, v0.2.x):**  
  - Pre-release versions are made once core functionality (magnetic modeling, devices, workspace) is robust.
  - Gather user feedback for further improvements.
  
- **Stable Release (v1.0.0):**  
  - Once testing and documentation are complete, release v1.0.0.
  - Use semantic versioning (MAJOR.MINOR.PATCH).
