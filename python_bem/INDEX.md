# Python BEM Solver - File Index

Complete index of all files in the Python BEM solver package.

## 📁 Project Structure

```
python_bem/
├── 📄 README.md                    # Main documentation (START HERE!)
├── 📄 QUICKSTART.md                # 5-minute quick start guide
├── 📄 USAGE.md                     # Detailed usage examples
├── 📄 SUMMARY.md                   # Project summary and features
├── 📄 TEST.md                      # Testing instructions
├── 📄 requirements.txt             # Python dependencies
├── 📄 setup.py                     # Package installation
├── 🧪 test_installation.py         # Automated test suite
│
├── 📦 bem_solver/                  # Main package
│   ├── __init__.py                 # Package initialization
│   ├── background.py               # Material properties
│   ├── wave.py                     # Wave parameters
│   ├── element.py                  # Boundary elements
│   ├── basis.py                    # Basis functions
│   ├── solver.py                   # Main BEM solver (CORE)
│   └── visualization.py            # Plotting utilities
│
└── 📂 examples/                    # Example scripts
    ├── example_circular.py         # Circular fiber
    ├── example_ellipse.py          # Elliptical fiber
    ├── example_custom.py           # Custom geometry
    └── example_comprehensive.py    # Complete demo
```

## 📖 Documentation Files

### Start Here
- **README.md** (Main documentation)
  - Overview and features
  - Installation instructions
  - Theory and mathematics
  - API reference
  - Validation and benchmarks

### Quick Start
- **QUICKSTART.md** (5-minute tutorial)
  - Minimal setup
  - First simulation
  - Common tasks
  - Troubleshooting

### Detailed Guides
- **USAGE.md** (Complete usage guide)
  - Geometry definition
  - Material properties
  - Wave parameters
  - Analysis methods
  - Advanced topics
  - Best practices

### Reference
- **SUMMARY.md** (Project summary)
  - What has been implemented
  - Feature checklist
  - Comparison with C++ version
  - Performance benchmarks
  - File structure

- **TEST.md** (Testing info)
  - How to run tests
  - What is tested
  - Expected output

## 🔧 Core Package Files

### bem_solver/__init__.py
- Package initialization
- Exports main classes
- Version information

### bem_solver/background.py (~50 lines)
**Purpose**: Material properties and physical constants

**Key class**: `Background`
- Vacuum/air properties (c₀, μ₀, ε₀, Z₀)
- Scatterer properties (μ₁, ε₁, c₁, Z₁)
- Handles both lossless and lossy materials

**Usage**:
```python
from bem_solver import Background
bg = Background(mur=1.0, epsr=2.25)  # Glass
```

### bem_solver/wave.py (~60 lines)
**Purpose**: Incident wave parameters

**Key class**: `Wave`
- Frequency and wavelength
- Wave vectors (k₀, k_z, k_transverse)
- Polarization mode (TM/TE)
- Incident angles

**Usage**:
```python
from bem_solver import Wave
wave = Wave(background, phi_i=0, mode='TM', freq=5e14, theta=np.pi/2)
```

### bem_solver/element.py (~80 lines)
**Purpose**: Boundary element discretization

**Key class**: `Element`
- Line segment on boundary
- Quadrature points
- Normal vectors
- Tangent vectors

**Usage**:
```python
el = Element(node1, node2, pos1, pos2, basis_indices, quad_points)
```

### bem_solver/basis.py (~30 lines)
**Purpose**: Basis functions for unknowns

**Key class**: `Basis`
- Associates with two elements
- Piecewise linear basis functions

### bem_solver/solver.py (~600 lines) ⭐ CORE
**Purpose**: Main BEM solver implementation

**Key class**: `BEMSolver`
- Matrix assembly (dense MoM)
- System solution (LU decomposition)
- Post-processing (BSDF, energy balance)

**Main methods**:
- `__init__()` - Initialize from nodes
- `from_circle()` - Create circular geometry
- `from_ellipse()` - Create elliptical geometry
- `assemble_system_matrix()` - Build impedance matrix Z
- `compute_incident_field()` - Build excitation vector
- `solve()` - Solve Z·x = b
- `compute_energy_balance()` - Verify conservation
- `compute_bsdf()` - Calculate scattering distribution

**Usage**:
```python
solver = BEMSolver.from_circle(100, 1e-6, freq=5e14, epsr=2.25)
solution = solver.solve()
```

### bem_solver/visualization.py (~200 lines)
**Purpose**: Plotting and visualization utilities

**Key functions**:
- `plot_geometry()` - Plot fiber shape
- `plot_scattering_polar()` - Polar scattering plot
- `plot_scattering_cartesian()` - Cartesian plot
- `plot_comparison()` - Compare multiple results
- `plot_full_results()` - Comprehensive visualization
- `save_results()` - Export to files

**Usage**:
```python
from bem_solver.visualization import plot_full_results
fig = plot_full_results(solver, solution, sigma, angles)
```

## 🧪 Test File

### test_installation.py (~250 lines)
**Purpose**: Comprehensive test suite

**Tests**:
1. Module imports
2. Background class
3. Wave class
4. Element class
5. Solver creation
6. BEM solution
7. Energy balance
8. BSDF computation

**Run**: `python test_installation.py`

## 📚 Example Files

### example_circular.py (~150 lines)
**What it does**:
- Simulates circular glass fiber
- Computes TM and TE modes
- Calculates scattering patterns
- Plots polar and Cartesian results
- Verifies energy balance

**Output**: `circular_fiber_scattering.png`

**Run time**: ~30 seconds

### example_ellipse.py (~180 lines)
**What it does**:
- Elliptical fiber (aspect ratio 1.6)
- Multiple incident angles (0°, 45°, 90°)
- Compares scattering patterns
- Shows geometry diagram

**Output**: `elliptical_fiber_scattering.png`

**Run time**: ~2 minutes

### example_custom.py (~150 lines)
**What it does**:
- Custom geometry (rounded square)
- Surface current visualization
- Scattering analysis

**Output**: `custom_fiber_scattering.png`

**Run time**: ~40 seconds

### example_comprehensive.py (~300 lines) ⭐
**What it does**:
- Wavelength sweep (400-700 nm)
- Geometry comparison (3 shapes)
- Polarization comparison (TM vs TE)
- Incident angle sweep
- Complete visualization suite

**Output**: 
- `wavelength_sweep.png`
- `geometry_comparison.png`
- `polarization_comparison.png`
- `angle_sweep.png`

**Run time**: ~5-10 minutes

## 📦 Configuration Files

### requirements.txt
```
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.3.0
```

### setup.py
Standard Python package setup using setuptools

## 🎯 Quick Navigation Guide

**I want to...**

- **Get started quickly** → Read `QUICKSTART.md`
- **Understand the physics** → Read `README.md` Theory section
- **Learn all features** → Read `USAGE.md`
- **See the code** → Look at `bem_solver/solver.py`
- **Run examples** → Go to `examples/` directory
- **Test installation** → Run `test_installation.py`
- **Modify the solver** → Edit `bem_solver/solver.py`
- **Add new features** → See `SUMMARY.md` for ideas

## 📊 File Statistics

| Category | Files | Lines of Code |
|----------|-------|---------------|
| Core Package | 7 | ~1100 |
| Examples | 4 | ~680 |
| Tests | 1 | ~250 |
| Documentation | 5 | ~2500 |
| **Total** | **17** | **~4530** |

## 🔑 Key Files by Purpose

**To understand the math**: 
- `README.md` (Theory section)
- `bem_solver/solver.py` (Implementation)

**To get running quickly**:
- `QUICKSTART.md`
- `test_installation.py`
- `examples/example_circular.py`

**To modify/extend**:
- `bem_solver/solver.py` (Core algorithm)
- `bem_solver/visualization.py` (Plotting)
- `examples/example_comprehensive.py` (Advanced usage)

**To verify correctness**:
- `test_installation.py`
- Energy balance checks in examples

## 🎓 Learning Path

1. **Beginner**: Start with `QUICKSTART.md` → Run `example_circular.py`
2. **Intermediate**: Read `USAGE.md` → Try modifying examples
3. **Advanced**: Study `solver.py` → Read `README.md` theory
4. **Expert**: Extend the code → Add new features

---

**Last updated**: November 2025  
**Total implementation**: ~4500 lines of code + documentation
