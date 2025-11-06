# BEM Solver - Usage Guide

This guide provides detailed examples and best practices for using the Python BEM solver.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Geometry Definition](#geometry-definition)
3. [Material Properties](#material-properties)
4. [Wave Parameters](#wave-parameters)
5. [Solving and Analysis](#solving-and-analysis)
6. [Advanced Topics](#advanced-topics)
7. [Best Practices](#best-practices)

## Basic Usage

### Minimal Example

```python
from bem_solver import BEMSolver

# Create a circular fiber solver
solver = BEMSolver.from_circle(
    numel=100,           # Number of boundary elements
    radius=1e-6,         # 1 micrometer
    mode='TM',           # Polarization
    freq=5e14,           # 500 THz (600 nm)
    epsr=2.25            # Relative permittivity (n=1.5)
)

# Solve the system
solution = solver.solve()

# Analyze results
energy = solver.compute_energy_balance()
sigma = solver.compute_bsdf(n_theta=360, distance=1000*solver.radius)
```

## Geometry Definition

### Circular Cross-Section

```python
# Simple circular fiber
solver = BEMSolver.from_circle(
    numel=100,
    radius=1e-6,
    # ... other parameters
)
```

### Elliptical Cross-Section

```python
# Elliptical fiber
solver = BEMSolver.from_ellipse(
    numel=120,
    radius1=1.6e-6,  # Semi-major axis
    radius2=1.0e-6,  # Semi-minor axis
    # ... other parameters
)
```

### Custom Geometry from Nodes

```python
import numpy as np

# Define boundary nodes (must form closed loop)
n_points = 100
theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)

# Example: rounded square (superellipse)
n = 3  # Exponent
x = 1.5e-6 * np.sign(np.cos(theta)) * np.abs(np.cos(theta))**(2/n)
y = 1.5e-6 * np.sign(np.sin(theta)) * np.abs(np.sin(theta))**(2/n)

nodes = np.column_stack([x, y, np.zeros(n_points)])

solver = BEMSolver(nodes=nodes, mode='TM', freq=5e14, epsr=2.25)
```

### Loading Geometry from File

```python
# Load x and y coordinates
x_coords = np.loadtxt('xcoords.txt')
y_coords = np.loadtxt('ycoords.txt')

nodes = np.column_stack([x_coords, y_coords, np.zeros(len(x_coords))])
solver = BEMSolver(nodes=nodes, mode='TM', freq=5e14, epsr=2.25)
```

## Material Properties

### Lossless Dielectric

```python
# Glass fiber (n = 1.5)
n_fiber = 1.5
epsr = n_fiber**2  # Real permittivity

solver = BEMSolver.from_circle(
    numel=100,
    radius=1e-6,
    epsr=epsr,
    mur=1.0,  # Non-magnetic
    # ... other parameters
)
```

### Lossy Material (Absorption)

```python
# Material with absorption (complex refractive index)
n_real = 1.55
n_imag = 0.01  # Extinction coefficient
n_complex = n_real - 1j * n_imag
epsr = n_complex**2  # Complex permittivity

solver = BEMSolver.from_circle(
    numel=100,
    radius=1e-6,
    epsr=epsr,
    # ... other parameters
)
```

### Wavelength-Dependent Materials

```python
# Simulate multiple wavelengths with different materials
wavelengths = np.linspace(400e-9, 700e-9, 50)  # 400-700 nm
results = []

for wl in wavelengths:
    # Get wavelength-dependent refractive index
    n = get_refractive_index(wl)  # Your function
    epsr = n**2
    
    freq = 299792458.0 / wl
    solver = BEMSolver.from_circle(100, 1e-6, freq=freq, epsr=epsr)
    sol = solver.solve()
    results.append(sol)
```

## Wave Parameters

### Frequency and Wavelength

```python
# Option 1: Specify by wavelength
wavelength = 600e-9  # 600 nm
c0 = 299792458.0
freq = c0 / wavelength

# Option 2: Specify by frequency directly
freq = 5e14  # 500 THz

solver = BEMSolver.from_circle(100, 1e-6, freq=freq, epsr=2.25)
```

### Incident Angles

```python
# theta: angle from cylinder axis (0 = parallel, pi/2 = perpendicular)
# phi_i: azimuthal angle in xy-plane

# Perpendicular incidence
solver = BEMSolver.from_circle(
    100, 1e-6, 
    theta=np.pi/2,  # Perpendicular to cylinder axis
    phi_i=0.0,       # From +x direction
    freq=5e14, 
    epsr=2.25
)

# Oblique incidence at 45°
solver = BEMSolver.from_circle(
    100, 1e-6,
    theta=np.pi/4,   # 45° from z-axis
    phi_i=np.pi/4,   # 45° in xy-plane
    freq=5e14,
    epsr=2.25
)
```

### Polarization Modes

```python
# TM mode (E has z-component)
solver_tm = BEMSolver.from_circle(100, 1e-6, mode='TM', freq=5e14, epsr=2.25)

# TE mode (H has z-component)
solver_te = BEMSolver.from_circle(100, 1e-6, mode='TE', freq=5e14, epsr=2.25)

# Average both polarizations
sol_tm = solver_tm.solve()
sol_te = solver_te.solve()

sigma_tm = solver_tm.compute_bsdf(360, 1000*solver_tm.radius)
sigma_te = solver_te.compute_bsdf(360, 1000*solver_te.radius)
sigma_avg = (sigma_tm + sigma_te) / 2
```

## Solving and Analysis

### Basic Solution

```python
solver = BEMSolver.from_circle(100, 1e-6, freq=5e14, epsr=2.25)

# Solve the BEM system
solution = solver.solve()

# Solution contains surface currents
# solution[0:N]       - J_t (tangential electric current)
# solution[N:2N]     - J_z (z-component electric current)
# solution[2N:3N]    - M_t (tangential magnetic current)
# solution[3N:4N]    - M_z (z-component magnetic current)
```

### Energy Balance

```python
# Compute energy conservation
energy = solver.compute_energy_balance()

# For lossless materials: energy ≈ 0 (scattered = incident)
# For lossy materials: energy < 0 (scattered + absorbed = incident)

print(f"Energy balance: {energy:.6f}")
print(f"Absorption: {-energy*100:.2f}%")
```

### Scattering Distribution (BSDF)

```python
# Compute scattering at multiple angles
n_angles = 360
distance = 1000 * solver.radius  # Far-field distance

sigma = solver.compute_bsdf(n_angles, distance)

# sigma[i] is scattered intensity at angle i * 2π / n_angles

# Find peak scattering direction
angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
peak_idx = np.argmax(sigma)
peak_angle = angles[peak_idx]
print(f"Peak scattering at {np.degrees(peak_angle):.1f}°")
```

### Parametric Studies

```python
# Study effect of fiber radius
radii = np.logspace(-7, -5, 20)  # 0.1 to 10 micrometers
total_scattering = []

for r in radii:
    solver = BEMSolver.from_circle(100, r, freq=5e14, epsr=2.25)
    solver.solve()
    sigma = solver.compute_bsdf(360, 1000*r)
    total_scattering.append(np.sum(sigma))

# Plot size parameter vs scattering
import matplotlib.pyplot as plt
size_param = 2 * np.pi * radii / (299792458.0 / 5e14)
plt.plot(size_param, total_scattering)
plt.xlabel('Size parameter (2πr/λ)')
plt.ylabel('Total scattering')
plt.show()
```

## Advanced Topics

### Reusing Geometry for Multiple Wavelengths

```python
# Create geometry once
nodes = BEMSolver._create_elliptical_nodes(100, 1.6e-6, 1.0e-6)

# Solve for multiple wavelengths efficiently
wavelengths = [400e-9, 500e-9, 600e-9, 700e-9]
results = {}

for wl in wavelengths:
    freq = 299792458.0 / wl
    solver = BEMSolver(nodes, freq=freq, epsr=2.25)
    sol = solver.solve()
    results[wl] = {
        'solution': sol,
        'energy': solver.compute_energy_balance(),
        'bsdf': solver.compute_bsdf(360, 1000*solver.radius)
    }
```

### Updating Wave Parameters

```python
# Create solver once
solver = BEMSolver.from_circle(100, 1e-6, freq=5e14, epsr=2.25)

# Solve for multiple incident angles without rebuilding geometry
phi_angles = np.linspace(0, 2*np.pi, 36)
results = []

for phi in phi_angles:
    solver.update_wave(phi_i=phi, mode='TM', freq=5e14, theta=np.pi/2)
    sol = solver.solve()
    results.append(sol)
```

### Computing Near Fields

```python
# After solving, extract surface currents
n = solver.numel
J_t = solution[0:n]
J_z = solution[n:2*n]
M_t = solution[2*n:3*n]
M_z = solution[3*n:4*n]

# Use these to compute fields at any point (not implemented in basic version)
# See full paper for field reconstruction formulas
```

## Best Practices

### Element Count Selection

```python
# Rule of thumb: ~10 elements per wavelength along boundary
wavelength = 600e-9
radius = 1e-6
perimeter = 2 * np.pi * radius
min_elements = int(perimeter / wavelength * 10)

print(f"Recommended elements: {min_elements}")

# Typical ranges:
# - Circle, r=1μm, λ=600nm: 100-200 elements
# - Ellipse with features: 150-300 elements
# - Complex shapes: 200-500 elements
```

### Quadrature Order

```python
# Quadrature=2 (default) is sufficient for most cases
solver = BEMSolver.from_circle(100, 1e-6, quadrature=2, freq=5e14, epsr=2.25)

# Use higher orders for:
# - Very high material contrast (large |epsr|)
# - Sharp geometric features
# - Validation studies
solver_accurate = BEMSolver.from_circle(100, 1e-6, quadrature=4, freq=5e14, epsr=2.25)
```

### Convergence Testing

```python
# Test convergence with increasing resolution
n_elements = [50, 100, 200, 400]
energies = []

for n in n_elements:
    solver = BEMSolver.from_circle(n, 1e-6, freq=5e14, epsr=2.25)
    solver.solve()
    energies.append(solver.compute_energy_balance())

# Check convergence
print("Elements | Energy Balance")
for n, e in zip(n_elements, energies):
    print(f"{n:8d} | {e:14.8f}")

# Should see energy balance approach 0 for lossless case
```

### Error Checking

```python
# Check for numerical issues
solver = BEMSolver.from_circle(100, 1e-6, freq=5e14, epsr=2.25)
sol = solver.solve()

# 1. Check solution is finite
if not np.all(np.isfinite(sol)):
    print("Warning: Non-finite values in solution")

# 2. Check energy balance
energy = solver.compute_energy_balance()
if abs(energy) > 0.1:  # For lossless material
    print(f"Warning: Large energy imbalance: {energy}")

# 3. Check BSDF positivity
sigma = solver.compute_bsdf(360, 1000*solver.radius)
if np.any(sigma < 0):
    print("Warning: Negative BSDF values")
```

### Memory Considerations

```python
# Matrix size: (4*N) × (4*N) complex doubles
# Memory ≈ 16 * (4*N)^2 bytes

def estimate_memory(n_elements):
    """Estimate memory usage in MB."""
    n = 4 * n_elements
    return 16 * n * n / 1e6

print(f"100 elements: {estimate_memory(100):.1f} MB")
print(f"500 elements: {estimate_memory(500):.1f} MB")
print(f"1000 elements: {estimate_memory(1000):.1f} MB")

# For large problems, consider sparse methods or FMM acceleration
```

## Troubleshooting

### "Matrix is singular"
- Increase number of elements
- Check geometry is closed (first node ≠ last node)
- Verify material parameters are physical

### Poor Energy Conservation
- Increase number of elements
- Try higher quadrature order
- Check wavelength vs feature size

### Slow Performance
- Reduce number of elements if possible
- Use PyPy interpreter for ~2x speedup
- Consider implementing sparse/iterative solver for large problems

## Further Examples

See the `examples/` directory for complete working examples:
- `example_circular.py` - Basic circular fiber
- `example_ellipse.py` - Elliptical cross-section
- `example_custom.py` - Custom geometry

Run tests:
```bash
python test_installation.py
```
