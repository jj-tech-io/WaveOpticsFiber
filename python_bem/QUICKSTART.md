# Quick Start Guide - Python BEM Solver

Get started with the BEM solver in 5 minutes!

## Step 1: Installation (1 minute)

```bash
cd python_bem
pip install -r requirements.txt
```

That's it! No compilation needed.

## Step 2: Test Installation (30 seconds)

```bash
python test_installation.py
```

You should see:
```
✓ PASS: Imports
✓ PASS: Background
✓ PASS: Wave
✓ PASS: Element
✓ PASS: Solver Creation
✓ PASS: BEM Solution
✓ PASS: Energy Balance
✓ PASS: BSDF Computation

🎉 All tests passed!
```

## Step 3: Run Your First Simulation (2 minutes)

Create a file `my_first_simulation.py`:

```python
from bem_solver import BEMSolver
import numpy as np
import matplotlib.pyplot as plt

# Define a circular glass fiber
radius = 1e-6        # 1 micrometer
n_glass = 1.5        # Refractive index
wavelength = 600e-9  # 600 nm (orange light)

# Create solver
solver = BEMSolver.from_circle(
    numel=100,
    radius=radius,
    mode='TM',
    freq=299792458.0 / wavelength,
    epsr=n_glass**2
)

# Solve!
print("Solving...")
solution = solver.solve()
print(f"✓ Solution found! Max current: {np.max(np.abs(solution)):.4e}")

# Compute scattering
print("Computing scattering...")
angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
sigma = solver.compute_bsdf(360, 1000*radius)

# Check energy balance
energy = solver.compute_energy_balance()
print(f"✓ Energy balance: {energy:.6f} (should be near 0)")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Polar plot
ax1 = plt.subplot(121, projection='polar')
ax1.plot(angles, sigma, 'k-', linewidth=2)
ax1.set_theta_zero_location('E')
ax1.set_title('Scattering Pattern')

# Cartesian plot
ax2 = plt.subplot(122)
ax2.plot(np.degrees(angles), sigma, 'k-', linewidth=2)
ax2.set_xlabel('Angle (degrees)')
ax2.set_ylabel('Intensity')
ax2.set_title('Scattering Distribution')
ax2.grid(True)

plt.tight_layout()
plt.savefig('my_first_scattering.png')
print("✓ Plot saved: my_first_scattering.png")
plt.show()
```

Run it:
```bash
python my_first_simulation.py
```

## Step 4: Try the Examples (1 minute)

```bash
cd examples

# Circular fiber
python example_circular.py

# Elliptical fiber  
python example_ellipse.py

# Custom geometry
python example_custom.py

# Complete demonstration
python example_comprehensive.py
```

## Common Tasks

### Change Wavelength

```python
# Blue light (450 nm)
wavelength = 450e-9
freq = 299792458.0 / wavelength
solver = BEMSolver.from_circle(100, 1e-6, freq=freq, epsr=2.25)
```

### Change Fiber Size

```python
# Smaller fiber (0.5 micrometers)
solver = BEMSolver.from_circle(100, 0.5e-6, freq=5e14, epsr=2.25)

# Larger fiber (5 micrometers)
solver = BEMSolver.from_circle(200, 5e-6, freq=5e14, epsr=2.25)
```

### Try Elliptical Shape

```python
# Elongated ellipse
solver = BEMSolver.from_ellipse(
    numel=120,
    radius1=2e-6,  # Major axis
    radius2=1e-6,  # Minor axis
    freq=5e14,
    epsr=2.25
)
```

### Add Material Absorption

```python
# Lossy material
n_complex = 1.55 - 0.01j  # Real - imaginary
epsr = n_complex**2
solver = BEMSolver.from_circle(100, 1e-6, freq=5e14, epsr=epsr)
```

### Compare Polarizations

```python
# TM mode
solver_tm = BEMSolver.from_circle(100, 1e-6, mode='TM', freq=5e14, epsr=2.25)
sol_tm = solver_tm.solve()
sigma_tm = solver_tm.compute_bsdf(360, 1000*solver_tm.radius)

# TE mode
solver_te = BEMSolver.from_circle(100, 1e-6, mode='TE', freq=5e14, epsr=2.25)
sol_te = solver_te.solve()
sigma_te = solver_te.compute_bsdf(360, 1000*solver_te.radius)

# Average
sigma_avg = (sigma_tm + sigma_te) / 2
```

## Next Steps

- Read the [full documentation](README.md) for theory and details
- Check [USAGE.md](USAGE.md) for advanced examples
- See [SUMMARY.md](SUMMARY.md) for complete feature list

## Troubleshooting

**Problem**: `ModuleNotFoundError: No module named 'bem_solver'`  
**Solution**: Make sure you're in the `python_bem` directory or install with `pip install -e .`

**Problem**: Simulation is slow  
**Solution**: Reduce number of elements (try 50-80 for testing)

**Problem**: Energy balance is not close to 0  
**Solution**: Increase number of elements or try quadrature=3

**Problem**: "Matrix is singular"  
**Solution**: Check your geometry is closed and increase number of elements

## Help

Need help? Check:
1. Test suite: `python test_installation.py`
2. Documentation: `README.md`
3. Examples: `examples/` directory
4. Usage guide: `USAGE.md`

## Performance Tips

- Start with 50-100 elements for quick tests
- Use 100-200 for production circular geometries
- Use 200-500 for complex shapes
- Quadrature=2 is sufficient for most cases
- Each doubling of elements increases time by ~8×

Enjoy simulating! 🎉
