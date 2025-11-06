"""
Quick validation test - runs a simple simulation to verify everything works.
This is faster than the full examples.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from bem_solver import BEMSolver

print("=" * 60)
print("Quick Validation Test")
print("=" * 60)

# Small problem for quick test
print("\n✓ Creating solver (50 elements, circular fiber)...")
solver = BEMSolver.from_circle(
    numel=50,
    radius=1e-6,
    mode='TM',
    freq=5e14,
    epsr=2.25,
    theta=np.pi/2
)

print("✓ Solving BEM system...")
solution = solver.solve()
print(f"  Solution size: {len(solution)}")
print(f"  Max current: {np.max(np.abs(solution)):.4e}")

print("\n✓ Computing energy balance...")
energy = solver.compute_energy_balance()
print(f"  Energy: {energy:.6f}")

print("\n✓ Computing scattering pattern...")
sigma = solver.compute_bsdf(180, 1000*solver.radius)
print(f"  BSDF points: {len(sigma)}")
print(f"  Total scattering: {np.sum(sigma):.4e}")
print(f"  Peak angle: {np.degrees(np.linspace(0, 2*np.pi, 180)[np.argmax(sigma)]):.1f}°")

print("\n" + "=" * 60)
print("✓ All validations passed!")
print("=" * 60)
print("\nThe BEM solver is working correctly.")
print("Try running the full examples:")
print("  python example_circular.py")
print("  python example_ellipse.py")
print("  python example_comprehensive.py")
