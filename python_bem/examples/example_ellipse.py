"""
Example: Scattering from an elliptical fiber.

This example demonstrates how to use the BEM solver to simulate
electromagnetic scattering from an elliptical cross-section fiber.
"""
import sys
import os
# Add parent directory to path to import bem_solver
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from bem_solver import BEMSolver

def main():
    print("=" * 60)
    print("BEM Scattering Simulation - Elliptical Fiber")
    print("=" * 60)
    
    # Fiber parameters
    radius_major = 1.6e-6  # 1.6 micrometers (semi-major axis)
    radius_minor = 1.0e-6  # 1.0 micrometer (semi-minor axis)
    n_elements = 120  # More elements for ellipse
    
    # Material properties
    n_fiber = 1.55  # Refractive index
    epsr = n_fiber**2
    
    # Wave parameters
    wavelength = 600e-9  # 600 nm
    c0 = 299792458.0
    freq = c0 / wavelength
    theta = np.pi / 2  # Perpendicular incidence
    
    print(f"\nFiber semi-major axis: {radius_major*1e6:.2f} μm")
    print(f"Fiber semi-minor axis: {radius_minor*1e6:.2f} μm")
    print(f"Aspect ratio: {radius_major/radius_minor:.2f}")
    print(f"Refractive index: {n_fiber}")
    print(f"Wavelength: {wavelength*1e9:.1f} nm")
    print(f"Number of elements: {n_elements}")
    
    # Test multiple incident angles
    phi_angles = [0, np.pi/4, np.pi/2]  # 0°, 45°, 90°
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, phi_i in enumerate(phi_angles):
        print("\n" + "-" * 60)
        print(f"Incident angle: θ={np.degrees(theta):.1f}°, φ={np.degrees(phi_i):.1f}°")
        
        # Solve TM mode
        solver_tm = BEMSolver.from_ellipse(
            numel=n_elements,
            radius1=radius_major,
            radius2=radius_minor,
            quadrature=2,
            phi_i=phi_i,
            mode='TM',
            freq=freq,
            mur=1.0,
            epsr=epsr,
            theta=theta
        )
        
        sol_tm = solver_tm.solve()
        print(f"TM mode solved. Max current: {np.max(np.abs(sol_tm)):.4e}")
        
        energy_tm = solver_tm.compute_energy_balance()
        print(f"Energy balance (TM): {energy_tm:.6f}")
        
        # Compute scattering
        n_angles = 360
        distance = 1000 * radius_major
        sigma_tm = solver_tm.compute_bsdf(n_angles, distance)
        
        # Solve TE mode
        solver_te = BEMSolver.from_ellipse(
            numel=n_elements,
            radius1=radius_major,
            radius2=radius_minor,
            quadrature=2,
            phi_i=phi_i,
            mode='TE',
            freq=freq,
            mur=1.0,
            epsr=epsr,
            theta=theta
        )
        
        sol_te = solver_te.solve()
        print(f"TE mode solved. Max current: {np.max(np.abs(sol_te)):.4e}")
        
        energy_te = solver_te.compute_energy_balance()
        print(f"Energy balance (TE): {energy_te:.6f}")
        
        sigma_te = solver_te.compute_bsdf(n_angles, distance)
        sigma_avg = (sigma_tm + sigma_te) / 2
        
        # Plot on polar subplot
        angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
        ax = plt.subplot(2, 2, idx+1, projection='polar')
        ax.plot(angles, sigma_avg, 'k-', linewidth=2)
        ax.set_theta_zero_location('E')
        ax.set_title(f'φᵢ = {np.degrees(phi_i):.0f}°', fontsize=14, pad=20)
        ax.grid(True)
        
        # Mark incident direction
        ax.plot([phi_i], [0], 'ro', markersize=10, label='Incident')
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        
        print(f"Total scattering: {np.sum(sigma_avg):.4e}")
    
    # Plot geometry in last subplot
    ax = axes[3]
    ax.axis('off')
    
    # Create a separate axes for the geometry
    ax_geom = fig.add_subplot(2, 2, 4)
    
    # Plot ellipse shape
    theta_plot = np.linspace(0, 2*np.pi, 100)
    x = radius_major * np.cos(theta_plot) * 1e6
    y = radius_minor * np.sin(theta_plot) * 1e6
    ax_geom.plot(x, y, 'b-', linewidth=2)
    ax_geom.fill(x, y, alpha=0.3, color='blue')
    
    # Mark incident directions
    for phi_i in phi_angles:
        arrow_len = 2.5
        dx = -arrow_len * np.cos(phi_i)
        dy = -arrow_len * np.sin(phi_i)
        start_x = 3 * np.cos(phi_i)
        start_y = 3 * np.sin(phi_i)
        ax_geom.arrow(start_x, start_y, dx, dy, 
                     head_width=0.3, head_length=0.2, 
                     fc='red', ec='red', linewidth=2)
        ax_geom.text(start_x + 0.5*dx, start_y + 0.5*dy, 
                    f'{np.degrees(phi_i):.0f}°',
                    fontsize=10, ha='center')
    
    ax_geom.set_xlim(-4, 4)
    ax_geom.set_ylim(-4, 4)
    ax_geom.set_aspect('equal')
    ax_geom.set_xlabel('x (μm)', fontsize=12)
    ax_geom.set_ylabel('y (μm)', fontsize=12)
    ax_geom.set_title('Fiber Geometry & Incident Directions', fontsize=14)
    ax_geom.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('elliptical_fiber_scattering.png', dpi=150, bbox_inches='tight')
    print("\n" + "-" * 60)
    print("Plot saved as 'elliptical_fiber_scattering.png'")
    plt.show()
    
    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
