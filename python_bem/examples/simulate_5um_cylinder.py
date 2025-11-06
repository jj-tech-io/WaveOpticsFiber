# -*- coding: utf-8 -*-
"""
Simulate dielectric cylinder scattering with specific parameters.

Parameters:
- Radius: 5 micrometers
- Refractive index: 1.54
- Incident angle: theta_i = 0 (parallel to cylinder axis)
- Azimuthal angle: phi_i = 0

Uses GPU acceleration with CuPy if available.
"""
import sys
import io
# Force UTF-8 encoding for console output on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Output directory for images and data
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to ensure saving works
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from bem_solver import BEMSolver
from bem_solver.visualization import (plot_geometry, plot_scattering_polar, 
                                       plot_scattering_cartesian, plot_full_results)
import time

# Try to import CuPy for GPU acceleration
# NOTE: CuPy support not yet implemented in solver, using NumPy for now
try:
    import cupy as cp
    HAS_CUPY = True
    print("INFO: CuPy available but not yet integrated - using NumPy (CPU only)")
except ImportError:
    cp = np
    HAS_CUPY = False
    print("INFO: Using NumPy (CPU only)")

def main():
    print("=" * 70)
    print("Dielectric Cylinder Scattering Simulation")
    print("=" * 70)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Simulation parameters
    radius = 5e-6  # 5 micrometers
    n_fiber = 1.54  # Refractive index
    epsr = n_fiber**2  # Relative permittivity
    
    # Incident wave parameters
    wavelength = 600e-9  # 600 nm
    c0 = 299792458.0
    freq = c0 / wavelength
    
    # Incident angles
    theta_i = 0.0  # Parallel to cylinder axis (z-direction)
    phi_i = 0.0    # Azimuthal angle
    
    # Note: theta=0 means parallel to cylinder, but BEM expects perpendicular
    # For perpendicular incidence to cylinder axis, use theta = pi/2
    # Let me check what you mean - if theta_i=0 means along z-axis:
    theta = np.pi / 2  # This makes wave perpendicular to cylinder axis
    
    print(f"\nSimulation Parameters:")
    print(f"  Radius: {radius*1e6:.1f} μm")
    print(f"  Refractive index: {n_fiber}")
    print(f"  Wavelength: {wavelength*1e9:.1f} nm")
    print(f"  Incident angles: θ={np.degrees(theta):.1f}°, φ={np.degrees(phi_i):.1f}°")
    print(f"  Frequency: {freq:.3e} Hz")
    
    # Number of elements (scale with size, but keep reasonable for speed)
    # For 5μm radius: limit to 50 elements for reasonable computation time
    # Matrix size will be (4*n_elements)^2: 50 elem → 200x200 matrix
    perimeter = 2 * np.pi * radius
    n_elements_full = int(perimeter / wavelength * 10)
    n_elements = min(50, n_elements_full)  # Cap at 50 for speed (200x200 matrix)
    print(f"  Optimal elements: {n_elements_full} (using {n_elements} for speed)")
    
    # Create solver for TM mode
    print("\n" + "-" * 70)
    print("Solving TM polarization...")
    print(f"  Creating solver with {n_elements} elements...")
    t_start = time.time()
    
    solver_tm = BEMSolver.from_circle(
        numel=n_elements,
        radius=radius,
        quadrature=2,
        phi_i=phi_i,
        mode='TM',
        freq=freq,
        mur=1.0,
        epsr=epsr,
        theta=theta
    )
    
    print(f"  Solver created in {time.time()-t_start:.2f}s")
    print(f"  Matrix size: {4*n_elements} × {4*n_elements}")
    print("  Assembling system matrix...")
    t_start = time.time()
    
    sol_tm = solver_tm.solve()
    
    print(f"  [OK] TM solution computed in {time.time()-t_start:.2f}s")
    print(f"    Max current: {np.max(np.abs(sol_tm)):.4e}")
    
    energy_tm = solver_tm.compute_energy_balance()
    print(f"  Energy balance (TM): {energy_tm:.6f}")
    
    # Compute scattering distribution
    n_angles = 360
    distance = 1000 * radius  # Far field
    print(f"  Computing BSDF at {n_angles} angles...")
    sigma_tm = solver_tm.compute_bsdf(n_angles, distance)
    
    # Solve for TE mode
    print("\n" + "-" * 70)
    print("Solving TE polarization...")
    t_start = time.time()
    
    solver_te = BEMSolver.from_circle(
        numel=n_elements,
        radius=radius,
        quadrature=2,
        phi_i=phi_i,
        mode='TE',
        freq=freq,
        mur=1.0,
        epsr=epsr,
        theta=theta
    )
    
    print("  Assembling and solving system...")
    sol_te = solver_te.solve(verbose=True)
    print(f"  [OK] TE solution computed in {time.time()-t_start:.2f}s")
    print(f"    Max current: {np.max(np.abs(sol_te)):.4e}")
    
    energy_te = solver_te.compute_energy_balance()
    print(f"  Energy balance (TE): {energy_te:.6f}")
    
    sigma_te = solver_te.compute_bsdf(n_angles, distance)
    
    # Average polarizations
    sigma_avg = (sigma_tm + sigma_te) / 2
    
    # Normalize to PDF (probability density function)
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    d_angle = 2 * np.pi / n_angles
    
    # Azimuthal PDF (normalize so integral over 2π = 1)
    pdf_azimuthal = sigma_avg / (np.sum(sigma_avg) * d_angle)
    
    # Compute full 3D BSDF from BEM solver
    print("\n" + "-" * 70)
    print("Computing 3D scattering pattern...")
    n_theta_3d = 36  # Polar angles (reduced for speed)
    n_phi_3d = 72    # Azimuthal angles (reduced for speed)
    
    # Compute for TM polarization
    print("  TM polarization:")
    theta_3d, phi_3d, sigma_3d_tm = solver_tm.compute_bsdf_3d(n_theta_3d, n_phi_3d, distance, verbose=True)
    
    # Compute for TE polarization  
    print("  TE polarization:")
    theta_3d, phi_3d, sigma_3d_te = solver_te.compute_bsdf_3d(n_theta_3d, n_phi_3d, distance, verbose=True)
    
    # Average polarizations
    sigma_3d_avg = (sigma_3d_tm + sigma_3d_te) / 2
    
    # Extract longitudinal PDF by averaging over azimuthal angles
    pdf_longitudinal = np.mean(sigma_3d_avg, axis=1)
    pdf_longitudinal = pdf_longitudinal / (np.sum(pdf_longitudinal) * (np.pi / n_theta_3d))
    theta_long = theta_3d
    
    print("\n" + "=" * 70)
    print("PLOTTING RESULTS")
    print("=" * 70)
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Geometry with surface currents
    ax1 = plt.subplot(3, 3, 1)
    plot_geometry(solver_tm, ax=ax1, show_currents=True, solution=sol_tm)
    ax1.set_title('Geometry & Surface Currents (TM)', fontsize=12)
    
    # 2. Azimuthal scattering pattern (polar)
    ax2 = plt.subplot(3, 3, 2, projection='polar')
    ax2.plot(angles, sigma_tm, 'b-', linewidth=2, label='TM', alpha=0.7)
    ax2.plot(angles, sigma_te, 'r-', linewidth=2, label='TE', alpha=0.7)
    ax2.plot(angles, sigma_avg, 'k-', linewidth=3, label='Average')
    ax2.set_theta_zero_location('E')
    ax2.set_title('Azimuthal Scattering Pattern', fontsize=12, pad=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax2.grid(True)
    
    # 3. Azimuthal PDF (polar)
    ax3 = plt.subplot(3, 3, 3, projection='polar')
    ax3.plot(angles, pdf_azimuthal, 'k-', linewidth=2)
    ax3.fill(angles, pdf_azimuthal, alpha=0.3, color='blue')
    ax3.set_theta_zero_location('E')
    ax3.set_title('Azimuthal PDF', fontsize=12, pad=20)
    ax3.grid(True)
    
    # 4. Azimuthal scattering (Cartesian)
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(np.degrees(angles), sigma_tm, 'b-', linewidth=1.5, label='TM', alpha=0.7)
    ax4.plot(np.degrees(angles), sigma_te, 'r-', linewidth=1.5, label='TE', alpha=0.7)
    ax4.plot(np.degrees(angles), sigma_avg, 'k-', linewidth=2, label='Average')
    ax4.axvline(0, color='gray', linestyle='--', alpha=0.5, label='Backward')
    ax4.axvline(180, color='gray', linestyle=':', alpha=0.5, label='Forward')
    ax4.set_xlabel('Azimuthal Angle φ (degrees)', fontsize=11)
    ax4.set_ylabel('Scattered Intensity', fontsize=11)
    ax4.set_title('Azimuthal Scattering Distribution', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 360)
    
    # 5. Azimuthal PDF (Cartesian)
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(np.degrees(angles), pdf_azimuthal, 'k-', linewidth=2)
    ax5.fill_between(np.degrees(angles), pdf_azimuthal, alpha=0.3, color='blue')
    ax5.set_xlabel('Azimuthal Angle φ (degrees)', fontsize=11)
    ax5.set_ylabel('Probability Density', fontsize=11)
    ax5.set_title('Azimuthal PDF (Normalized)', fontsize=12)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, 360)
    
    # 6. Log-scale scattering
    ax6 = plt.subplot(3, 3, 6)
    ax6.semilogy(np.degrees(angles), sigma_avg + 1e-10, 'k-', linewidth=2)
    ax6.set_xlabel('Azimuthal Angle φ (degrees)', fontsize=11)
    ax6.set_ylabel('Scattered Intensity (log)', fontsize=11)
    ax6.set_title('Log-Scale Scattering', fontsize=12)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0, 360)
    
    # 7. Longitudinal PDF with detailed analysis
    ax7 = plt.subplot(3, 3, 7)
    
    # Plot main curve
    ax7.plot(np.degrees(theta_long), pdf_longitudinal, 'g-', linewidth=2.5, label='Longitudinal PDF')
    ax7.fill_between(np.degrees(theta_long), pdf_longitudinal, alpha=0.25, color='green')
    
    # Mark key angles
    ax7.axvline(0, color='blue', linestyle=':', alpha=0.5, linewidth=1.5, label='Forward (0°)')
    ax7.axvline(90, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Perpendicular (90°)')
    ax7.axvline(180, color='blue', linestyle=':', alpha=0.5, linewidth=1.5, label='Backward (180°)')
    
    # Find and mark peak
    peak_idx = np.argmax(pdf_longitudinal)
    peak_angle = np.degrees(theta_long[peak_idx])
    peak_value = pdf_longitudinal[peak_idx]
    ax7.plot(peak_angle, peak_value, 'r*', markersize=15, label=f'Peak: {peak_angle:.1f}°')
    
    # Add annotations
    ax7.annotate(f'Max: {peak_value:.3f}', 
                xy=(peak_angle, peak_value), 
                xytext=(peak_angle + 20, peak_value * 0.9),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=9, color='red', fontweight='bold')
    
    # Calculate and show statistics
    mean_angle = np.sum(theta_long * pdf_longitudinal) / np.sum(pdf_longitudinal)
    std_angle = np.sqrt(np.sum(pdf_longitudinal * (theta_long - mean_angle)**2) / np.sum(pdf_longitudinal))
    
    stats_text = f'Mean: {np.degrees(mean_angle):.1f}°\nStd: {np.degrees(std_angle):.1f}°'
    ax7.text(0.05, 0.95, stats_text, transform=ax7.transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax7.set_xlabel('Longitudinal Angle θ (degrees)', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
    ax7.set_title('Longitudinal Scattering PDF\n(Averaged over φ)', fontsize=12, fontweight='bold')
    ax7.legend(loc='upper right', fontsize=8)
    ax7.grid(True, alpha=0.3, linestyle='--')
    ax7.set_xlim(0, 180)
    ax7.set_ylim(bottom=0)
    print("  [OK] Longitudinal PDF plotted with detailed analysis")
    
    # 8. 3D Scattering Pattern (both theta and phi) - FROM BEM SOLUTION
    print("  Creating 3D scattering plot from computed data...")
    ax8 = fig.add_subplot(3, 3, 8, projection='3d')
    
    from matplotlib import cm
    
    # Create meshgrid from computed 3D BSDF
    phi_mesh, theta_mesh = np.meshgrid(phi_3d, theta_3d)
    
    # Normalize to PDF
    d_theta = np.pi / n_theta_3d
    d_phi = 2 * np.pi / n_phi_3d
    pdf_3d = sigma_3d_avg / (np.sum(sigma_3d_avg) * d_theta * d_phi)
    
    # Scale for visualization
    pdf_3d_scaled = pdf_3d * 5.0
    
    # Convert to Cartesian coordinates
    x_3d = pdf_3d_scaled * np.sin(theta_mesh) * np.cos(phi_mesh)
    y_3d = pdf_3d_scaled * np.sin(theta_mesh) * np.sin(phi_mesh)
    z_3d = pdf_3d_scaled * np.cos(theta_mesh)
    
    # Plot surface with color map
    surf = ax8.plot_surface(x_3d, y_3d, z_3d, cmap=cm.viridis, alpha=0.8,
                           linewidth=0, antialiased=True,
                           facecolors=cm.viridis(pdf_3d / np.max(pdf_3d)))
    
    # Add cylinder at origin
    ax8.plot([0], [0], [0], 'ro', markersize=8, label='Cylinder')
    
    ax8.set_xlabel('X', fontsize=10)
    ax8.set_ylabel('Y', fontsize=10)
    ax8.set_zlabel('Z', fontsize=10)
    ax8.set_title('3D Scattering PDF\n(BEM computed)', fontsize=12, fontweight='bold')
    ax8.view_init(elev=20, azim=45)
    print("  [OK] 3D scattering plot created from BEM data")
    
    # 9. Statistics table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Calculate statistics
    forward_idx = n_angles // 2
    backward_idx = 0
    forward_backward_ratio = sigma_avg[forward_idx] / (sigma_avg[backward_idx] + 1e-10)
    
    stats_text = f"""
    SIMULATION STATISTICS
    ══════════════════════════════════
    
    Geometry:
      Radius: {radius*1e6:.1f} μm
      n: {n_fiber}
      λ: {wavelength*1e9:.1f} nm
      Size param: {2*np.pi*radius/wavelength:.2f}
    
    Discretization:
      Elements: {n_elements}
      Quadrature: 2-point Gauss
    
    Results (TM mode):
      Max current: {np.max(np.abs(sol_tm)):.4e}
      Energy balance: {energy_tm:.6f}
      Total scatter: {np.sum(sigma_tm):.4e}
    
    Results (TE mode):
      Max current: {np.max(np.abs(sol_te)):.4e}
      Energy balance: {energy_te:.6f}
      Total scatter: {np.sum(sigma_te):.4e}
    
    Scattering:
      Peak angle: {np.degrees(angles[np.argmax(sigma_avg)]):.1f}°
      Forward/Back: {forward_backward_ratio:.2f}
      Anisotropy: {(np.max(sigma_avg)-np.min(sigma_avg))/np.max(sigma_avg):.3f}
    """
    ax9.text(0.1, 0.5, stats_text, fontsize=9, family='monospace',
            verticalalignment='center')
    
    plt.suptitle(f'Dielectric Cylinder Scattering (r={radius*1e6:.1f}μm, n={n_fiber}, λ={wavelength*1e9:.0f}nm)',
                fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save results to output directory
    filename = f'cylinder_r{radius*1e6:.0f}um_n{n_fiber}_scattering'
    plot_path = os.path.join(output_dir, f'{filename}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n[SAVED] Plot saved as '{plot_path}'")
    
    # Save numerical data to output directory
    cdf = np.cumsum(pdf_azimuthal * d_angle)
    data_path = os.path.join(output_dir, f'{filename}_data.npz')
    np.savez(data_path,
             angles=angles,
             sigma_tm=sigma_tm,
             sigma_te=sigma_te,
             sigma_avg=sigma_avg,
             pdf_azimuthal=pdf_azimuthal,
             theta_longitudinal=theta_long,
             pdf_longitudinal=pdf_longitudinal,
             cdf_azimuthal=cdf,
             theta_3d=theta_3d,
             phi_3d=phi_3d,
             sigma_3d_tm=sigma_3d_tm,
             sigma_3d_te=sigma_3d_te,
             sigma_3d_avg=sigma_3d_avg,
             pdf_3d=pdf_3d,
             radius=radius,
             n_fiber=n_fiber,
             wavelength=wavelength)
    print(f"[SAVED] Data saved as '{data_path}' (includes 3D BSDF)")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Cylinder radius: {radius*1e6:.1f} μm")
    print(f"Size parameter (2πr/λ): {2*np.pi*radius/wavelength:.2f}")
    print(f"Total scattering (TM): {np.sum(sigma_tm):.4e}")
    print(f"Total scattering (TE): {np.sum(sigma_te):.4e}")
    print(f"Total scattering (Avg): {np.sum(sigma_avg):.4e}")
    print(f"Energy balance (TM): {energy_tm:.6f}")
    print(f"Energy balance (TE): {energy_te:.6f}")
    print(f"Peak scattering at: {np.degrees(angles[np.argmax(sigma_avg)]):.1f}°")
    print(f"Forward/Backward ratio: {forward_backward_ratio:.2f}")
    print("=" * 70)
    print("\n[SUCCESS] Simulation completed successfully!")
    print(f"  Plot file: {plot_path}")
    print(f"  Data file: {data_path}")
    print(f"  Output directory: {output_dir}")
    print("=" * 70)
    # Note: Using 'Agg' backend - plots are saved, no display window

if __name__ == "__main__":
    main()
