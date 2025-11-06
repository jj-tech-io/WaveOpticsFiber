# scattering_plotter.py
"""
Shared plotting utilities for analytic and BEM scattering scripts.
Contains the unified 3D spherical plot and common helpers.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
# ---------- Plotting Helpers (Now Shared) ----------

BASE_SIZE = 5.0
def sanitize_filename(s):
    """Sanitize a string to be used as a filename."""
    import re
    s = re.sub(r'[<>:"/\\|?*]', '_', s)
    s = re.sub(r'\s+', '_', s)
    return s


def compute_triad_vectors(theta_deg, phi_deg, pol_angle_deg=0.0):
    """
    Computes and returns the mutually orthogonal unit vectors (k̂, Ê, Ĥ)
    for a Transverse Electro-Magnetic (TEM) plane wave.
    
    ANGLE CONVENTION (elevation-based):
    - θ: elevation angle from xy-plane (0° = xy-plane, 90° = +z axis)
    - φ: azimuthal angle in xy-plane (0° = +x axis, 90° = +y axis)
    - α (pol_angle): polarization rotation angle
    
    The wave vector is:
        k̂ = (cos θ cos φ, cos θ sin φ, sin θ)
    
    Parameters
    ----------
    theta_deg : float
        Elevation angle from xy-plane in degrees [0, 90]
    phi_deg : float
        Azimuthal angle in xy-plane in degrees [0, 360]
    pol_angle_deg : float
        Polarization angle in degrees (rotation about k̂)
    
    Returns
    -------
    k_hat : ndarray
        Wave propagation direction unit vector
    E_hat : ndarray
        Electric field polarization unit vector
    H_hat : ndarray
        Magnetic field direction unit vector (H = k × E for TEM wave)
    """
    θ, ϕ, α = np.deg2rad([theta_deg, phi_deg, pol_angle_deg])

    # UPDATED: Elevation angle convention from image
    # k̂ = (cos θ cos φ, cos θ sin φ, sin θ)
    k_hat = np.array([
        np.cos(θ) * np.cos(ϕ),
        np.cos(θ) * np.sin(ϕ),
        np.sin(θ)
    ], dtype=float)
    k_hat /= np.linalg.norm(k_hat)

    # Find two orthogonal vectors perpendicular to k̂
    x_hat = np.array([1.0, 0.0, 0.0])
    y_hat = np.array([0.0, 1.0, 0.0])
    
    # Gram-Schmidt: project out k̂ component
    p1 = x_hat - np.dot(x_hat, k_hat) * k_hat
    if np.linalg.norm(p1) < 1e-8:
        p1 = y_hat - np.dot(y_hat, k_hat) * k_hat
    p1 /= np.linalg.norm(p1)

    # Second perpendicular vector
    p2 = np.cross(k_hat, p1)
    p2 /= np.linalg.norm(p2)

    # Rotate by polarization angle α
    E_hat = np.cos(α) * p1 + np.sin(α) * p2
    
    # H perpendicular to both k and E (right-hand rule: H = k × E)
    H_hat = np.cross(k_hat, E_hat)
    H_hat /= np.linalg.norm(H_hat)

    return k_hat, E_hat, H_hat


def plot_em_wave_triad(
        k_hat, E_hat, H_hat,
        wavelength=1.0, length=1.0, n_points=300,
        amp_ratio=0.15, phase_offset=-360,
        view_elev=20, view_azim=45,
        show_orthogonality=True,
        shape=None,               # 'cylinder' or 'sphere'
        pol_angle_deg=0.0,
        theta_deg=0.0,
        phi_deg=0.0,
        results_dir="results/ortho_triads",
        wavelength_nm=None,
        radius_um=None
    ):
    """
    Draw a 3-D TEM wave and optionally a shape (cylinder or sphere) at the origin,
    with multiple transparent wavefronts to show incidence on the shape,
    displayed in three side-by-side views.
    
    ANGLE CONVENTION: Uses elevation angle θ from xy-plane (see compute_triad_vectors)
    
    Parameters
    ----------
    k_hat, E_hat, H_hat : ndarray
        Orthogonal unit vectors defining the wave triad
    wavelength : float
        Wavelength of the EM wave (dimensionless units for plotting)
    length : float
        Spatial extent of the wave visualization
    n_points : int
        Number of points along the wave
    amp_ratio : float
        Amplitude ratio for E and H field oscillations
    phase_offset : float
        Phase offset in degrees
    view_elev, view_azim : float
        View angles for the 3D plot (first subplot)
    show_orthogonality : bool
        Whether to show wavefront planes
    shape : str or None
        'cylinder', 'sphere', or None
    pol_angle_deg, theta_deg, phi_deg : float
        Angles for labeling and filename generation
    results_dir : str
        Directory to save the figure
    wavelength_nm : float, optional
        Wavelength in nanometers (for filename)
    radius_um : float, optional
        Radius in micrometers (for filename)
    """
    # Normalize direction vectors
    k = k_hat / np.linalg.norm(k_hat)
    e = E_hat / np.linalg.norm(E_hat)
    h = H_hat / np.linalg.norm(H_hat)

    # Compute wave data
    s = np.linspace(0, length, n_points)
    phase_offset_rad = np.deg2rad(phase_offset)
    phase = 2 * np.pi * s / wavelength + phase_offset_rad
    E_amp = amp_ratio * wavelength * np.sin(phase)
    H_amp = amp_ratio * wavelength * np.sin(phase) / 3
    origins = np.outer(s, k)                      # (n_points, 3)
    tips_E = origins + E_amp[:, None] * e         # (n_points, 3)
    tips_H = origins + H_amp[:, None] * h         # (n_points, 3)

    # Collect all coordinates for axis-limit calculation
    all_x = np.hstack([tips_E[:,0], tips_H[:,0], origins[:,0]])
    all_y = np.hstack([tips_E[:,1], tips_H[:,1], origins[:,1]])
    all_z = np.hstack([tips_E[:,2], tips_H[:,2], origins[:,2]])

    # Setup figure with three 3D subplots
    fig = plt.figure(figsize=compute_figsize(1, 3, base_size=BASE_SIZE))
    views = [
        (30, 45),   # wave-current perspective
        (100, 0),   # top-down
        (10, 90)    # front-on
    ]
    axes = [fig.add_subplot(1, 3, i + 1, projection='3d') for i in range(3)]

    for ax, (elev, azim) in zip(axes, views):
        ax.view_init(elev=elev, azim=azim)

        # ---------------------------------------
        # 1) Draw wavefront planes (if requested)
        # ---------------------------------------
        if show_orthogonality:
            u = np.linspace(-length * 0.5, length * 0.5, 15)
            U, V = np.meshgrid(u, u)
            Xp = U * e[0] + V * h[0]
            Yp = U * e[1] + V * h[1]
            Zp = U * e[2] + V * h[2]
            fronts = np.linspace(-0.5 * length, 0.5 * length, 3)
            for s0 in fronts:
                Xw = Xp + k[0] * s0
                Yw = Yp + k[1] * s0
                Zw = Zp + k[2] * s0
                ax.plot_surface(
                    Xw, Yw, Zw,
                    color="#14C8F9", alpha=0.125, linewidth=0,
                    zorder=0  # lowest zorder so everything else goes on top
                )
                all_x = np.hstack([all_x, Xw.ravel()])
                all_y = np.hstack([all_y, Yw.ravel()])
                all_z = np.hstack([all_z, Zw.ravel()])

        # ---------------------------------------
        # 2) Draw shape at origin (sphere or cylinder)
        # ---------------------------------------
        if shape == 'cylinder':
            z = np.linspace(-1, 1, 50)
            theta = np.linspace(0, 2*np.pi, 50)
            Theta, Zc = np.meshgrid(theta, z)
            R = 0.1
            Xc = R * np.cos(Theta)
            Yc = R * np.sin(Theta)
            ax.plot_surface(
                Xc, Yc, Zc,
                color="#F2F21B", alpha=0.5, linewidth=0, zorder=1
            )
            all_x = np.hstack([all_x, Xc.ravel()])
            all_y = np.hstack([all_y, Yc.ravel()])
            all_z = np.hstack([all_z, Zc.ravel()])

        elif shape == 'sphere':
            u2 = np.linspace(0, 2 * np.pi, 50)
            v2 = np.linspace(0, np.pi, 50)
            U2, V2 = np.meshgrid(u2, v2)
            R = 0.1
            Xs = R * np.cos(U2) * np.sin(V2)
            Ys = R * np.sin(U2) * np.sin(V2)
            Zs = R * np.cos(V2)
            ax.plot_surface(
                Xs, Ys, Zs,
                color="#F2F21B", alpha=0.5, linewidth=0, zorder=1
            )
            all_x = np.hstack([all_x, Xs.ravel()])
            all_y = np.hstack([all_y, Ys.ravel()])
            all_z = np.hstack([all_z, Zs.ravel()])

        # ---------------------------------------
        # 3) Draw E and H curves (moderate zorder)
        # ---------------------------------------
        ax.plot(
            *tips_E.T, 'r-', lw=2,
            zorder=5
        )
        ax.plot(
            *tips_H.T, 'b-', lw=2,
            zorder=5
        )

        # ---------------------------------------
        # 4) Draw quiver arrows (slightly above curves)
        # ---------------------------------------
        stride = max(1, n_points // 15)
        mask = (np.abs(E_amp) + np.abs(H_amp)) > amp_ratio * wavelength * 0.1
        for j in range(0, n_points, stride):
            if not mask[j]:
                continue
            ax.quiver(
                *(origins[j]),
                *(tips_E[j] - origins[j]),
                color='r', lw=1.5,
                arrow_length_ratio=0.1,
                zorder=6
            )
            ax.quiver(
                *(origins[j]),
                *(tips_H[j] - origins[j]),
                color='b', lw=1.5,
                arrow_length_ratio=0.1,
                zorder=6
            )

        # ---------------------------------------
        # 5) Draw triad unit vectors at origin (slightly above arrows)
        # ---------------------------------------
        ax.quiver(
            0, 0, 0,
            *(k * length),
            color='k', lw=2.5,
            arrow_length_ratio=0.08,
            zorder=7
        )
        ax.plot(
            [0, e[0] * amp_ratio * wavelength],
            [0, e[1] * amp_ratio * wavelength],
            [0, e[2] * amp_ratio * wavelength],
            'r--', lw=2, alpha=0.7,
            zorder=7
        )
        ax.plot(
            [0, h[0] * amp_ratio * wavelength],
            [0, h[1] * amp_ratio * wavelength],
            [0, h[2] * amp_ratio * wavelength],
            'b--', lw=2, alpha=0.7,
            zorder=7
        )

        # ---------------------------------------
        # 6) Draw text labels last (highest zorder)
        # ---------------------------------------
        text_z = 8
        ax.text(
            *(k * length * 1.05),
            'k̂', color='k', fontsize=12, fontweight='bold',
            zorder=text_z
        )
        ax.text(
            *(e * amp_ratio * wavelength * 1.05),
            'Ê', color='r', fontsize=12, fontweight='bold',
            zorder=text_z
        )
        ax.text(
            *(h * amp_ratio * wavelength * 1.05),
            'Ĥ', color='b', fontsize=12, fontweight='bold',
            zorder=text_z
        )

        # If showing orthogonality, add θ, φ, α arcs with labels
        if show_orthogonality:
            # θ arc (elevation from xy-plane)
            θ = np.deg2rad(theta_deg)
            
            # Project k onto xy-plane
            k_xy_proj = np.array([k[0], k[1], 0.0])
            k_xy_norm = np.linalg.norm(k_xy_proj)
            
            if k_xy_norm > 1e-8:
                k_xy_hat = k_xy_proj / k_xy_norm
                # Arc from xy-plane to k direction
                t = np.linspace(0, θ, 100)
                arcθ = np.cos(t)[:, None] * k_xy_hat + np.sin(t)[:, None] * np.array([0, 0, 1.0])
                arcθ *= (wavelength * 0.3)
                ax.plot(
                    arcθ[:, 0], arcθ[:, 1], arcθ[:, 2],
                    color='orange', linestyle='--', lw=1.5,
                    zorder=6
                )
                mid = len(arcθ) // 2
                ax.text(
                    *(arcθ[mid] * 1.1),
                    'θ', color='orange', fontsize=10, fontweight='bold',
                    zorder=text_z
                )

            # φ arc (azimuthal in xy-plane)
            ϕ = np.deg2rad(phi_deg)
            if phi_deg > 0.5:
                r = wavelength * 0.3
                t2 = np.linspace(0, ϕ, 100)
                arcφ = np.vstack((
                    np.cos(t2) * r,
                    np.sin(t2) * r,
                    np.zeros_like(t2)
                )).T
                ax.plot(
                    arcφ[:, 0], arcφ[:, 1], arcφ[:, 2],
                    color='yellow', linestyle='--', lw=1.5,
                    zorder=6
                )
                mid = len(arcφ) // 2
                ax.text(
                    *(arcφ[mid] * 1.2),
                    'φ', color='yellow', fontsize=10, fontweight='bold',
                    zorder=text_z
                )

            # α arc (polarization rotation about k)
            α = np.deg2rad(pol_angle_deg)
            if pol_angle_deg > 0.5:
                base = np.array([1.0, 0.0, 0.0])
                tmp = base - (base.dot(k)) * k
                if np.linalg.norm(tmp) < 1e-8:
                    base = np.array([0.0, 1.0, 0.0])
                    tmp = base - (base.dot(k)) * k
                p1 = tmp / np.linalg.norm(tmp)
                p2 = np.cross(k, p1)
                p2 /= np.linalg.norm(p2)
                t3 = np.linspace(0, α, 100)
                arcα = (np.outer(np.cos(t3), p1) + np.outer(np.sin(t3), p2)) * (wavelength * 0.3)
                arcα += k * (wavelength * 0.15)
                ax.plot(
                    arcα[:, 0], arcα[:, 1], arcα[:, 2],
                    color='purple', linestyle='--', lw=1.5,
                    zorder=6
                )
                mid = len(arcα) // 2
                ax.text(
                    *(arcα[mid] * 1.1),
                    'α', color='purple', fontsize=10, fontweight='bold',
                    zorder=text_z
                )

        # ---------------------------------------
        # 7) Enforce 1:1:1 aspect ratio and symmetric limits
        # ---------------------------------------
        ax.set_box_aspect([1, 1, 1])
        lim = max(abs(all_x).max(), abs(all_y).max(), abs(all_z).max()) * 1.05
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)

        # ---------------------------------------
        # 8) Axes formatting
        # ---------------------------------------
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.grid(True)
        ax.set_xticks([-lim, 0, lim])
        ax.set_yticks([-lim, 0, lim])
        ax.set_zticks([-lim, 0, lim])
        ax.set_xticklabels(['-', '0', '+'], fontsize=10)
        ax.set_yticklabels(['-', '0', '+'], fontsize=10)
        ax.set_zticklabels(['-', '0', '+'], fontsize=10)
        ax.tick_params(length=5)

        if ax is axes[-1]:
            handles = [
                Line2D([0], [0], color='r', lw=2, label='E'),
                Line2D([0], [0], color='b', lw=2, label='H'),
                Line2D([0], [0], color='k', lw=2, label='k')
            ]
            if show_orthogonality:
                handles.append(Patch(facecolor='#14C8F9', alpha=0.3, label='Incident Plane'))
            if shape:
                handles.append(Patch(facecolor='#F2F21B', alpha=0.5, label=shape.capitalize()))

            ax.legend(
                handles=handles,
                loc='center left',
                bbox_to_anchor=(1.02, 0.5),
                bbox_transform=ax.transAxes,
                frameon=False,
                fontsize=12
            )

    # ---------------------------------------
    # Save full "wave + triad" figure with comprehensive filename
    # ---------------------------------------
    os.makedirs(results_dir, exist_ok=True)
    
    # Build filename with all parameters
    fname_parts = ["triad_wave"]
    if radius_um is not None:
        fname_parts.append(f"r{radius_um:.2f}um")
    if wavelength_nm is not None:
        fname_parts.append(f"wl{wavelength_nm:.0f}nm")
    fname_parts.append(f"thi{theta_deg:.1f}")
    fname_parts.append(f"phi{phi_deg:.1f}")
    fname_parts.append(f"pol{pol_angle_deg:.1f}")
    if shape:
        fname_parts.append(shape)
    
    filename = "_".join(fname_parts) + ".png"
    
    fig.savefig(os.path.join(results_dir, filename), dpi=400, bbox_inches='tight')
    print(f"Saved: {os.path.join(results_dir, filename)}")
    plt.close(fig)


def plot_incident_triad(
    theta_deg, phi_deg, pol_angle_deg=0.0,
    scale=1.0,
    title: str = None,
    results_dir="results",
    view_angles=None, arc_resolution=100,
    wavelength_nm=None,
    radius_um=None
):
    """
    Plot the E–H–k triad for an incident plane wave from one or more camera views,
    then call `plot_em_wave_triad` to overlay the full wave on a sphere.
    
    ANGLE CONVENTION: Uses elevation angle θ from xy-plane (see compute_triad_vectors)
    k̂ = (cos θ cos φ, cos θ sin φ, sin θ)

    Parameters
    ----------
    theta_deg : float
        Elevation angle from xy-plane [0°=horizontal, 90°=vertical]
    phi_deg : float
        Azimuthal angle in xy-plane [0°=+x, 90°=+y]
    pol_angle_deg : float
        Polarization rotation angle about k̂
    scale : float
        Scale factor for vector lengths
    title : str, optional
        Custom title for the figure and filename base
    results_dir : str
        Output directory
    view_angles : list of tuples, optional
        List of (elevation, azimuth) view angles
    arc_resolution : int
        Number of points in angle arcs
    wavelength_nm : float, optional
        Wavelength in nanometers (for filename)
    radius_um : float, optional
        Radius in micrometers (for filename)
    """
    os.makedirs(results_dir, exist_ok=True)

    # Compute k̂, Ê, Ĥ using elevation angle convention
    k_hat, E_hat, H_hat = compute_triad_vectors(theta_deg, phi_deg, pol_angle_deg)

    if view_angles is None:
        view_angles = [(45, 45)]

    n_views = len(view_angles)
    fig = plt.figure(figsize=compute_figsize(1, n_views, base_size=BASE_SIZE + 2))
    axes = []
    for i, (elev, azim) in enumerate(view_angles, start=1):
        ax = fig.add_subplot(1, n_views, i, projection='3d')
        ax.view_init(elev=elev, azim=azim)

        # Draw triad at origin
        O = np.zeros(3)
        ax.quiver(*O, *(k_hat * scale), color='k', lw=2.5, arrow_length_ratio=0.08, zorder=5)
        ax.quiver(*O, *(E_hat * scale), color='r', lw=1.5, arrow_length_ratio=0.1, zorder=5)
        ax.quiver(*O, *(H_hat * scale), color='b', lw=1.5, arrow_length_ratio=0.1, zorder=5)

        # Label the tips
        ax.text(*(k_hat * scale * 1.05), 'k̂', color='k', fontsize=12, fontweight='bold', zorder=10)
        ax.text(*(E_hat * scale * 1.05), 'Ê', color='r', fontsize=12, fontweight='bold', zorder=10)
        ax.text(*(H_hat * scale * 1.05), 'Ĥ', color='b', fontsize=12, fontweight='bold', zorder=10)

        # Dashed reference lines for E and H directions
        ax.plot(
            [0, E_hat[0] * scale],
            [0, E_hat[1] * scale],
            [0, E_hat[2] * scale],
            'r--', lw=1, alpha=0.7, zorder=4
        )
        ax.plot(
            [0, H_hat[0] * scale],
            [0, H_hat[1] * scale],
            [0, H_hat[2] * scale],
            'b--', lw=1, alpha=0.7, zorder=4
        )

        # θ arc (elevation from xy-plane)
        θ = np.deg2rad(theta_deg)
        k_xy_proj = np.array([k_hat[0], k_hat[1], 0.0])
        k_xy_norm = np.linalg.norm(k_xy_proj)
        
        if k_xy_norm > 1e-8:
            k_xy_hat = k_xy_proj / k_xy_norm
            t = np.linspace(0, θ, arc_resolution)
            arcθ = np.cos(t)[:, None] * k_xy_hat + np.sin(t)[:, None] * np.array([0, 0, 1.0])
            arcθ *= scale * 0.3
            ax.plot(arcθ[:, 0], arcθ[:, 1], arcθ[:, 2], color='orange', linestyle='--', lw=1.5, zorder=6)
            mid = len(arcθ) // 2
            ax.text(*(arcθ[mid] * 1.1), 'θ', color='orange', fontsize=10, fontweight='bold', zorder=10)

        # φ arc (azimuthal in xy-plane)
        ϕ = np.deg2rad(phi_deg)
        if phi_deg > 0.5:
            r = scale * 0.3
            t2 = np.linspace(0, ϕ, arc_resolution)
            arcφ = np.vstack((np.cos(t2) * r, np.sin(t2) * r, np.zeros_like(t2))).T
            ax.plot(arcφ[:, 0], arcφ[:, 1], arcφ[:, 2], color='yellow', linestyle='--', lw=1.5, zorder=6)
            mid = len(arcφ) // 2
            ax.text(*(arcφ[mid] * 1.2), 'φ', color='yellow', fontsize=10, fontweight='bold', zorder=10)

        # α arc (polarization rotation about k)
        α = np.deg2rad(pol_angle_deg)
        if pol_angle_deg > 0.5:
            base = np.array([1.0, 0.0, 0.0])
            tmp = base - (base.dot(k_hat)) * k_hat
            if np.linalg.norm(tmp) < 1e-8:
                base = np.array([0.0, 1.0, 0.0])
                tmp = base - (base.dot(k_hat)) * k_hat
            p1 = tmp / np.linalg.norm(tmp)
            p2 = np.cross(k_hat, p1)
            p2 /= np.linalg.norm(p2)
            t3 = np.linspace(0, α, arc_resolution)
            arcα = (np.outer(np.cos(t3), p1) + np.outer(np.sin(t3), p2)) * (scale * 0.3)
            arcα += k_hat * (scale * 0.15)
            ax.plot(arcα[:, 0], arcα[:, 1], arcα[:, 2], color='purple', linestyle='--', lw=1.5, zorder=6)
            mid = len(arcα) // 2
            ax.text(*(arcα[mid] * 1.1), 'α', color='purple', fontsize=10, fontweight='bold', zorder=10)

        # Enforce 1:1:1 aspect
        ax.set_box_aspect([1, 1, 1])

        # Centered limits
        lim = scale * 1.3
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)

        # Tick labels
        ax.set_xticks([-lim, 0, lim])
        ax.set_yticks([-lim, 0, lim])
        ax.set_zticks([-lim, 0, lim])
        ax.set_xticklabels(['-', '0', '+'], fontsize=10)
        ax.set_yticklabels(['-', '0', '+'], fontsize=10)
        ax.set_zticklabels(['-', '0', '+'], fontsize=10)
        ax.grid(True)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)

        axes.append(ax)

    # If a title was provided, use it as suptitle
    if title is not None:
        fig.suptitle(title, fontsize=14)

    # # Save the "triad-only" figure with comprehensive filename
    fname_parts = ["triad_only"]
    if title:
        fname_parts.append(sanitize_filename(title))
    if radius_um is not None:
        fname_parts.append(f"r{radius_um:.2f}um")
    if wavelength_nm is not None:
        fname_parts.append(f"wl{wavelength_nm:.0f}nm")
    fname_parts.append(f"thi{theta_deg:.1f}")
    fname_parts.append(f"phi{phi_deg:.1f}")
    fname_parts.append(f"pol{pol_angle_deg:.1f}")
    
    out_fname = "_".join(fname_parts) + ".png"

    # fig.savefig(os.path.join(results_dir, out_fname), dpi=400, bbox_inches='tight')
    print(f"Saved: {os.path.join(results_dir, out_fname)}")
    plt.close(fig)

    # Now call the "full EM wave" version, using the first view from view_angles
    elev0, azim0 = view_angles[0]
    plot_em_wave_triad(
        k_hat, E_hat, H_hat,
        wavelength=1.0,
        length=scale,
        n_points=300,
        amp_ratio=0.15,
        phase_offset=-360,
        view_elev=elev0,
        view_azim=azim0,
        show_orthogonality=True,
        shape="sphere",
        pol_angle_deg=pol_angle_deg,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
        results_dir=results_dir,
        wavelength_nm=wavelength_nm,
        radius_um=radius_um
    )
def _mkdir(p): os.makedirs(p, exist_ok=True)
def _deg(x): return np.rad2deg(x)
def _wrap_2pi(x): return np.mod(x, 2*np.pi)
def _log_norm(I, floor=1e-12):
    I = np.asarray(I, float).copy()
    I[I < floor] = floor
    L = np.log10(I)
    return L - L.max()

def _ensure_even(n): return n if n % 2 == 0 else n + 1

def _head(params):
    """Generate header string for plot titles from params dict."""
    return f"(m_ext={params['m_ext']}, φ₀={params.get('incidence_azimuth_deg',0.0):.1f}°, ψ={params.get('pol_angle_deg',0.0):.1f}°, mode={params.get('longitudinal_mode','exact')})"

def _rephase_to_reference(theta, zero_deg=0.0):
    """Return angle array expressed in a display frame with 0° at 'zero_deg' (lab)."""
    zero = np.deg2rad(zero_deg)
    th = _wrap_2pi(np.asarray(theta) - zero)
    # sort so the plotted line starts from 0 and increases to 2π
    order = np.argsort(th)
    return th[order], order
def _phi0_lab_deg(params):
    """
    DEPRECATED/Unreliable: This function has conflicting logic.
    Use direct param checks instead.
    """
    if 'incidence_azimuth_deg' in params:
        return params['incidence_azimuth_deg']
    elif 'incidence_azimuth_deg_lab' in params:
        phi0_lab = params['incidence_azimuth_deg_lab']
        phi0_model = (90.0 - phi0_lab) % 360.0 # This is solver-specific logic
        return phi0_model
    else:
        return 0.0

def compute_figsize(rows=1, cols=1, base_size=BASE_SIZE):
    return (cols * base_size * 1.2, rows * base_size)

def set_equal_3d_axes(ax, X, Y, Z, zoom=1.0):
    mr = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / (2.0 * zoom)
    mx, my, mz = (X.max()+X.min())*0.5, (Y.max()+Y.min())*0.5, (Z.max()+Z.min())*0.5
    ax.set_xlim(mx - mr, mx + mr); ax.set_ylim(my - mr, my + mr); ax.set_zlim(mz - mr, mz + mr)

def _log_norm_intensity(intensity, floor=1e-12):
    I = np.array(intensity, dtype=float)
    I[I < floor] = floor
    logI = np.log10(I)
    return logI - logI.max()

def _polar_east_ccw(ax):
    """Ensure theta=0° at East, angles increase CCW."""
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

# ---------- Internal Filename Helper ----------

def _build_filename(params, plot_type_prefix):
    """
    Builds a standardized filename from the params dict.
    Format: [prefix]_r{R}um_wl{L}nm_th90_phi{P}_psi{S}_{method}.png
    """
    
    # 1. Radius (um)
    try:
        # From analytic_comparison.py 'layers'
        radius_m = params['layers'][-1]['R']
    except (KeyError, IndexError, TypeError):
        # Fallback for bem.py 'R_eff_long'
        radius_m = params.get('R_eff_long', 0.0) 
    r_str = f"r{radius_m * 1e6:.2f}um"

    # 2. Wavelength (nm)
    wl_m = params.get('lambda_0', 0.0)
    wl_str = f"wl{wl_m * 1e9:.0f}nm"

    # 3. Theta_i (implicit 90 for 2D scattering)
    th_str = "th90"

    # 4. Phi_i (lab frame)
    # Use 'incidence_azimuth_deg_lab' if present (analytic_comparison)
    # Fallback to 'incidence_azimuth_deg' (bem.py)
    phi_lab = params.get('incidence_azimuth_deg_lab', params.get('incidence_azimuth_deg', 0.0))
    phi_str = f"phi{phi_lab:.0f}"
    
    # 5. Polarization (psi)
    psi_deg = params.get('pol_angle_deg', 0.0)
    psi_str = f"psi{psi_deg:.0f}"

    # 6. Method (longitudinal mode)
    method = params.get('longitudinal_mode', 'exact')

    # Assemble
    tag = f"{r_str}_{wl_str}_{th_str}_{phi_str}_{psi_str}_{method}"
    return f"{plot_type_prefix}_{tag}.png"


# ---------- Unified 3D Spherical Plot ----------

def save_3d_surface(phi_rad_az, I_az, th_rad_long, I_long, params, title_suffix, tag=None, view_elev=30, view_azim=-135):
    """
    Saves a 3D spherical plot of the full scattering PDF using a separable model.
    
    Parameters
    ----------
    phi_rad_az : ndarray
        Azimuthal angles in radians (0 to 2π)
    I_az : ndarray
        Azimuthal intensity distribution
    th_rad_long : ndarray
        Longitudinal/polar angles in radians (0 to π)
    I_long : ndarray
        Longitudinal intensity distribution
    params : dict
        Parameter dictionary containing 'results_dir', 'm_ext', 'pol_angle_deg', 'longitudinal_mode', etc.
    title_suffix : str
        Additional text to append to the plot title
    tag : str, optional
        If provided, uses simple filename format; otherwise uses _build_filename
    view_elev : float, optional
        Elevation angle for 3D view in degrees (default: 30)
    view_azim : float, optional
        Azimuthal angle for 3D view in degrees (default: -135)
    """
    results_dir = params['results_dir']; os.makedirs(results_dir, exist_ok=True)
    mode = params.get('longitudinal_mode', 'exact')
    
    # Use 'incidence_azimuth_deg_lab' if present, else 'incidence_azimuth_deg'
    phi0_lab = params.get('incidence_azimuth_deg_lab', params.get('incidence_azimuth_deg', 0.0))
    
    psi = params.get('pol_angle_deg', 0.0)

    # --- Data Preparation ---
    I_az_lin_norm = I_az / (I_az.max() + 1e-12)
    I_long_lin_norm = I_long / (I_long.max() + 1e-12)

    R = I_long_lin_norm[:, None] * I_az_lin_norm[None, :]
    Theta, Phi = np.meshgrid(th_rad_long, phi_rad_az, indexing='ij')

    X = R * np.sin(Theta) * np.cos(Phi)
    Y = R * np.sin(Theta) * np.sin(Phi)
    Z = R * np.cos(Theta)

    # --- Plot Setup ---
    fig = plt.figure(figsize=compute_figsize(1, 1.2))
    ax = fig.add_subplot(111, projection='3d')
    
    plot_stride = 4
    norm = plt.Normalize(R.min(), R.max())

    # Use cmap directly instead of facecolors for compatibility with stride
    surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.cividis, norm=norm,
                           rstride=plot_stride, cstride=plot_stride,
                           linewidth=0.0, antialiased=True, shade=True)

    fig.colorbar(surf, ax=ax, shrink=0.7, aspect=10, pad=0.1, label="Normalized Intensity")

    set_equal_3d_axes(ax, X, Y, Z, zoom=1.2)

    ax.set_title(f"3D Spherical PDF (Separable Model)\n(m_ext={params['m_ext']}, φ₀(lab)={phi0_lab:.1f}°, ψ={psi:.1f}°, mode={mode})\n{title_suffix}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=view_elev, azim=view_azim)

    # --- Save Figure ---
    fig.tight_layout()
    
    # --- FILENAME LOGIC UPDATED ---
    if tag:
        # Used by compare_bem_vs_analytic.py
        fname = f"surface3d_SPHERICAL_{tag}.png"
    else:
        # Used by analytic_comparison.py (and bem.py if run standalone)
        fname = _build_filename(params, "surface3d_SPHERICAL")
    
    fig.savefig(os.path.join(results_dir, fname), dpi=200)
    print("Saved:", os.path.join(results_dir, fname))
    plt.close(fig)

# ---------- BEM-specific plotting functions ----------

def save_paper_style_all(theta_az, I_az, th_long, I_long, params, title_suffix):
    """Save two-panel plot: longitudinal (cartesian) + azimuthal (polar) for BEM."""
    os.makedirs(params['results_dir'], exist_ok=True)
    head = _head(params)
    fig = plt.figure(figsize=compute_figsize(1, 2))

    # Left: longitudinal (cartesian)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(np.rad2deg(th_long), _log_norm_intensity(I_long))
    ax1.set_xlabel("Polar angle θ (deg)")
    ax1.set_ylabel("Log(I / I_max)")
    ax1.set_title(f"Longitudinal {head}\n{title_suffix}")
    ax1.set_xlim(0, 180)
    ax1.grid(True, ls='--', alpha=0.6)

    # Right: azimuthal (polar) — EAST & CCW
    ax2 = fig.add_subplot(1, 2, 2, projection='polar')
    _polar_east_ccw(ax2)
    r = np.clip(_log_norm_intensity(I_az), -4.0, 0.0) + 4.0
    ax2.plot(theta_az, r)
    ax2.set_title(f"Azimuthal (0–360°), 0° at East\n{title_suffix}")

    fig.tight_layout()
    fn = f"paper_style_bem_global_{params['m_ext']:.3f}_{params.get('longitudinal_mode','exact')}.png"
    fig.savefig(os.path.join(params['results_dir'], fn), dpi=200)
    plt.close(fig)

def save_azimuthal_both(theta, I, params, title_suffix):
    """Save both polar and cartesian azimuthal plots for BEM."""
    os.makedirs(params['results_dir'], exist_ok=True)
    head = _head(params)
    mode = params.get('longitudinal_mode', 'exact')

    # Polar — EAST & CCW
    fig = plt.figure(figsize=compute_figsize(1, 1.2))
    ax = fig.add_subplot(111, projection='polar')
    _polar_east_ccw(ax)
    r = np.clip(_log_norm_intensity(I), -4.0, 0.0) + 4.0
    ax.plot(theta, r)
    ax.set_title(f"Azimuthal (0–360°), 0° at East\n{title_suffix}")
    fn = f"azimuthal_polar_bem_global_{params['m_ext']:.3f}_{mode}.png"
    fig.savefig(os.path.join(params['results_dir'], fn), dpi=200)
    plt.close(fig)

    # Cartesian
    fig = plt.figure(figsize=compute_figsize(1, 1.2))
    ax = fig.add_subplot(111)
    ax.plot(np.rad2deg(theta), _log_norm_intensity(I))
    ax.set_xlabel("Azimuth angle φ (deg)")
    ax.set_ylabel("Log(I / I_max)")
    ax.set_title(f"Azimuthal (Cartesian) {head}\n{title_suffix}")
    ax.set_xlim(0, 360)
    ax.grid(True, ls='--', alpha=0.6)
    fn = f"azimuthal_cart_bem_global_{params['m_ext']:.3f}_{mode}.png"
    fig.savefig(os.path.join(params['results_dir'], fn), dpi=200)
    plt.close(fig)

def save_longitudinal_both(th_long, I_long, params, title_suffix):
    """Save longitudinal plot (cartesian) for BEM."""
    os.makedirs(params['results_dir'], exist_ok=True)
    head = _head(params)
    mode = params.get('longitudinal_mode', 'exact')
    fig = plt.figure(figsize=compute_figsize(1, 1.2))
    ax = fig.add_subplot(111)
    ax.plot(np.rad2deg(th_long), _log_norm_intensity(I_long))
    ax.set_xlabel("Polar angle θ (deg)")
    ax.set_ylabel("Log(I / I_max)")
    ax.set_title(f"Longitudinal {head}\n{title_suffix}")
    ax.grid(True, ls='--', alpha=0.6)
    ax.set_xlim(0, 180)
    fn = f"longitudinal_cart_bem_global_{params['m_ext']:.3f}_{mode}.png"
    fig.savefig(os.path.join(params['results_dir'], fn), dpi=200)
    plt.close(fig)

# ---------- Other Plotting Functions ----------
    
def _polar_east_ccw(ax):
    """Ensure theta=0° at East, angles increase CCW."""
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)


def save_paper_style(theta_rad, intensity, th_long, I_long, params, title_suffix):
    results_dir = params['results_dir']; os.makedirs(results_dir, exist_ok=True)

    theta_deg_long = np.rad2deg(th_long)
    logI_long_norm = _log_norm_intensity(I_long)

    xlo, xhi = params.get('paper_style_xlim_deg', (0.0, 45.0))
    peak_deg = theta_deg_long[np.argmax(I_long)]
    if not (xlo <= peak_deg <= xhi):
        xlo, xhi = 0.0, 180.0

    phi0_lab = params.get('incidence_azimuth_deg_lab', params.get('incidence_azimuth_deg', 0.0))
    psi = params.get('pol_angle_deg', 0.0)
    head = f"(m_ext={params['m_ext']}, φ₀(lab)={phi0_lab:.1f}°, ψ={psi:.1f}°, mode={params.get('longitudinal_mode','exact')})"

    fig = plt.figure(figsize=compute_figsize(1, 2))

    # Cartesian longitudinal
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(theta_deg_long, logI_long_norm)
    ax1.set_xlabel("Polar angle θ (deg)")
    ax1.set_ylabel("Log(I / I_max)")
    ax1.set_title(f"Longitudinal\n{head}")
    ax1.set_xlim(xlo, xhi)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Polar azimuthal — EAST & CCW
    ax2 = fig.add_subplot(1, 2, 2, projection='polar')
    _polar_east_ccw(ax2)
    r = np.clip(_log_norm_intensity(intensity), -4.0, 0.0) + 4.0
    ax2.plot(theta_rad, r)
    ax2.set_title(f"Azimuthal (0–360°), 0° at East\n{title_suffix}")

    fig.tight_layout()
    # --- FILENAME UPDATED ---
    fname = _build_filename(params, "paper_style")
    fig.savefig(os.path.join(results_dir, fname), dpi=200)
    print("Saved:", os.path.join(results_dir, fname))
    plt.close(fig)

def save_azimuthal(theta_rad, intensity, params, title_suffix=""):
    results_dir = params['results_dir']; os.makedirs(results_dir, exist_ok=True)
    phi0_lab = params.get('incidence_azimuth_deg_lab', params.get('incidence_azimuth_deg', 0.0))
    psi = params.get('pol_angle_deg', 0.0); mode = params.get('longitudinal_mode','exact')

    r = np.clip(_log_norm_intensity(intensity), -4.0, 0.0) + 4.0
    fig = plt.figure(figsize=compute_figsize(1, 1.2))
    ax = fig.add_subplot(111, projection='polar')
    _polar_east_ccw(ax)
    ax.plot(theta_rad, r)
    ax.set_title(f"Azimuthal (0–360°), 0° at East\n(m_ext={params['m_ext']}, φ₀(lab)={phi0_lab:.1f}°, ψ={psi:.1f}°, mode={mode}) - {title_suffix}")
    fig.tight_layout()
    # --- FILENAME UPDATED ---
    fname = _build_filename(params, "azimuthal")
    fig.savefig(os.path.join(results_dir, fname), dpi=200)
    print("Saved:", os.path.join(results_dir, fname))
    plt.close(fig)

def save_longitudinal(th_long, I_long, params, title_suffix):
    results_dir = params['results_dir']; os.makedirs(results_dir, exist_ok=True)
    phi0_lab = params.get('incidence_azimuth_deg_lab', params.get('incidence_azimuth_deg', 0.0))
    psi = params.get('pol_angle_deg', 0.0); mode = params.get('longitudinal_mode', 'exact')

    fig = plt.figure(figsize=compute_figsize(1, 1.2))
    ax = fig.add_subplot(111)
    ax.plot(np.rad2deg(th_long), _log_norm_intensity(I_long))
    ax.set_xlabel("Polar angle θ (deg)"); ax.set_ylabel("Log(I / I_max)")
    ax.set_title(f"Longitudinal (0–180°)\n(m_ext={params['m_ext']}, φ₀(lab)={phi0_lab:.1f}°, ψ={psi:.1f}°, mode={mode}) - {title_suffix}")
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    # --- FILENAME UPDATED ---
    fname = _build_filename(params, "longitudinal")
    fig.savefig(os.path.join(results_dir, fname), dpi=200)
    print("Saved:", os.path.join(results_dir, fname))
    plt.close(fig)

# ---------- Overlay Plotting Functions ----------
# (These functions do not take the full `params` dict, 
# so they rely on the `tag` for filename info)

def plot_overlays(theta_az_an, I_az_an, theta_az_bem, I_az_bem,
                  long_dict, outdir, tag):
    _mkdir(outdir)

    # 1) Azimuthal overlay (Cartesian)
    plt.figure(figsize=(7.5, 4.2))
    plt.plot(_deg(theta_az_an),  _log_norm(I_az_an),  label="Analytic")
    plt.plot(_deg(theta_az_bem), _log_norm(I_az_bem), label="BEM-Global")
    plt.xlim(0, 360)
    plt.xlabel("Azimuth angle φ (deg) — lab frame (East=0°, CCW)")
    plt.ylabel("Log(I / I_max)")
    plt.grid(True, ls="--", alpha=0.6)
    plt.legend()
    f1 = os.path.join(outdir, f"overlay_azimuthal_{tag}.png")
    plt.tight_layout(); plt.savefig(f1, dpi=200); plt.close()

    # 2) Longitudinal overlay
    plt.figure(figsize=(7.5, 4.2))
    for name, (th, I) in long_dict.items():
        plt.plot(_deg(th), _log_norm(I), label=name)
    plt.xlim(0, 180)
    plt.xlabel("Polar angle θ (deg)")
    plt.ylabel("Log(I / I_max)")
    plt.grid(True, ls="--", alpha=0.6)
    plt.legend()
    f2 = os.path.join(outdir, f"overlay_longitudinal_{tag}.png")
    plt.tight_layout(); plt.savefig(f2, dpi=200); plt.close()

    # 3) Polar overlay (East=0°, CCW) — NO rotation of data here
    fig = plt.figure(figsize=(7.5, 4.2))
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    r_an  = np.clip(_log_norm(I_az_an),  -4.0, 0.0) + 4.0
    r_bem = np.clip(_log_norm(I_az_bem), -4.0, 0.0) + 4.0
    ax.plot(theta_az_an,  r_an,  label="Analytic")
    ax.plot(theta_az_bem, r_bem, label="BEM-Global")
    ax.set_title("Azimuthal (0–360°), 0° at East")
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    f3 = os.path.join(outdir, f"overlay_polar_{tag}.png")
    plt.tight_layout(); plt.savefig(f3, dpi=200); plt.close()

    return f1, f2, f3
def save_paper_style_overlay(theta_az_an, I_az_an, theta_az_bem, I_az_bem,
                             long_dict, outdir, tag, zero_ref_deg=0.0):
    """
    Save a two-panel 'paper style' overlay:
    ...[docstring unchanged]...
    """
    _mkdir(outdir)

    # Rephase both azimuthal angle arrays to share the same display zero
    th_an_disp, idx_an   = _rephase_to_reference(theta_az_an,  zero_ref_deg)
    th_bem_disp, idx_bem = _rephase_to_reference(theta_az_bem, zero_ref_deg)
    I_an_disp  = I_az_an[idx_an]
    I_bem_disp = I_az_bem[idx_bem]

    fig = plt.figure(figsize=(10.0, 4.2))

    # Left: longitudinal overlay
    ax1 = fig.add_subplot(1, 2, 1)
    for name, (th, I) in long_dict.items():
        ax1.plot(_deg(th), _log_norm(I), label=name)
    ax1.set_xlim(0, 180)
    ax1.set_xlabel("Polar angle θ (deg)")
    ax1.set_ylabel("Log(I / I_max)")
    ax1.grid(True, ls='--', alpha=0.6)
    ax1.legend()

    # Right: azimuthal polar overlay (East=0°, CCW) with chosen display zero
    ax2 = fig.add_subplot(1, 2, 2, projection='polar')
    ax2.set_theta_zero_location("E")
    ax2.set_theta_direction(1)
    r_an  = np.clip(_log_norm(I_an_disp),  -4.0, 0.0) + 4.0
    r_bem = np.clip(_log_norm(I_bem_disp), -4.0, 0.0) + 4.0
    ax2.plot(th_an_disp,  r_an,  label="Analytic")
    ax2.plot(th_bem_disp, r_bem, label="BEM-Global")
    ax2.set_title(f"Azimuthal (0–360°), 0° at {zero_ref_deg:.1f}° (lab)")
    ax2.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

    f = os.path.join(outdir, f"overlay_paperstyle_{tag}.png")
    plt.tight_layout(); plt.savefig(f, dpi=200); plt.close(fig)
    return f