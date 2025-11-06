"""
BEM Solver for electromagnetic scattering from infinite cylinders.
"""
import numpy as np
from scipy.special import hankel2
from scipy.linalg import lu_factor, lu_solve
from .background import Background
from .wave import Wave
from .element import Element
from .basis import Basis

class BEMSolver:
    """
    Boundary Element Method solver for 3D electromagnetic scattering.
    
    Solves scattering of plane waves from an infinite cylinder with
    arbitrary 2D cross-section using the Method of Moments (MoM).
    """
    
    def __init__(self, nodes, quadrature=2, phi_i=0.0, mode='TM', 
                 freq=5e14, mur=1.0, epsr=1.55**2, theta=np.pi/2):
        """
        Initialize the BEM solver.
        
        Parameters:
            nodes (list or np.ndarray): List of 3D node positions defining the boundary
            quadrature (int): Number of quadrature points per element (2, 3, or 4)
            phi_i (float): Azimuthal incident angle (radians)
            mode (str): Polarization mode ('TM' or 'TE')
            freq (float): Frequency (Hz)
            mur (float): Relative permeability of scatterer
            epsr (complex): Relative permittivity of scatterer (can be complex)
            theta (float): Longitudinal incident angle from z-axis (radians)
        """
        self.nodes = np.array(nodes)
        self.numel = len(nodes)
        self.quadrature = quadrature
        self.theta = theta
        
        # Initialize material properties
        self.background = Background(mur, epsr)
        self.wave = Wave(self.background, phi_i, mode, freq, theta)
        
        # Calculate radius (maximum distance from origin)
        self.radius = np.max(np.linalg.norm(self.nodes, axis=1))
        
        # Set up numerical integration
        self._setup_quadrature()
        
        # Create elements and basis functions
        self._create_elements()
        self._create_basis()
        
        # Initialize system matrices
        n = 4 * self.numel
        self.Z = np.zeros((n, n), dtype=complex)
        self.vvec = np.zeros(n, dtype=complex)
        self.sol = None
        
    def _setup_quadrature(self):
        """Set up Gaussian quadrature points and weights."""
        if self.quadrature == 2:
            self.xvec = [-np.sqrt(1/3), np.sqrt(1/3)]
            self.weights = [1.0, 1.0]
        elif self.quadrature == 3:
            self.xvec = [0.0, -np.sqrt(3/5), np.sqrt(3/5)]
            self.weights = [8/9, 5/9, 5/9]
        elif self.quadrature == 4:
            a = 2/7 * np.sqrt(6/5)
            self.xvec = [np.sqrt(3/7 - a), -np.sqrt(3/7 - a),
                        np.sqrt(3/7 + a), -np.sqrt(3/7 + a)]
            w1 = (18 + np.sqrt(30)) / 36
            w2 = (18 - np.sqrt(30)) / 36
            self.weights = [w1, w1, w2, w2]
        else:
            raise ValueError("Quadrature must be 2, 3, or 4")
        
        # Basis function shape functions (linear interpolation on [-1, 1])
        self.f = np.zeros((2, self.quadrature))
        for i, xi in enumerate(self.xvec):
            self.f[0, i] = (1 - xi) / 2  # Shape function for node 1
            self.f[1, i] = (1 + xi) / 2  # Shape function for node 2
        self.g = self.f.copy()
        
    def _create_elements(self):
        """Create boundary elements from nodes."""
        self.elements = []
        for i in range(self.numel):
            i_next = (i + 1) % self.numel
            i_prev = (i - 1) % self.numel
            basis_indices = [i_prev, i]
            
            el = Element(i, i_next, self.nodes[i], self.nodes[i_next],
                        basis_indices, self.xvec)
            self.elements.append(el)
    
    def _create_basis(self):
        """Create basis functions."""
        self.basis = []
        for i in range(self.numel):
            i_next = (i + 1) % self.numel
            b = Basis(i, i_next)
            self.basis.append(b)
    
    @staticmethod
    def _create_elliptical_nodes(numel, radius1, radius2):
        """
        Create nodes for an elliptical cross-section.
        
        Parameters:
            numel (int): Number of elements
            radius1 (float): Semi-major axis (m)
            radius2 (float): Semi-minor axis (m)
            
        Returns:
            np.ndarray: Array of node positions
        """
        angles = np.linspace(0, 2*np.pi, numel, endpoint=False)
        nodes = np.zeros((numel, 3))
        nodes[:, 0] = radius1 * np.cos(angles)
        nodes[:, 1] = radius2 * np.sin(angles)
        return nodes
    
    @classmethod
    def from_ellipse(cls, numel, radius1, radius2, quadrature=2, phi_i=0.0,
                    mode='TM', freq=5e14, mur=1.0, epsr=1.55**2, theta=np.pi/2):
        """
        Create solver for elliptical cross-section.
        
        Parameters:
            numel (int): Number of boundary elements
            radius1 (float): Semi-major axis (m)
            radius2 (float): Semi-minor axis (m)
            ... (other parameters same as __init__)
            
        Returns:
            BEMSolver: Initialized solver
        """
        nodes = cls._create_elliptical_nodes(numel, radius1, radius2)
        return cls(nodes, quadrature, phi_i, mode, freq, mur, epsr, theta)
    
    @classmethod
    def from_circle(cls, numel, radius, quadrature=2, phi_i=0.0,
                   mode='TM', freq=5e14, mur=1.0, epsr=1.55**2, theta=np.pi/2):
        """
        Create solver for circular cross-section.
        
        Parameters:
            numel (int): Number of boundary elements
            radius (float): Circle radius (m)
            ... (other parameters same as __init__)
            
        Returns:
            BEMSolver: Initialized solver
        """
        return cls.from_ellipse(numel, radius, radius, quadrature, phi_i,
                               mode, freq, mur, epsr, theta)
    
    def update_wave(self, phi_i, mode, freq, theta):
        """Update wave parameters without rebuilding geometry."""
        self.wave = Wave(self.background, phi_i, mode, freq, theta)
        self.theta = theta
        self.vvec.fill(0)
    
    def compute_incident_field(self):
        """Compute incident field excitation vector."""
        n = self.numel
        self.vvec.fill(0)
        
        # Incident wave direction and polarization
        d1 = np.array([np.sin(self.theta) * np.cos(self.wave.phi_i),
                      np.sin(self.theta) * np.sin(self.wave.phi_i),
                      np.cos(self.theta)])
        d2 = np.array([-np.cos(self.theta) * np.cos(self.wave.phi_i),
                      -np.cos(self.theta) * np.sin(self.wave.phi_i),
                      np.sin(self.theta)])
        d3 = np.cross(d2, d1)
        
        for i, el in enumerate(self.elements):
            tan = el.get_tangent()
            length = el.get_length()
            
            for j in range(2):  # Two basis functions per element
                b1 = el.get_basis_index(j)
                
                for k in range(self.quadrature):
                    qp = el.get_quadrature_point(k)
                    nor = el.get_normal(k)
                    
                    weight = length / 2 * self.weights[k]
                    phase = np.exp(1j * self.wave.k0 * np.dot(qp, d1))
                    constant = weight * phase
                    
                    eb = self.f[j, k]
                    hb = self.g[j, k]
                    
                    tmp = 1.0 / self.background.Z0
                    
                    if self.wave.mode == 'TM':
                        # E_z component
                        self.vvec[n + b1] += eb * constant * d2[2]
                        # H_t component
                        self.vvec[2*n + b1] += tmp * hb * constant * np.dot(d3, tan)
                        # E_t component
                        self.vvec[b1] += eb * constant * np.dot(d2, tan)
                        # H_z component
                        self.vvec[3*n + b1] += tmp * hb * constant * d3[2]
                    else:  # TE mode
                        # E_z component
                        self.vvec[n + b1] += eb * constant * (-d3[2])
                        # H_t component
                        self.vvec[2*n + b1] += tmp * hb * constant * np.dot(d2, tan)
                        # E_t component
                        self.vvec[b1] += eb * constant * np.dot(-d3, tan)
                        # H_z component
                        self.vvec[3*n + b1] += tmp * hb * constant * d2[2]
    
    def _compute_hankel(self, arg1, arg2):
        """
        Compute Hankel functions H0 and H1.
        
        For small arguments, uses scipy.special.hankel2.
        For larger arguments, could use pre-computed tables (not implemented here).
        
        Returns:
            tuple: (H0(arg1), H1(arg1), H0(arg2), H1(arg2))
        """
        temp1 = hankel2(0, arg1)
        temp2 = hankel2(1, arg1)
        temp3 = hankel2(0, arg2)
        temp4 = hankel2(1, arg2)
        return temp1, temp2, temp3, temp4
    
    def _assembly_same_element(self, b1, b2, f1, f2, g1, g2, tan1, tan2, nor,
                               pow1, pow2, constant, sing1, sing2, sing3, sing4):
        """
        Assemble matrix contributions for same element (singularity treatment).
        
        This handles the singular integrals when source and field points coincide.
        """
        n = self.numel
        
        # Z-Z coupling
        common1 = constant * f1
        outside = (self.wave.omega * self.background.mu0 / 4 * common1 * sing1 -
                  self.wave.kz**2 / (4 * self.wave.omega * self.background.eps0) * common1 * sing1)
        inside = (self.wave.omega * self.background.mu1 / 4 * common1 * sing3 -
                 self.wave.kz**2 / (4 * self.wave.omega * self.background.eps1) * common1 * sing3)
        
        self.Z[n+b1, n+b2] += outside + inside
        self.Z[3*n+b1, 3*n+b2] += (self.background.eps0/self.background.mu0 * outside +
                                    self.background.eps1/self.background.mu1 * inside)
        
        # Z-T and T-Z coupling
        z_vec = np.array([0, 0, 1])
        temp = f1 * np.dot(tan1, np.cross(z_vec, g2 * tan2))
        outside = self.wave.kz / 4 * constant * temp * sing2
        inside = self.wave.kz / 4 * constant * temp * sing4
        
        self.Z[b1, 2*n+b2] += outside + inside
        self.Z[2*n+b1, b2] += -(outside + inside)
        
        # T-T coupling
        common1 = self.wave.omega * constant * g1
        common2 = -1 / (4 * self.wave.omega) * constant * pow1 * pow2
        
        outside = (self.background.mu0 / 4 * common1 * sing1 * np.dot(tan1, tan2) +
                  1 / self.background.eps0 * common2 * sing2)
        inside = (self.background.mu1 / 4 * common1 * sing3 * np.dot(tan1, tan2) +
                 1 / self.background.eps1 * common2 * sing4)
        
        self.Z[b1, b2] += outside + inside
        self.Z[2*n+b1, 2*n+b2] += (self.background.eps0/self.background.mu0 * outside +
                                    self.background.eps1/self.background.mu1 * inside)
        
        # Oblique incidence specific terms
        common3 = -1j * self.wave.kz / (4 * self.wave.omega) * constant * pow1
        outside = 1 / self.background.eps0 * common3 * sing1
        inside = 1 / self.background.eps1 * common3 * sing3
        
        self.Z[b1, n+b2] += outside + inside
        self.Z[2*n+b1, 3*n+b2] += (self.background.eps0/self.background.mu0 * outside +
                                    self.background.eps1/self.background.mu1 * inside)
        
        common3 = 1j * self.wave.kz / (4 * self.wave.omega) * constant * pow2
        outside = 1 / self.background.eps0 * common3 * sing2 * f1
        inside = 1 / self.background.eps1 * common3 * sing4 * f1
        
        self.Z[n+b1, b2] += outside + inside
        self.Z[3*n+b1, 2*n+b2] += (self.background.eps0/self.background.mu0 * outside +
                                    self.background.eps1/self.background.mu1 * inside)
    
    def _assembly_different_elements(self, b1, b2, f1, f2, g1, g2,
                                     const1, const2, pow1, pow2,
                                     tan1, tan2, R, temp1, temp2, temp3, temp4):
        """
        Assemble matrix contributions for different elements.
        
        Uses Hankel functions for Green's function evaluation.
        """
        n = self.numel
        z_vec = np.array([0, 0, 1])
        
        # Z-Z coupling
        common1 = const1 * f1 * const2 * f2
        outside = (self.wave.omega * self.background.mu0 / 4 * common1 * temp1 -
                  self.wave.kz**2 / (4 * self.wave.omega * self.background.eps0) * common1 * temp1)
        inside = (self.wave.omega * self.background.mu1 / 4 * common1 * temp3 -
                 self.wave.kz**2 / (4 * self.wave.omega * self.background.eps1) * common1 * temp3)
        
        self.Z[n+b1, n+b2] += outside + inside
        self.Z[3*n+b1, 3*n+b2] += (self.background.eps0/self.background.mu0 * outside +
                                    self.background.eps1/self.background.mu1 * inside)
        
        # Z-T coupling
        c = np.cross(R, g2 * tan2)
        common1 = 1j / 4 * const1 * f1 * const2
        
        self.Z[n+b1, 2*n+b2] += common1 * (self.wave.newk * temp2 * c[2] +
                                           self.wave.newk1 * temp4 * c[2])
        self.Z[3*n+b1, b2] += -common1 * (self.wave.newk * temp2 * c[2] +
                                          self.wave.newk1 * temp4 * c[2])
        
        # T-Z coupling
        cdot = np.dot(tan1, np.cross(R, z_vec * f2))
        
        self.Z[b1, 3*n+b2] += common1 * (self.wave.newk * cdot * temp2 +
                                         self.wave.newk1 * cdot * temp4)
        self.Z[2*n+b1, n+b2] += -common1 * (self.wave.newk * cdot * temp2 +
                                            self.wave.newk1 * cdot * temp4)
        
        # Tangent-normal coupling
        cdot = np.dot(tan1, np.cross(z_vec, g2 * tan2))
        outside = self.wave.kz / 4 * const1 * f1 * const2 * cdot * temp1
        inside = self.wave.kz / 4 * const1 * f1 * const2 * cdot * temp3
        
        self.Z[b1, 2*n+b2] += outside + inside
        self.Z[2*n+b1, b2] += -(outside + inside)
        
        # T-T coupling
        common1 = const1 * g1 * const2 * g2
        common2 = -1 / (4 * self.wave.omega) * const1 * const2 * pow1 * pow2
        
        outside = (self.wave.omega * self.background.mu0 / 4 * common1 *
                  np.dot(tan1, tan2) * temp1 + 1 / self.background.eps0 * common2 * temp1)
        inside = (self.wave.omega * self.background.mu1 / 4 * common1 *
                 np.dot(tan1, tan2) * temp3 + 1 / self.background.eps1 * common2 * temp3)
        
        self.Z[b1, b2] += outside + inside
        self.Z[2*n+b1, 2*n+b2] += (self.background.eps0/self.background.mu0 * outside +
                                    self.background.eps1/self.background.mu1 * inside)
        
        # Oblique incidence coupling
        common1 = -1j * self.wave.kz / (4 * self.wave.omega) * const1 * const2 * pow1 * f2
        outside = 1 / self.background.eps0 * common1 * temp1
        inside = 1 / self.background.eps1 * common1 * temp3
        
        self.Z[b1, n+b2] += outside + inside
        self.Z[2*n+b1, 3*n+b2] += (self.background.eps0/self.background.mu0 * outside +
                                    self.background.eps1/self.background.mu1 * inside)
        
        common1 = 1j * self.wave.kz / (4 * self.wave.omega) * const1 * const2 * f1 * pow2
        outside = 1 / self.background.eps0 * common1 * temp1
        inside = 1 / self.background.eps1 * common1 * temp3
        
        self.Z[n+b1, b2] += outside + inside
        self.Z[3*n+b1, 2*n+b2] += (self.background.eps0/self.background.mu0 * outside +
                                    self.background.eps1/self.background.mu1 * inside)
    
    def assemble_system_matrix(self, verbose=False):
        """
        Assemble the system matrix Z using dense assembly.
        
        This is the core of the BEM - builds the impedance matrix relating
        surface currents to incident fields.
        """
        self.Z.fill(0)
        
        consb = 1.781 / 2.0
        cons = consb * self.wave.newk
        cons1 = consb * self.wave.newk1
        
        n = self.numel
        
        # Progress tracking
        if verbose:
            import sys
            print(f"    Assembling {n}x{n} element interactions", end='', flush=True)
        
        for i in range(n):
            # Progress indicator every 10 elements
            if verbose and i % max(1, n//10) == 0:
                print('.', end='', flush=True)
                sys.stdout.flush()
            el_i = self.elements[i]
            len1 = el_i.get_length()
            tan1 = el_i.get_tangent()
            
            for m in range(2):
                b1 = el_i.get_basis_index(m)
                pow1 = (-1)**(m+1) / len1
                
                for p in range(self.quadrature):
                    p1 = el_i.get_quadrature_point(p)
                    nor = el_i.get_normal(p)
                    constant1 = len1 / 2 * self.weights[p]
                    
                    for j in range(n):
                        el_j = self.elements[j]
                        len2 = el_j.get_length()
                        tan2 = el_j.get_tangent()
                        
                        for nn in range(2):
                            b2 = el_j.get_basis_index(nn)
                            pow2 = (-1)**(nn+1) / len2
                            
                            if i == j:  # Same element - handle singularity
                                # Compute singular integrals analytically
                                if nn == 0:
                                    xp = np.linalg.norm(p1 - self.nodes[el_i.node2])
                                else:
                                    xp = np.linalg.norm(p1 - self.nodes[el_i.node1])
                                
                                xp1 = np.linalg.norm(p1 - self.nodes[el_i.node1])
                                
                                # Analytical singular integrals
                                sing1 = (len2/2 - 2/np.pi * (xp**2/(2*len2) * np.log(xp/(len2-xp)) +
                                        len2/2 * np.log(cons*(len2-xp)) - xp/2 - len2/4) * 1j)
                                
                                sing2 = (len2 - 1j * 2/np.pi * (len2 * np.log(cons*2*(len2-xp1)/(2*np.e)) +
                                        xp1 * np.log(xp1/(len2-xp1))))
                                
                                sing3 = (len2/2 - 2/np.pi * (xp**2/(2*len2) * np.log(xp/(len2-xp)) +
                                        len2/2 * np.log(cons1*(len2-xp)) - xp/2 - len2/4) * 1j)
                                
                                sing4 = (len2 - 2/np.pi * (len2 * np.log(2*cons1*(len2-xp1)/(2*np.e)) +
                                        xp1 * np.log(xp1/(len2-xp1))) * 1j)
                                
                                self._assembly_same_element(b1, b2, self.f[m,p], self.f[nn,p],
                                                           self.g[m,p], self.g[nn,p],
                                                           tan1, tan2, nor, pow1, pow2, constant1,
                                                           sing1, sing2, sing3, sing4)
                            else:  # Different elements
                                for q in range(self.quadrature):
                                    p2 = el_j.get_quadrature_point(q)
                                    diff = p1 - p2
                                    dis = np.linalg.norm(diff)
                                    R = diff / dis
                                    constant2 = len2 / 2 * self.weights[q]
                                    
                                    # Compute Hankel functions
                                    temp1, temp2, temp3, temp4 = self._compute_hankel(
                                        self.wave.newk * dis, self.wave.newk1 * dis)
                                    
                                    self._assembly_different_elements(
                                        b1, b2, self.f[m,p], self.f[nn,q],
                                        self.g[m,p], self.g[nn,q],
                                        constant1, constant2, pow1, pow2,
                                        tan1, tan2, R, temp1, temp2, temp3, temp4)
    
    def solve(self, verbose=False):
        """
        Solve the BEM system Z * sol = vvec.
        
        Uses LU decomposition for efficient solution.
        """
        self.assemble_system_matrix(verbose=verbose)
        if verbose:
            print(" done")
        self.compute_incident_field()
        
        # Solve using LU factorization
        if verbose:
            print("    LU decomposition and solve...", end='', flush=True)
        lu, piv = lu_factor(self.Z)
        self.sol = lu_solve((lu, piv), self.vvec)
        if verbose:
            print(" done")
        
        return self.sol
    
    def compute_energy_balance(self):
        """
        Compute energy balance (absorbed + scattered power).
        
        Returns:
            float: Energy balance ratio
        """
        if self.sol is None:
            raise ValueError("Must solve system before computing energy balance")
        
        n = self.numel
        energy = 0.0
        
        for j in range(self.numel):
            el = self.elements[j]
            tanvec = el.get_tangent()
            length = el.get_length()
            nor = np.array([tanvec[1], -tanvec[0], 0])
            
            J = [np.zeros(3, dtype=complex), np.zeros(3, dtype=complex)]
            M = [np.zeros(3, dtype=complex), np.zeros(3, dtype=complex)]
            
            for k in range(2):
                b1 = el.get_basis_index(k)
                J_coef_z = self.sol[n + b1]
                J_coef_t = self.sol[b1]
                M_coef_t = self.sol[2*n + b1]
                M_coef_z = self.sol[3*n + b1]
                
                for p in range(self.quadrature):
                    z_vec = np.array([0, 0, 1])
                    Jz = J_coef_z * self.f[k, p] * z_vec
                    Mz = M_coef_z * self.g[k, p] * z_vec
                    Jt = J_coef_t * self.f[k, p] * tanvec
                    Mt = M_coef_t * self.g[k, p] * tanvec
                    
                    J[k] += Jz + Jt
                    M[k] += Mz + Mt
            
            poynting = (np.cross(J[0].conj(), M[0]).real +
                       np.cross(J[1].conj(), M[1]).real)
            energy += np.dot(nor, poynting) * length / 2 * self.weights[0]
        
        energy *= self.background.Z0 / (2 * self.radius * np.sin(self.theta))
        return energy
    
    def compute_bsdf(self, n_theta, distance):
        """
        Compute Bidirectional Scattering Distribution Function (BSDF).
        
        Parameters:
            n_theta (int): Number of azimuthal angles
            distance (float): Distance from origin to evaluate scattered field (m)
            
        Returns:
            np.ndarray: BSDF values at each angle
        """
        if self.sol is None:
            raise ValueError("Must solve system before computing BSDF")
        
        if not hasattr(self, 'wave'):
            raise AttributeError("Wave object not initialized. Check solver initialization.")
        
        n = self.numel
        sigma = np.zeros(n_theta)
        
        for i in range(n_theta):
            theta_out = i * 2 * np.pi / n_theta
            x = distance * np.cos(theta_out)
            y = distance * np.sin(theta_out)
            direction = np.array([np.cos(theta_out), np.sin(theta_out), 0])
            
            E = np.zeros(3, dtype=complex)
            H = np.zeros(3, dtype=complex)
            
            propdir = np.array([np.sin(self.theta) * direction[0],
                               np.sin(self.theta) * direction[1],
                               -np.cos(self.theta)])
            
            for j in range(self.numel):
                el = self.elements[j]
                tanvec = el.get_tangent()
                length = el.get_length()
                
                for k in range(2):
                    b1 = el.get_basis_index(k)
                    
                    J_coef_z = self.sol[n + b1]
                    J_coef_t = self.sol[b1]
                    M_coef_t = self.sol[2*n + b1]
                    M_coef_z = self.sol[3*n + b1]
                    
                    for p in range(self.quadrature):
                        p1 = el.get_quadrature_point(p)
                        diffvec = np.array([x, y, 0]) - p1
                        dis = np.linalg.norm(diffvec)
                        
                        # Asymptotic Hankel function for large distances
                        hankel = np.sqrt(2j / (np.pi * self.wave.newk * dis)) * \
                                np.exp(-1j * self.wave.newk * dis)
                        
                        common = length / 2 * self.weights[p]
                        z_vec = np.array([0, 0, 1])
                        
                        tmp = -self.wave.omega * self.background.mu0 / 4 * common * z_vec * hankel
                        E += J_coef_z * self.f[k, p] * tmp
                        H += (self.background.eps0 / self.background.mu0 *
                             M_coef_z * self.g[k, p] * tmp)
                        
                        tmp = -self.wave.omega * self.background.mu0 / 4 * common * tanvec * hankel
                        E += J_coef_t * self.f[k, p] * tmp
                        H += (self.background.eps0 / self.background.mu0 *
                             M_coef_t * self.g[k, p] * tmp)
            
            # Remove component along propagation direction
            E_s = E - np.dot(propdir, E) * propdir
            H_s = H - np.dot(propdir, H) * propdir
            E_s += np.cross(-propdir, H_s) * self.background.Z0
            
            sigma[i] = np.abs(E_s)**2 @ np.ones(3)
        
        sigma *= distance / (2 * self.radius)
        return sigma
    
    def compute_bsdf_3d(self, n_theta, n_phi, distance, verbose=False):
        """
        Compute full 3D Bidirectional Scattering Distribution Function.
        
        This computes scattering in all directions on a sphere around the cylinder.
        
        Parameters:
            n_theta (int): Number of polar angles (0 to π)
            n_phi (int): Number of azimuthal angles (0 to 2π)
            distance (float): Distance from origin to evaluate scattered field (m)
            verbose (bool): Print progress updates
            
        Returns:
            tuple: (theta_array, phi_array, sigma_3d)
                - theta_array: 1D array of polar angles
                - phi_array: 1D array of azimuthal angles  
                - sigma_3d: 2D array of shape (n_theta, n_phi) with BSDF values
        """
        if self.sol is None:
            raise ValueError("Must solve system before computing BSDF")
        
        n = self.numel
        theta_array = np.linspace(0, np.pi, n_theta)
        phi_array = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
        sigma_3d = np.zeros((n_theta, n_phi))
        
        if verbose:
            print(f"    Computing 3D BSDF: {n_theta} x {n_phi} = {n_theta*n_phi} directions", end='', flush=True)
            progress_step = max(1, n_theta // 10)
        
        for i, theta_obs in enumerate(theta_array):
            if verbose and i % progress_step == 0:
                print('.', end='', flush=True)
            
            for j, phi_obs in enumerate(phi_array):
                # Observation point in spherical coordinates
                x = distance * np.sin(theta_obs) * np.cos(phi_obs)
                y = distance * np.sin(theta_obs) * np.sin(phi_obs)
                z = distance * np.cos(theta_obs)
                
                # Observation direction (unit vector from origin to observation point)
                obs_dir = np.array([np.sin(theta_obs) * np.cos(phi_obs),
                                   np.sin(theta_obs) * np.sin(phi_obs),
                                   np.cos(theta_obs)])
                
                E = np.zeros(3, dtype=complex)
                H = np.zeros(3, dtype=complex)
                
                # Sum contributions from all surface currents
                for k in range(self.numel):
                    el = self.elements[k]
                    tanvec = el.get_tangent()
                    length = el.get_length()
                    
                    for m in range(2):
                        b1 = el.get_basis_index(m)
                        
                        J_coef_z = self.sol[n + b1]
                        J_coef_t = self.sol[b1]
                        M_coef_t = self.sol[2*n + b1]
                        M_coef_z = self.sol[3*n + b1]
                        
                        for p in range(self.quadrature):
                            p1 = el.get_quadrature_point(p)
                            diffvec = np.array([x, y, z]) - p1
                            dis = np.linalg.norm(diffvec)
                            
                            # Far-field approximation: use asymptotic Hankel function
                            hankel = np.sqrt(2j / (np.pi * self.wave.newk * dis)) * \
                                    np.exp(-1j * self.wave.newk * dis)
                            
                            common = length / 2 * self.weights[p]
                            z_vec = np.array([0, 0, 1])
                            
                            # Contributions from J_z (z-directed current)
                            tmp = -self.wave.omega * self.background.mu0 / 4 * common * z_vec * hankel
                            E += J_coef_z * self.f[m, p] * tmp
                            H += (self.background.eps0 / self.background.mu0 *
                                 M_coef_z * self.g[m, p] * tmp)
                            
                            # Contributions from J_t (tangential current)
                            tmp = -self.wave.omega * self.background.mu0 / 4 * common * tanvec * hankel
                            E += J_coef_t * self.f[m, p] * tmp
                            H += (self.background.eps0 / self.background.mu0 *
                                 M_coef_t * self.g[m, p] * tmp)
                
                # Remove component along propagation direction (far-field)
                E_s = E - np.dot(obs_dir, E) * obs_dir
                H_s = H - np.dot(obs_dir, H) * obs_dir
                E_s += np.cross(-obs_dir, H_s) * self.background.Z0
                
                # Power in scattered field
                sigma_3d[i, j] = np.abs(E_s)**2 @ np.ones(3)
        
        if verbose:
            print(" done")
        
        sigma_3d *= distance / (2 * self.radius)
        return theta_array, phi_array, sigma_3d
