"""
Boundary element class for discretizing the scatterer surface.
"""
import numpy as np

class Element:
    """
    A boundary element (line segment) on the 2D cross-section.
    
    Attributes:
        node1 (int): Index of first node
        node2 (int): Index of second node
        length (float): Length of the element
        basis (list): Indices of associated basis functions
        tanvec (np.ndarray): Tangent vector (unit vector from node1 to node2)
        quadpoints (list): Quadrature points for integration
        normals (list): Outward normal vectors at each quadrature point
    """
    
    def __init__(self, node1, node2, n1, n2, basis_indices, xvec):
        """
        Initialize a boundary element.
        
        Parameters:
            node1 (int): Index of first node
            node2 (int): Index of second node
            n1 (np.ndarray): Position of first node (3D vector)
            n2 (np.ndarray): Position of second node (3D vector)
            basis_indices (list): Indices of the two basis functions on this element
            xvec (list): Quadrature points in reference coordinates [-1, 1]
        """
        self.node1 = node1
        self.node2 = node2
        self.basis = basis_indices
        
        # Compute element properties
        diff = n2 - n1
        self.length = np.linalg.norm(diff)
        self.tanvec = diff / self.length  # Unit tangent vector
        
        # Generate quadrature points and normals
        self.quadpoints = []
        self.normals = []
        for xi in xvec:
            # Map from reference coordinates [-1, 1] to physical coordinates
            qp = n1 + (1 + xi) / 2 * (n2 - n1)
            self.quadpoints.append(qp)
            
            # Outward normal (2D: rotate tangent 90° counterclockwise)
            normal = np.array([self.tanvec[1], -self.tanvec[0], 0.0])
            self.normals.append(normal)
    
    def get_quadrature_point(self, index):
        """Get quadrature point by index."""
        return self.quadpoints[index]
    
    def get_normal(self, index):
        """Get normal vector at quadrature point."""
        return self.normals[index]
    
    def get_tangent(self):
        """Get element tangent vector."""
        return self.tanvec
    
    def get_length(self):
        """Get element length."""
        return self.length
    
    def get_basis_index(self, local_index):
        """Get global basis function index (0 or 1 for local index)."""
        return self.basis[local_index]
    
    def __repr__(self):
        return f"Element(nodes={self.node1}->{self.node2}, length={self.length:.4e})"
