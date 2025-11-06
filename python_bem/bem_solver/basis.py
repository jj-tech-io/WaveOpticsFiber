"""
Basis function class for BEM discretization.
"""

class Basis:
    """
    Basis function associated with an edge between two elements.
    
    Each basis function is associated with a node and has support
    on the two adjacent elements sharing that node.
    
    Attributes:
        elements (list): Indices of the two elements supporting this basis function
    """
    
    def __init__(self, element1, element2):
        """
        Initialize a basis function.
        
        Parameters:
            element1 (int): Index of first element
            element2 (int): Index of second element
        """
        self.elements = [element1, element2]
    
    def __repr__(self):
        return f"Basis(elements={self.elements})"
