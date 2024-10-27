import numpy as np

class QuantumGates:
    """Common quantum gates as numpy arrays."""
    
    @staticmethod
    def I():
        """Identity gate"""
        return np.array([[1, 0], [0, 1]], dtype=complex)
    
    @staticmethod
    def X():
        """Pauli-X (NOT) gate"""
        return np.array([[0, 1], [1, 0]], dtype=complex)
    
    @staticmethod
    def H():
        """Hadamard gate"""
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    
    @staticmethod
    def CNOT():
        """Control-NOT gate"""
        return np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0]], dtype=complex)