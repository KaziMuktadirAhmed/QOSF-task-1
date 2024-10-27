# src/simulators/base.py
from abc import ABC, abstractmethod
import numpy as np

class QuantumSimulator(ABC):
    """Abstract base class for quantum simulators."""
    
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
    
    @abstractmethod
    def apply_single_qubit_gate(self, gate: np.ndarray, target_qubit: int) -> None:
        pass
    
    @abstractmethod
    def apply_cnot(self, control: int, target: int) -> None:
        pass

# src/simulators/naive_simulator.py
import numpy as np
from .base import QuantumSimulator
from ..gates import QuantumGates

class NaiveSimulator(QuantumSimulator):
    """Quantum circuit simulator using naive matrix multiplication."""
    
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        self.state = np.zeros(2**num_qubits, dtype=complex)
        self.state[0] = 1
    
    def apply_single_qubit_gate(self, gate: np.ndarray, target_qubit: int) -> None:
        """Apply a single qubit gate to the target qubit."""
        op = np.array([[1]])
        for i in range(self.num_qubits):
            if i == target_qubit:
                op = np.kron(op, gate)
            else:
                op = np.kron(op, QuantumGates.I())
        self.state = op @ self.state
    
    def apply_cnot(self, control: int, target: int) -> None:
        """Apply CNOT gate between control and target qubits."""
        if control >= self.num_qubits or target >= self.num_qubits:
            raise ValueError("Qubit indices out of range")
            
        perm = list(range(self.num_qubits))
        if abs(control - target) > 1:
            perm.remove(target)
            perm.insert(control + 1, target)
            
        shape = [2] * self.num_qubits
        state_tensor = self.state.reshape(shape)
        state_tensor = np.transpose(state_tensor, perm)
        
        cnot = QuantumGates.CNOT()
        state_tensor = state_tensor.reshape(-1, 4, -1)
        for i in range(state_tensor.shape[0]):
            for j in range(state_tensor.shape[2]):
                state_tensor[i, :, j] = cnot @ state_tensor[i, :, j]
                
        state_tensor = state_tensor.reshape([2] * self.num_qubits)
        inv_perm = [perm.index(i) for i in range(len(perm))]
        state_tensor = np.transpose(state_tensor, inv_perm)
        self.state = state_tensor.flatten()

# src/simulators/tensor_simulator.py
import numpy as np
from .base import QuantumSimulator
from ..gates import QuantumGates

class TensorSimulator(QuantumSimulator):
    """Quantum circuit simulator using tensor operations."""
    
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        shape = [2] * num_qubits
        self.state = np.zeros(shape, dtype=complex)
        self.state[tuple([0] * num_qubits)] = 1
    
    def apply_single_qubit_gate(self, gate: np.ndarray, target_qubit: int) -> None:
        """Apply a single qubit gate to the target qubit."""
        self.state = np.tensordot(gate, self.state, axes=([1], [target_qubit]))
        axes = list(range(self.num_qubits + 1))
        axes.remove(0)
        axes.insert(target_qubit, 0)
        self.state = np.transpose(self.state, axes)
    
    def apply_cnot(self, control: int, target: int) -> None:
        """Apply CNOT gate between control and target qubits."""
        if control >= self.num_qubits or target >= self.num_qubits:
            raise ValueError("Qubit indices out of range")
            
        shape = list(self.state.shape)
        perm = list(range(self.num_qubits))
        if abs(control - target) > 1:
            perm.remove(target)
            perm.insert(control + 1, target)
            self.state = np.transpose(self.state, perm)
            
        cnot = QuantumGates.CNOT()
        reshaped_state = self.state.reshape(-1, 4, -1)
        for i in range(reshaped_state.shape[0]):
            for j in range(reshaped_state.shape[2]):
                reshaped_state[i, :, j] = cnot @ reshaped_state[i, :, j]
                
        self.state = reshaped_state.reshape(shape)
        if abs(control - target) > 1:
            inv_perm = [perm.index(i) for i in range(len(perm))]
            self.state = np.transpose(self.state, inv_perm)