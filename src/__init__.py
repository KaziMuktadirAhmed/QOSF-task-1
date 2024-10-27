from quantum_simulator.src.simulators.naive_simulator import NaiveSimulator
from quantum_simulator.src.gates import QuantumGates

# Create a 2-qubit system
sim = NaiveSimulator(2)

# Apply Hadamard gate to first qubit
sim.apply_single_qubit_gate(QuantumGates.H(), 0)

# Apply CNOT between qubits 0 and 1
sim.apply_cnot(0, 1)

# Print final state
print(sim.state)