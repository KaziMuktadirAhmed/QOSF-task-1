# examples/basic_circuit.py
from quantum_simulator.src.simulators.naive_simulator import NaiveSimulator
from quantum_simulator.src.simulators.tensor_simulator import TensorSimulator
from quantum_simulator.src.gates import QuantumGates

def main():
    # Create a 2-qubit system
    naive_sim = NaiveSimulator(2)
    tensor_sim = TensorSimulator(2)
    
    # Apply Hadamard to first qubit
    naive_sim.apply_single_qubit_gate(QuantumGates.H(), 0)
    tensor_sim.apply_single_qubit_gate(QuantumGates.H(), 0)
    
    # Apply CNOT between qubits
    naive_sim.apply_cnot(0, 1)
    tensor_sim.apply_cnot(0, 1)
    
    print("Naive simulator final state:", naive_sim.state)
    print("Tensor simulator final state:", tensor_sim.state.flatten())

if __name__ == "__main__":
    main()

# examples/benchmark_comparison.py
from quantum_simulator.src.utils.benchmarking import benchmark_simulators
from quantum_simulator.src.utils.visualization import plot_benchmark

def main():
    naive_times, tensor_times = benchmark_simulators(max_qubits=10)
    plot_benchmark(naive_times, tensor_times)

if __name__ == "__main__":
    main()