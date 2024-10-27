# src/utils/benchmarking.py
import time
from typing import List, Tuple
from ..simulators.naive_simulator import NaiveSimulator
from ..simulators.tensor_simulator import TensorSimulator
from ..gates import QuantumGates

def benchmark_simulators(max_qubits: int = 10) -> Tuple[List[float], List[float]]:
    """Benchmark both simulators and return their runtimes."""
    naive_times = []
    tensor_times = []
    
    for n in range(1, max_qubits + 1):
        # Naive simulator
        start_time = time.time()
        naive_sim = NaiveSimulator(n)
        naive_sim.apply_single_qubit_gate(QuantumGates.H(), 0)
        if n > 1:
            naive_sim.apply_cnot(0, 1)
        naive_times.append(time.time() - start_time)
        
        # Tensor simulator
        start_time = time.time()
        tensor_sim = TensorSimulator(n)
        tensor_sim.apply_single_qubit_gate(QuantumGates.H(), 0)
        if n > 1:
            tensor_sim.apply_cnot(0, 1)
        tensor_times.append(time.time() - start_time)
    
    return naive_times, tensor_times

# src/utils/visualization.py
import matplotlib.pyplot as plt
from typing import List

def plot_benchmark(naive_times: List[float], tensor_times: List[float]) -> None:
    """Plot benchmark results."""
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, len(naive_times) + 1), naive_times, 'o-', label='Naive Simulation')
    plt.semilogy(range(1, len(tensor_times) + 1), tensor_times, 'o-', label='Tensor Simulation')
    plt.xlabel('Number of Qubits')
    plt.ylabel('Runtime (seconds)')
    plt.title('Quantum Circuit Simulator Performance')
    plt.grid(True)
    plt.legend()
    plt.show()