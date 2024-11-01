# Quantum Circuit Simulator

A Python implementation of a quantum circuit simulator using both naive matrix multiplication and tensor-based approaches. This project was created as a solution to the QOSF Mentorship Task 1.

## Features

- Two implementation approaches:
  - Naive simulation using matrix multiplication
  - Advanced simulation using tensor operations
- Common quantum gates (I, X, H, CNOT)
- Benchmarking tools
- Visualization utilities
- Example circuits

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/quantum-simulator.git
cd quantum-simulator
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Install the package in development mode:

```bash
pip install -e .
```

## Usage

### Basic Example

```python
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
```

### Running Examples

The project includes example scripts in the `examples/` directory:

1. Basic circuit example:

```bash
python examples/basic_circuit.py
```

2. Benchmark comparison:

```bash
python examples/benchmark_comparison.py
```

## Project Structure

```
quantum_simulator/
├── src/                    # Source code
│   ├── gates.py           # Quantum gate definitions
│   ├── simulators/        # Simulator implementations
│   └── utils/             # Utility functions
├── tests/                 # Test files
├── examples/              # Example scripts
├── requirements.txt       # Project dependencies
└── setup.py              # Package configuration
```

## Running Tests

To run the tests:

```bash
pytest tests/
```

## Benchmarking

The project includes benchmarking tools to compare the performance of both simulation approaches:

```python
from quantum_simulator.src.utils.benchmarking import benchmark_simulators
from quantum_simulator.src.utils.visualization import plot_benchmark

naive_times, tensor_times = benchmark_simulators(max_qubits=10)
plot_benchmark(naive_times, tensor_times)
```

## Authors

Kazi Muktadir Ahmed

## Acknowledgments

- QOSF Mentorship Program
- Quantum Computing Community
