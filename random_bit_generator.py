from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

def quantum_random_bits(num_bits=8):
    """
    Generate num_bits random bits using Hadamard gates + measurement.
    """
    # Create a quantum circuit with num_bits qubits
    qc = QuantumCircuit(num_bits, num_bits)

    # Apply Hadamard to each qubit (puts into superposition)
    qc.h(range(num_bits))

    # Measure all qubits
    qc.measure(range(num_bits), range(num_bits))

    # Use Aer simulator
    sim = AerSimulator()
    compiled = transpile(qc, sim)

    # Run once (shots=1 gives one random bit string)
    result = sim.run(compiled, shots=1).result()

    # Extract the measured bitstring
    bitstring = list(result.get_counts().keys())[0]

    # Reverse because qiskit returns classical bits in little-endian order
    return bitstring[::-1]

# Example: generate 16 random bits
if __name__ == "__main__":
    bits = quantum_random_bits(16)
    print("Random bitstring:", bits)
