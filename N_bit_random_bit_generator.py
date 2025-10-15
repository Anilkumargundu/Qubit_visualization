from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt

import qiskit
import qiskit_aer
print("Qiskit version:", qiskit.__version__)
print("Qiskit Aer version:", qiskit_aer.__version__)

def quantum_random_numbers(num_numbers=16, N=8):
    """
    Generate multiple quantum random N-bit numbers using Hadamard gates + measurement.
    
    Parameters:
        num_numbers (int): How many N-bit random numbers to generate.
        N (int): Number of qubits / bits per number.
    
    Returns:
        List of strings: Each string is a random N-bit number.
    """
    # Create quantum circuit
    qc = QuantumCircuit(N, N)
    
    # Apply Hadamard to all qubits
    qc.h(range(N))
    
    # Measure all qubits
    qc.measure(range(N), range(N))
    
    # Use Aer simulator
    sim = AerSimulator()
    compiled = transpile(qc, sim)
    
    random_numbers = []
    
    for _ in range(num_numbers):
        result = sim.run(compiled, shots=1).result()
        bitstring = list(result.get_counts().keys())[0]
        random_numbers.append(bitstring[::-1])  # Reverse for correct order
    
    return random_numbers

# ------------------ Example Usage ------------------ #
if __name__ == "__main__":
    N = 8
    num_numbers = 500  # Generate 1000 random numbers
    numbers_bin = quantum_random_numbers(num_numbers=num_numbers, N=N)
    
    # Convert binary strings to decimal
    numbers_dec = [int(b, 2) for b in numbers_bin]
    
    # Print first 20 numbers as example
    print("First 20 generated random numbers (decimal):")
    print(numbers_dec[:20])
    
    # Plot histogram of the whole random subspace
    plt.figure(figsize=(8,5))
    plt.hist(numbers_dec, bins=2**N, range=(0, 2**N-1), color='skyblue', edgecolor='k')
    plt.xlabel(f"{N}-bit Random Number (Decimal)")
    plt.ylabel("Counts")
    plt.title(f"Histogram of {num_numbers} {N}-bit Quantum Random Numbers")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    #print("Qiskit IBMQ Provider version:", qiskit_ibmq_provider.__version__)
    #print("Qiskit Quantum Info version:", qiskit_quantum_info.__version__)


