from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np

# Function to create and simulate the entangling circuit
def entangled_state(a, b, c):
    qc = QuantumCircuit(3)
    if a: qc.x(0)
    if b: qc.x(1)
    if c: qc.x(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    return Statevector.from_instruction(qc)

# Function to extract the symbolic entangled form
def symbolic_form(psi):
    # Get non-zero amplitudes and their basis labels
    basis = psi.to_dict()
    non_zero = [(k, complex(v)) for k, v in basis.items() if abs(v) > 1e-9]
    if len(non_zero) == 2:
        (k1, v1), (k2, v2) = non_zero
        sign = '+' if np.isclose(v2, v1) else '−'
        return f"(1/√2)(|{k1}⟩ {sign} |{k2}⟩)"
    else:
        return psi

# Generate and print all combinations
for i in range(8):
    bits = [(i >> j) & 1 for j in reversed(range(3))]
    psi = entangled_state(*bits)
    symbolic = symbolic_form(psi)
    print(f"|{''.join(map(str, bits))}⟩ → {symbolic}")
