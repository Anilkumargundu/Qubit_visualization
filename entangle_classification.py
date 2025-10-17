import numpy as np
from itertools import combinations

def tensor_with_qubit(psi_n, phi):
    """
    Tensor an N-qubit state with a single qubit |phi> to get (N+1)-qubit state
    psi_n : numpy array of shape (2**N,)
    phi   : numpy array of shape (2,) normalized
    """
    psi_n1 = np.kron(psi_n, phi)
    psi_n1 /= np.linalg.norm(psi_n1)  # normalize
    return psi_n1

def two_qubit_determinant(psi, q1, q2):
    """
    For an N-qubit state psi, compute 2x2 amplitude matrix for qubits q1 and q2
    (trace out other qubits)
    Returns determinant ad-bc
    """
    N = int(np.log2(len(psi)))
    # Reshape to N indices
    psi_tensor = psi.reshape([2]*N)
    
    # Sum out all qubits except q1, q2
    axes_to_trace = tuple(i for i in range(N) if i not in [q1, q2])
    psi_2 = np.tensordot(psi_tensor, np.ones([2]*len(axes_to_trace)), axes=(axes_to_trace, range(len(axes_to_trace))))
    
    # Flatten 2x2
    a = psi_2[0,0]
    b = psi_2[0,1]
    c = psi_2[1,0]
    d = psi_2[1,1]
    det = a*d - b*c
    return det

# -------------------------------
# Step 1: Start with 2-qubit state
psi_2 = np.array([np.sqrt(0.2), -np.sqrt(0.), np.sqrt(0.3), -np.sqrt(0.3)])  # 2-qubit state

# New qubit state |phi> = |+>
phi = np.array([1/np.sqrt(2), 1/np.sqrt(2)])

# Step 2: Extend to 3 qubits
psi_3 = tensor_with_qubit(psi_2, phi)
print("3-qubit state amplitudes:")
print(np.round(psi_3, 4))

# Step 3: Extend to 4 qubits
psi_4 = tensor_with_qubit(psi_3, phi)
print("\n4-qubit state amplitudes:")
print(np.round(psi_4, 4))

# Step 4: Check determinant for all 2-qubit combinations
N = 4
print("\n2-qubit determinants for all combinations in 4-qubit state:")
for q1, q2 in combinations(range(N), 2):
    det = two_qubit_determinant(psi_4, q1, q2)
    print(f"Determinant for qubits ({q1},{q2}): {det:.6f}")
