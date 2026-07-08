import matplotlib.pyplot as plt

import numpy as np

import pennylane as qml
import jax
jax.config.update('jax_enable_x64', True)

from scipy.integrate import quad

import warnings
warnings.filterwarnings('ignore')


def integrand(X, eps=1e-30):
    return np.sin(1/(X+eps))**2

def quantization(x, n):
    Delta = (np.max(x)-np.min(x))/(2**n)
    # we can implement the ⌊(x-min(x))/Delta⌋ by just generating the exact bin indices directly with numpy.arange
    return np.min(x) + Delta*(np.arange(2**n) + 1/2)

def sampling(f, eps=1e-30):
    return (np.abs(np.asarray(f)) + eps) / np.sum(np.abs(np.asarray(f)) + eps) # we add eps for avoiding divergences

def weight_func(f, M, eps=1e-30):
    return np.sum(np.abs(np.asarray(f)) + eps) / (M * (np.abs(np.asarray(f)) + eps))

def TT_SVD(P, n, rel_acc):
    B = [] # Low-rank approximation tensor

    # Reshape sqrt probability vector P into a 2x...x2 tensor A
    A = P.reshape([2]*n)
    d = A.ndim

    # Boundary conditions
    r0 = 1
    rd = 1
    r = r0 # we initialize the rank as r0

    # Truncation parameter
    delta = rel_acc / np.sqrt(d - 1) * np.linalg.norm(A)

    for _ in range(d-1):

        # unfolding
        M = A.reshape(2 * r, -1)

        # Compute δ-Truncated SVD
        U, S, Vh = np.linalg.svd(M, full_matrices=False)
        rk = len(S)
        err = 0

        # Truncation (Eckart-Young-Mirksy)
        # ε = √σ
        # Truncate when ε > δ
        for i in reversed(range(len(S))):
            err += S[i]**2
            if err > delta**2:
                break
            rk -= 1

        U = U[:,:rk]
        S = S[:rk]
        Vh = Vh[:rk,:]

        # Save TT-core and update next core
        B.append(U.reshape(r, 2, rk))
        A = np.matmul(np.diag(S), Vh[:len(S), :])
        r = rk

    # Last matrix
    B.append(A.reshape(r, 2, 1))
    return B

def padding(U):
    dim = U.shape[0]
    pad_dim = int(2**np.ceil(np.log2(dim)))

    if pad_dim == dim:
        return U

    # Create an identity matrix of padding dimensiona
    # Add the original U in the upper-left corner
    U_padded = np.eye(pad_dim, dtype=U.dtype)
    U_padded[:dim, :dim] = U
    return U_padded

def TT_QC(G):
    W = [] # list of unitary gates

    for i in range(len(G)-1):
        M = G[i].reshape(G[i].shape[0] * G[i].shape[1], G[i].shape[2])

        U, S, Vh = np.linalg.svd(M)

        W.append(padding(U)) # apply zero-padding to U to get a parametrizable gate
        R = np.matmul(np.diag(S), Vh)
        G[i+1] = np.tensordot(R, G[i+1], axes=([1],[0]))

    # Last TT-core

    M = G[-1].reshape(G[-1].shape[0] * G[-1].shape[1], G[-1].shape[2])

    U, S, Vh = np.linalg.svd(M, full_matrices=True)
    W.append(padding(U))

    # Normalization Coefficient (global phase)
    C = np.matmul(np.diag(S), Vh)[0][0]

    return W, C

def run_qc(W_gates, C, n_qubits, shots, transpile, show_qc, show_specs):
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.set_shots(shots)
    @qml.qnode(dev, interface='jax')
    def qc():
        # Normalization coefficient as a global phase (not needed)
        # qml.GlobalPhase(C)

        for i, gate in enumerate(reversed(W_gates)):

            dim = gate.shape
            n = int(np.log2(dim[0]))

            start_qubit = len(W_gates) - n - i
            qubits = tuple(range(start_qubit, start_qubit + n))

            # Custom unitary gates labeling stuff
            def custom_label(self, decimals=None, base_label=None, cache=None):
                return self.__class__.__name__
            gate_name = rf'$W_{{{len(W_gates) - i - 1}}}$'

            W = type(gate_name, (qml.QubitUnitary,), {'label': custom_label})
            W(np.array(gate, dtype=complex), wires=qubits)

        return qml.probs(wires=[i for i in range(n_qubits)])

    if transpile:
        qc = qml.compile(basis_set=['CZ', 'U3'])(qc)

    probs = qc()

    if show_qc:
        qml.draw_mpl(qc)()
        plt.show()

    if show_specs:
        print(qml.specs(qc,level=None, compute_depth=True)())

    return np.asarray(probs, dtype=float)

def run(n, rel_acc, shots, x_min, x_max, transpile, show_qc, show_specs):
    V = x_max - x_min # Integration volume
 
    # Original distribution
    X = np.linspace(x_min,x_max,1000)
    Y = integrand(X)

    # Quantized distribution
    z = quantization(X,n) # Midrise quantization
    f = integrand(z)

    p = sampling(f) # Importance sampling
    P = np.sqrt(p) # Born's rule
    w = weight_func(f, 2**n) # Sampling weight function

    B = TT_SVD(P,n,rel_acc)
    W, C = TT_QC(B)

    p_q = run_qc(W, C, n, shots, transpile, show_qc, show_specs)

    estimator = V * np.sum(p_q * f * w)
    var = 1 / (shots - 1) * (V**2 * np.sum(p_q * (f * w)**2) - estimator**2)
    std = np.sqrt(max(0.0, var))
    trunc_err = rel_acc * abs(estimator)

    total_std = np.sqrt(std**2 + trunc_err**2)

    return estimator, total_std


if __name__ == '__main__':

    ######## Algorithm Parameters #########

    n = 15 # number of qubits
    rel_acc = .5 # relative accuracy (TT_SVD truncation) [0: No Truncation]
    shots = 1000
    x_min = -100.
    x_max = 100.

    # Flags

    transpile = 0
    show_qc = 0
    show_specs = 0

    #######################################

    estimator, std = run(n, rel_acc, shots, x_min, x_max, transpile, show_qc, show_specs)

    analytical, analytical_std = quad(lambda x: integrand(x), x_min, x_max)

    print('Estimated integral value:', estimator,'±', std)
    print('Analytical integral value:', analytical,'±', analytical_std)
    print(f'Difference: {abs(estimator - analytical):.16e}')
