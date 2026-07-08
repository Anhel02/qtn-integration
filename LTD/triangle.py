import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import numpy as np

import pennylane as qml
import jax
jax.config.update('jax_enable_x64', True)

from scipy.integrate import quad

import warnings
warnings.filterwarnings('ignore')


class Triangle:

    def __init__(self,m):

        self.m = m
        self.i0 = 1e-10j


    def _loopMom(self, z):

        l = self.m * z / (1 - z) # define loop momentum in terms of z
        J = self.m / (1-z)**2 # Jacobian
        return l, J


    def _deformed_loopMom(self, z, sqrt_s, eta=0.4, width=15.0):
        l_real = self.m * z / (1 - z)
        J = self.m / (1 - z)**2

        l_sing = np.sqrt((sqrt_s / 2)**2 - self.m**2)

        gaussian = np.exp(-((l_real - l_sing)**2) / (width**2))
        y = eta * l_real * gaussian

        dy_dl = eta * gaussian * (1 - l_real * 2 * (l_real - l_sing) / width**2)

        l_complex = l_real - 1j * y
        J_complex = (1 - 1j * dy_dl) * J

        return l_complex, J_complex

    def integrand(self, z, cos_theta, sqrt_s):
        z = np.clip(z, 0., 1.0-1e-10) # z ∈ [0,1)

        # External momenta energies
        p12 = sqrt_s # p12,0 = p1,0 + p2,0 = sqrt(s)
        p1 = sqrt_s / 2 # assuming same momenta for both particles
        p2 = sqrt_s / 2

        if sqrt_s >= 2*self.m:
            l, J = self._deformed_loopMom(z, sqrt_s)
        else:
            l, J = self._loopMom(z, )

        # (l+p1)^2 = l^2 + p^2 + 2*l*p*cos(theta)
        l_plus_p1_sq = l**2 + p1**2 + 2 * l * p1 * cos_theta

        # on-shell energies
        q_1_plus = np.sqrt(l_plus_p1_sq + self.m**2 - self.i0)
        q_i_plus = np.sqrt(l**2 + self.m**2 - self.i0)

        # causal propagators
        lambda_12_plus = q_1_plus + q_i_plus + p2
        lambda_23_plus = q_i_plus + q_i_plus - p12
        lambda_31_plus = q_i_plus + q_1_plus + p1

        lambda_12_minus = q_1_plus + q_i_plus - p2
        lambda_23_minus = q_i_plus + q_i_plus + p12
        lambda_31_minus = q_i_plus + q_1_plus - p1

        x3 = 8 * q_1_plus * q_i_plus**2

        # phase term
        phase_space = (l**2) / (2 * np.pi**2)

        # integrand
        I = phase_space * (-1 / x3) * (1 / (lambda_12_minus * lambda_23_plus) + 1 / (lambda_23_minus * lambda_31_plus) + 1 / (lambda_31_minus * lambda_12_plus) + 1 / (lambda_12_plus * lambda_23_minus) + 1 / (lambda_23_plus * lambda_31_minus) + 1 / (lambda_31_plus * lambda_12_minus))

        return I * J

def quantization(x, n):
    Delta = (np.max(x)-np.min(x))/(2**n)
    # we can implement the ⌊(x-min(x))/Delta⌋ by just generating the exact bin indices directly with numpy.arange()
    return np.min(x) + Delta*(np.arange(2**n) + 1/2)

def remaping(grid, method='interleaving'):
    p = []
    n_bits = int(np.ceil(np.log2(len(grid)))) if len(grid) > 1 else 1

    rows = len(grid)
    cols = len(grid[0])

    if method == 'sequential':
        for i in range(rows):
            for j in range(cols):
                p.append(grid[i][j])

    elif method == 'mirroring':

        for j in range(rows):
            pi = []
            for l in range(cols):
                j_inv = f"{j:0{n_bits}b}"[::-1]
                l_bin = f"{l:0{n_bits}b}"

                pi.append(grid[int(j_inv,2)][int(l_bin,n_bits)])

            p.append(pi)

    elif method == 'interleaving':
        entries = rows * cols

        for i in range(entries):
            bin_i = f"{i:0{2 * n_bits}b}"

            j_bin = bin_i[0::2]
            l_bin = bin_i[1::2]

            p.append(grid[int(j_bin, 2)][int(l_bin, 2)])

    return p

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
            rk = max(1, rk - 1)

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

def run(diagram, n, rel_acc, sqrt_s, shots, z_min, z_max, cos_min, cos_max, transpile, show_qc, show_specs):
    V = (z_max - z_min)*(cos_max - cos_min) # Integration volume
 
    # Original distribution
    Z = np.linspace(z_min,z_max,1000)
    COS = np.linspace(cos_min,cos_max,1000)
    grid = diagram.integrand(Z,COS,sqrt_s)

    # Quantized distribution
    n_per_q = int(n / 2)
    z_q = quantization(Z, n_per_q) # Midrise quantization
    cos_q = quantization(COS, n_per_q)
    Z_q, COS_q = np.meshgrid(z_q, cos_q, indexing='ij')

    f = diagram.integrand(Z_q, COS_q, sqrt_s)
    f = remaping(f, method='interleaving')

    p = sampling(f) # Importance sampling
    P = np.sqrt(p) # Born's rule
    w = weight_func(f, 2**n) # Sampling weight function

    B = TT_SVD(P,n,rel_acc)
    W, C = TT_QC(B)

    p_q = run_qc(W, C, n, shots, transpile, show_qc, show_specs)

    # Final Estimation and Errors
    estimator = V * np.sum(p_q * f * w)

    var_real = 1 / (shots - 1) * (V**2 * np.sum(p_q * (np.asarray(f).real * w)**2) - estimator.real**2)
    var_imag = 1 / (shots - 1) * (V**2 * np.sum(p_q * (np.asarray(f).imag * w)**2) - estimator.imag**2)

    std_real = np.sqrt(max(0.0, var_real))
    std_imag = np.sqrt(max(0.0, var_imag))

    sys_real = rel_acc * abs(estimator.real)
    sys_imag = rel_acc * abs(estimator.imag)

    total_std_real = np.sqrt(std_real**2 + sys_real**2)
    total_std_imag = np.sqrt(std_imag**2 + sys_imag**2)

    total_std = total_std_real + 1j * total_std_imag

    return estimator, total_std


if __name__ == '__main__':

    ######## Algorithm Parameters #########

    n = 6 # number of qubits (multiple of 2)
    rel_acc = 0.0001 # relative accuracy (TT_SVD truncation) [0: No Truncation]
    shots = 1000
    z_min = 0.
    z_max = 1.
    cos_min = -1.
    cos_max = 1.

    # Diagram

    sqrt_s = 2.
    p = sqrt_s / 2
    m = 0.5 * sqrt_s

    # Flags

    transpile = 0
    show_qc = 0
    show_specs = 0

    #######################################

    diagram = Triangle(m / sqrt_s)

    estimator, std = run(diagram, n, rel_acc, sqrt_s, shots, z_min, z_max, cos_min, cos_max, transpile, show_qc, show_specs)

    print('Estimated integral values:')
    print('\tReal part:', estimator.real,'±', std.real)
    print('\tImaginary part:', estimator.imag,'±', std.imag)
